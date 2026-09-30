Binary toxic/insult text classification for "Detecting Insults in Social Commentary"; optimize ROC-AUC (higher is better). The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

## Approach

Use a two-track solution and blend by out-of-fold ROC-AUC:

1. Build a strong sparse text model ensemble with word TF-IDF, char TF-IDF, case-sensitive word TF-IDF, and small handcrafted insult/style features.
2. Train several linear classifiers under 5-fold stratified CV:
   - `LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced")`
   - `LogisticRegression(C=3.0, solver="liblinear", class_weight="balanced")`
   - `LogisticRegression(C=0.8, penalty="l1", solver="liblinear", class_weight="balanced")`
   - `CalibratedClassifierCV(LinearSVC(C=0.35, class_weight="balanced"), method="sigmoid", cv=3)`
   - `CalibratedClassifierCV(RidgeClassifier(alpha=2.0, class_weight="balanced"), method="sigmoid", cv=3)`
3. Search convex blend weights on the sparse models using OOF predictions and ROC-AUC.
4. Train a transformer, `microsoft/deberta-v3-large`, with 5-fold stratified CV:
   - epochs: `3`
   - max length: `256`
   - train batch size: `16`
   - eval batch size: `64`
   - learning rate: `1.5e-5`
   - weight decay: `0.01`
   - warmup ratio: `0.08`
   - dropout: `0.2`
   - loss: `BCEWithLogitsLoss(pos_weight=negative/positive)`
   - optimizer: `AdamW`
   - scheduler: cosine schedule with warmup
   - mixed precision: CUDA `bfloat16`
5. Blend the TF-IDF ensemble and DeBERTa OOF/test predictions. Try raw probability, rank-normalized, and logit-space blends over weights `0.000 ... 1.000`; choose by OOF ROC-AUC.
6. Submit clipped probabilities in the exact sample submission format.

This dataset is small, around 3.9k training rows, so reliable CV, probability ranking, and complementary text representations matter more than complicated training infrastructure.

## Data Handling

The winning code used only:

- `train.csv`
- `test.csv`
- `sample_submission_null.csv`
- target column: `Insult`
- text column: `Comment`
- submission column: `Insult`

Always decode escaped text and HTML entities before vectorization or transformer tokenization. Do not aggressively strip punctuation, quotes, repeated characters, or case information; they are useful insult signals.

```python
import html
import re
import pandas as pd

HEX_RE = re.compile(r"\\x([0-9a-fA-F]{2})")
UNI_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
WS_RE = re.compile(r"\s+")

def decode_escapes(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = html.unescape(text)
    text = HEX_RE.sub(lambda m: chr(int(m.group(1), 16)), text)
    text = UNI_RE.sub(lambda m: chr(int(m.group(1), 16)), text)
    text = text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    text = html.unescape(text)
    return WS_RE.sub(" ", text).strip()
```

## Sparse Model Track

The sparse track is the most important and fastest part of the solution. It is already very strong on its own when done correctly.

Use three complementary vectorizers:

- Lowercased word ngrams, `1..3`
- Character word-boundary ngrams, `2..6`
- Case-sensitive word ngrams, `1..2`

Add compact manual features capturing length, punctuation, uppercase ratio, repeated characters, profanity counts, second-person pronouns, and URLs.

```python
import re
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

BAD_WORDS = [
    "fuck", "fucking", "fucker", "shit", "bitch", "idiot", "moron", "stupid",
    "retard", "dumb", "asshole", "suck", "loser", "jerk", "crap", "douche",
    "ignorant", "racist", "nazi", "trash", "scum", "pathetic", "kill",
]

def manual_features(texts):
    rows = []
    bad_re = re.compile(r"\b(" + "|".join(map(re.escape, BAD_WORDS)) + r")\w*\b", re.I)

    for txt in texts:
        s = txt if isinstance(txt, str) else ""
        low = s.lower()
        letters = [c for c in s if c.isalpha()]
        words = re.findall(r"[a-zA-Z']+", s)
        n_words = max(len(words), 1)

        rows.append([
            len(s),
            n_words,
            len(set(w.lower() for w in words)) / n_words,
            sum(c.isupper() for c in letters) / max(len(letters), 1),
            s.count("!"),
            s.count("?"),
            s.count("."),
            s.count("@"),
            s.count('"') + s.count("'"),
            sum(c.isdigit() for c in s),
            len(re.findall(r"(.)\1{2,}", low)),
            len(bad_re.findall(low)),
            int("you " in low or low.startswith("you")),
            int("your " in low),
            int("http" in low or "www." in low),
        ])

    return np.asarray(rows, dtype=np.float32)

def build_sparse_features(train_text, test_text):
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w[\w']+\b|[!?]+",
        max_features=120000,
    )

    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 6),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        strip_accents="unicode",
        max_features=180000,
    )

    word_case = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=False,
        min_df=1,
        max_df=0.98,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w[\w']+\b|[!?]+",
        max_features=80000,
    )

    Xw = word.fit_transform(train_text)
    Xc = char.fit_transform(train_text)
    Xu = word_case.fit_transform(train_text)

    Tw = word.transform(test_text)
    Tc = char.transform(test_text)
    Tu = word_case.transform(test_text)

    scaler = StandardScaler()
    mf_train = scaler.fit_transform(manual_features(train_text))
    mf_test = scaler.transform(manual_features(test_text))

    X = hstack([Xw, Xc, Xu, csr_matrix(mf_train)], format="csr")
    T = hstack([Tw, Tc, Tu, csr_matrix(mf_test)], format="csr")
    return X, T
```

Use stratified folds and store both OOF and test predictions per model. The final blend should be selected using OOF ROC-AUC, not public leaderboard probing.

```python
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

SEED = 42

def fit_sparse_cv(X, T, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    specs = [
        ("lr_c1", lambda: LogisticRegression(
            C=1.0, solver="liblinear", class_weight="balanced",
            max_iter=3000, random_state=SEED,
        )),
        ("lr_c3", lambda: LogisticRegression(
            C=3.0, solver="liblinear", class_weight="balanced",
            max_iter=3000, random_state=SEED + 1,
        )),
        ("lr_l1", lambda: LogisticRegression(
            C=0.8, penalty="l1", solver="liblinear", class_weight="balanced",
            max_iter=3000, random_state=SEED + 2,
        )),
        ("svc_cal", lambda: CalibratedClassifierCV(
            estimator=LinearSVC(
                C=0.35, class_weight="balanced",
                random_state=SEED, max_iter=5000,
            ),
            method="sigmoid",
            cv=3,
        )),
        ("ridge_cal", lambda: CalibratedClassifierCV(
            estimator=RidgeClassifier(
                alpha=2.0, class_weight="balanced", random_state=SEED,
            ),
            method="sigmoid",
            cv=3,
        )),
    ]

    model_oofs = {}
    model_tests = {}

    for name, make_model in specs:
        oof = np.zeros(len(y), dtype=np.float64)
        test_folds = []

        for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
            model = make_model()
            model.fit(X[tr_idx], y[tr_idx])

            if hasattr(model, "predict_proba"):
                p_va = model.predict_proba(X[va_idx])[:, 1]
                p_te = model.predict_proba(T)[:, 1]
            else:
                d_va = model.decision_function(X[va_idx])
                d_te = model.decision_function(T)
                p_va = 1.0 / (1.0 + np.exp(-d_va))
                p_te = 1.0 / (1.0 + np.exp(-d_te))

            oof[va_idx] = p_va
            test_folds.append(p_te)

        model_oofs[name] = oof
        model_tests[name] = np.mean(test_folds, axis=0)
        print(name, roc_auc_score(y, oof))

    return model_oofs, model_tests

def search_convex_weights(model_oofs, y, trials=20000):
    names = list(model_oofs)
    mat = np.vstack([model_oofs[n] for n in names])
    rng = np.random.default_rng(SEED)

    candidates = []

    for i in range(len(names)):
        w = np.zeros(len(names))
        w[i] = 1.0
        candidates.append(w)

    candidates.append(np.ones(len(names)) / len(names))

    for _ in range(trials):
        candidates.append(rng.dirichlet(np.ones(len(names))))

    best_auc = -1.0
    best_w = None

    for w in candidates:
        pred = np.average(mat, axis=0, weights=w)
        auc = roc_auc_score(y, pred)
        if auc > best_auc:
            best_auc = auc
            best_w = w.copy()

    return names, best_w, best_auc
```

## Transformer Track

Use DeBERTa as a complementary ranker. It is slower and less deterministic than the sparse track, but the final blend can improve robustness if selected by OOF AUC.

The winning transformer used the CLS token from `microsoft/deberta-v3-large`, a single linear head, dropout `0.2`, weighted BCE loss, and 5-fold CV.

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class InsultModel(nn.Module):
    def __init__(self, name="microsoft/deberta-v3-large", dropout=0.2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name, torch_dtype=torch.float32)

        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        cls = out.last_hidden_state[:, 0]
        return self.head(self.dropout(cls)).squeeze(-1)
```

Use dynamic padding inside the collator instead of padding everything to max length globally.

```python
import torch
from torch.utils.data import Dataset

class CommentDS(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.labels = labels
        self.enc = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_len,
            padding=False,
            add_special_tokens=True,
        )

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, idx):
        item = {
            k: torch.tensor(v[idx], dtype=torch.long)
            for k, v in self.enc.items()
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

def collate(batch, pad_id):
    keys = [k for k in batch[0] if k != "labels"]
    max_len = max(len(x["input_ids"]) for x in batch)

    out = {}

    for key in keys:
        fill = pad_id if key == "input_ids" else 0
        vals = []

        for item in batch:
            v = item[key]
            if len(v) < max_len:
                v = torch.cat([
                    v,
                    torch.full((max_len - len(v),), fill, dtype=torch.long),
                ])
            vals.append(v)

        out[key] = torch.stack(vals)

    if "labels" in batch[0]:
        out["labels"] = torch.stack([x["labels"] for x in batch])

    return out
```

Training skeleton:

```python
from sklearn.metrics import roc_auc_score
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW

def train_one_fold(model, train_loader, valid_loader, test_loader, y_train_fold, device, args):
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    grouped = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    opt = AdamW(grouped, lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    sched = get_cosine_schedule_with_warmup(
        opt,
        int(total_steps * args.warmup),
        total_steps,
    )

    pos_weight = torch.tensor(
        [(len(y_train_fold) - y_train_fold.sum()) / y_train_fold.sum()],
        device=device,
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = -1.0
    best_valid = None
    best_test = None

    for epoch in range(1, args.epochs + 1):
        model.train()

        for batch in train_loader:
            labels = batch.pop("labels").to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch)
                loss = loss_fn(logits.float(), labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

        valid_pred = infer_sigmoid(model, valid_loader, device)
        auc = roc_auc_score(y_valid_fold, valid_pred)

        if auc > best_auc:
            best_auc = auc
            best_valid = valid_pred
            best_test = infer_sigmoid(model, test_loader, device)

    return best_valid, best_test, best_auc
```

Inference skeleton:

```python
import numpy as np
import torch

def infer_sigmoid(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
                if k != "labels"
            }

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch).float()

            preds.append(torch.sigmoid(logits).detach().cpu().numpy())

    return np.concatenate(preds)
```

Important transformer settings:

```python
args.model = "microsoft/deberta-v3-large"
args.folds = 5
args.epochs = 3
args.max_len = 256
args.batch_size = 16
args.eval_batch_size = 64
args.lr = 1.5e-5
args.weight_decay = 0.01
args.warmup = 0.08
args.dropout = 0.2
args.num_workers = 4
```

Do not run the transformer on CPU. The winning code explicitly asserted CUDA availability.

```python
assert torch.cuda.is_available(), "CUDA unavailable; transformer training must not run on CPU"
```

## Final Blending

Blend using OOF predictions only. Try raw probability blending, rank blending, and logit blending between the best sparse blend and DeBERTa.

```python
import numpy as np
from scipy.special import logit, expit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

def rank01(x):
    return (rankdata(x, method="average") - 1) / (len(x) - 1)

def choose_final_blend(y, tf_oofs, tf_tests, tf_weights, deb_oof, deb_test):
    tf_blend_oof = np.average(tf_oofs, axis=0, weights=tf_weights)
    tf_blend_test = np.average(tf_tests, axis=0, weights=tf_weights)

    best = None
    rows = []

    blend_modes = [
        ("raw", deb_oof, deb_test, tf_blend_oof, tf_blend_test),
        ("rank", rank01(deb_oof), rank01(deb_test), tf_blend_oof, tf_blend_test),
        (
            "logit",
            logit(np.clip(deb_oof, 1e-5, 1 - 1e-5)),
            logit(np.clip(deb_test, 1e-5, 1 - 1e-5)),
            logit(np.clip(tf_blend_oof, 1e-5, 1 - 1e-5)),
            logit(np.clip(tf_blend_test, 1e-5, 1 - 1e-5)),
        ),
    ]

    for mode, d_oof, d_test, t_oof, t_test in blend_modes:
        for w in np.linspace(0, 1, 1001):
            oof = w * d_oof + (1 - w) * t_oof
            test = w * d_test + (1 - w) * t_test

            if mode == "logit":
                oof = expit(oof)
                test = expit(test)

            auc = roc_auc_score(y, oof)
            rows.append({"kind": mode, "deberta_weight": float(w), "auc": float(auc)})

            if best is None or auc > best["auc"]:
                best = {
                    "kind": mode,
                    "deberta_weight": float(w),
                    "auc": float(auc),
                    "test": test,
                }

    return best, sorted(rows, key=lambda r: r["auc"], reverse=True)
```

Submission checks:

```python
sub = sample.copy()
sub["Insult"] = np.clip(best["test"], 1e-6, 1 - 1e-6)

assert list(sub.columns) == list(sample.columns)
assert len(sub) == len(sample)
assert sub["Insult"].between(0, 1).all()
assert sub["Insult"].nunique() > 100

sub.to_csv("submission.csv", index=False)
```

## What Actually Moved The Metric

The sparse TF-IDF ensemble was the decisive move. For this tiny binary text dataset, a careful linear ensemble with word ngrams, char ngrams, case-sensitive ngrams, decoded text, and manual insult/style features was already a very strong solution. Start here before spending time on transformers.

The features that mattered were complementary:
- Word `1..3` ngrams captured direct phrases and slurs.
- Char `2..6` ngrams caught misspellings, elongated words, punctuation-heavy insults, and morphological variants.
- Case-sensitive word features plus manual uppercase/exclamation/profanity/pronoun features preserved signals that would be lost by simple lowercasing.

The blend procedure mattered because the metric is ROC-AUC. The winning code optimized convex weights against OOF AUC, saved every model's OOF/test predictions, then blended TF-IDF and DeBERTa in raw/rank/logit spaces. This is safer than trusting a single fold or a single model's probability calibration.

The transformer helped only as a complement. Use DeBERTa-large with conservative fine-tuning and fold averaging, but do not let it replace the sparse model. On small noisy data, transformer OOF can be unstable; blend it only if OOF AUC improves.

## Pitfalls And Lessons

Do not output hard `0/1` labels. ROC-AUC evaluates ranking, so hard thresholding is probably the single most damaging mistake. Always submit continuous probabilities with many unique values.

Do not optimize accuracy, F1, or log loss as the main selection criterion. Select models and blend weights by OOF ROC-AUC.

Do not over-clean the comments. Removing punctuation, capitalization, repeated letters, quotes, question marks, exclamation marks, or profanity-like variants discards signal.

Do not train one random validation split and trust it. The dataset is small; use `StratifiedKFold(n_splits=5, shuffle=True, random_state=...)`.

Do not run DeBERTa on CPU. It is too slow and changes the practical solution path. If no GPU is available, submit the sparse TF-IDF ensemble rather than wasting time.

Do not use leaderboard probing to tune blend weights. Generate OOF predictions for each component and search weights locally.

The reliable lesson from the winning code is that the first strong sparse baseline was already a strong result on its own, and the transformer/blend layer was an incremental robustness step rather than the foundation.

## Recommended Execution Order

1. Inspect the CSV columns and sample submission.
2. Decode comments with `decode_escapes`.
3. Train the sparse TF-IDF ensemble under 5-fold stratified CV.
4. Save:
   - `tfidf_oofs.npy`
   - `tfidf_tests.npy`
   - `tfidf_model_names.json`
   - `tfidf_report.json`
5. If CUDA is available, train `microsoft/deberta-v3-large` under 5-fold stratified CV.
6. Save:
   - `deberta_oof.npy`
   - `deberta_test.npy`
   - `deberta_report.json`
7. Run final OOF blend search across raw/rank/logit blend modes.
8. Write `submission.csv`, clipped to `[1e-6, 1 - 1e-6]`, preserving sample submission columns and row count.

## Score Milestones

Relative ordering (what each step buys):
- A naive constant or random probability baseline sits near ROC-AUC `0.50`.
- A correct sparse TF-IDF linear ensemble (word + char + case-sensitive ngrams + manual features) under stratified CV is already a very strong result on its own if preprocessing is correct — this is the foundation.
- Adding an optional DeBERTa-large OOF blend, selected only when OOF AUC improves, is an incremental robustness step on top of the sparse ensemble.

Steer by the `criterion.json` pass line, not a fixed leaderboard threshold. What stays invariant is the ordering: the sparse ensemble is the decisive lever; the transformer blend is a complement.
