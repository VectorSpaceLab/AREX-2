Jigsaw Toxic Comment Classification Challenge: multilabel toxic-comment classification with 6 labels, evaluated by the mean ROC-AUC across label columns (higher is better). The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

## Approach

Use a two-family ensemble: a strong NB-SVM/TF-IDF linear model plus a strict 5-fold fine-tuned transformer. Do not rely on a single transformer or a single linear model.

1. Load `train.csv`, `test.csv`, and `sample_submission.csv`.
2. Predict these labels, in exactly this order:

```python
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SEED = 42
```

3. Use the sample submission as the canonical test row order:

```python
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
sample = pd.read_csv(DATA / "sample_submission.csv")

test = sample[["id"]].merge(test, on="id", how="left")
assert test["comment_text"].notna().all()

train["comment_text"] = train["comment_text"].fillna(" ")
test["comment_text"] = test["comment_text"].fillna(" ")
y = train[LABELS].values.astype(np.float32)
```

4. Use one shared strict CV split for every model:

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(kf.split(train))
```

5. Train a 5-fold NB-SVM model with word and character TF-IDF. This is the resilient baseline and an important ensemble member.
6. Fine-tune `microsoft/deberta-v3-large` for 1 epoch per fold with BCE loss and a 6-logit head.
7. Save OOF and test predictions after every fold. Transformer failures from infra/time are common; resumability matters.
8. Blend using local OOF mean ROC-AUC. The successful submission selected the best OOF blend among raw and rank-normalized predictions, especially NB rank + transformer rank.

Core hyperparameters from the strong run:

```text
CV:             5-fold KFold, shuffle=True, random_state=42
Transformer:    microsoft/deberta-v3-large
max_len:        192
epochs:         1
batch_size:     12
eval_batch:     64
grad_accum:     2
lr:             1.5e-5
warmup:         0.06
weight_decay:   0.01
precision:      CUDA autocast bfloat16
optimizer:      AdamW
scheduler:      linear warmup/decay
loss:           BCEWithLogitsLoss
head:           CLS token -> dropout(0.1) -> Linear(hidden, 6)
NB word TF-IDF: word 1-2 grams, min_df=3, max_df=0.95, max_features=150000
NB char TF-IDF: char 2-6 grams, min_df=3, max_features=250000
NB classifier:  SGDClassifier(log_loss, alpha=1e-5, max_iter=12, average=True)
```

## What Actually Moved The Metric

The first useful baseline was NB-SVM. It reached a strong OOF but well short of the pass line. Pure linear/TF-IDF modeling effectively topped out here.

The big lift came from adding a fine-tuned transformer and blending it with NB-SVM. A non-strict early blend of NB-SVM with transformer-style predictions already improved on the linear-only baseline, proving that the model-family combination mattered, but it was still not enough.

The decisive jump came from making the transformer run strict and fold-complete, then selecting the ensemble by OOF AUC. The final route was: strict 5-fold DeBERTa-v3-large OOF + NB-SVM OOF + rank/raw blend search. This produced the best OOF blend and the best submission of the trajectory.

Pitfalls and lessons:

- A bare transformer run without the NB-SVM ensemble regressed below the blend. Do not submit a naked single model unless debugging.
- More attempts failed from infrastructure than from modeling: 403 balance failures, 502 gateway failures, and incomplete transformer OOF runs. Build caching and per-fold resume markers before long training.
- The most damaging modeling mistake is treating a partial or non-strict transformer result as enough. The best configuration requires strict 5-fold OOF predictions so the blend weight is selected honestly.
- Do not spend time over-optimizing only the linear model. NB-SVM is necessary as an ensemble anchor, but it cannot reach the top range alone.
- Rank blending matters because ROC-AUC only depends on ordering. Search both raw and per-column rank-normalized blends.

## Key Code Snippets

### Metric And Rank Normalization

Use mean ROC-AUC across the six labels. Rank-normalize each prediction column before blend search.

```python
import numpy as np
from sklearn.metrics import roc_auc_score

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def mean_auc(y, p):
    return float(np.mean([roc_auc_score(y[:, i], p[:, i]) for i in range(y.shape[1])]))

def rank_cols(x):
    out = np.zeros_like(x, dtype=np.float32)
    n = x.shape[0]
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j], kind="mergesort")
        ranks = np.empty(n, dtype=np.float32)
        ranks[order] = (np.arange(n, dtype=np.float32) + 0.5) / n
        out[:, j] = ranks
    return out
```

### NB-SVM Fold Skeleton

Fit TF-IDF on train + test text, but compute NB log-count ratios using only the training fold labels. Produce true OOF validation predictions and average test predictions over folds.

```python
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

texts_tr = train["comment_text"].astype(str).values
texts_te = test["comment_text"].astype(str).values
all_text = np.concatenate([texts_tr, texts_te])

word_vec = TfidfVectorizer(
    analyzer="word",
    token_pattern=r"\w{1,}",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    max_features=150000,
    strip_accents="unicode",
    sublinear_tf=True,
)
char_vec = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 6),
    min_df=3,
    max_features=250000,
    strip_accents="unicode",
    sublinear_tf=True,
)

Xw = word_vec.fit_transform(all_text)
Xc = char_vec.fit_transform(all_text)
X = hstack([Xw, Xc]).tocsr()
Xtr = X[:len(train)]
Xte = X[len(train):]

nb_oof = np.zeros((len(train), len(LABELS)), dtype=np.float32)
nb_test = np.zeros((len(test), len(LABELS)), dtype=np.float32)

alpha = 1.0
for f, (tr_idx, va_idx) in enumerate(folds):
    X_tr_base = Xtr[tr_idx]
    X_va_base = Xtr[va_idx]

    for j, label in enumerate(LABELS):
        yj = y[:, j]

        pos = X_tr_base[yj[tr_idx] == 1].sum(axis=0) + alpha
        neg = X_tr_base[yj[tr_idx] == 0].sum(axis=0) + alpha
        r = np.asarray(np.log((pos / pos.sum()) / (neg / neg.sum()))).ravel()

        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-5,
            max_iter=12,
            tol=1e-4,
            random_state=SEED + f * 17 + j,
            average=True,
        )
        clf.fit(X_tr_base.multiply(r), yj[tr_idx])

        nb_oof[va_idx, j] = clf.predict_proba(X_va_base.multiply(r))[:, 1]
        nb_test[:, j] += clf.predict_proba(Xte.multiply(r))[:, 1] / len(folds)

    print(f"fold={f+1} nb_auc={mean_auc(y[va_idx], nb_oof[va_idx]):.6f}")
```

### Transformer Dataset And Collate

Do dynamic padding per batch. Tokenize once for all train/test texts.

```python
import torch
from torch.utils.data import Dataset

class ToxicDataset(Dataset):
    def __init__(self, enc, labels=None, indices=None):
        self.enc = enc
        self.labels = labels
        self.indices = np.arange(len(enc["input_ids"])) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, pos):
        i = int(self.indices[pos])
        item = {k: torch.tensor(v[i], dtype=torch.long) for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.float32)
        return item

def make_collate(pad_id):
    def collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        out = {}
        for k in [k for k in batch[0].keys() if k != "labels"]:
            pad_val = pad_id if k == "input_ids" else 0
            out[k] = torch.stack([
                torch.cat([b[k], torch.full((max_len - len(b[k]),), pad_val, dtype=torch.long)])
                for b in batch
            ])
        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch])
        return out
    return collate

def tokenize_texts(tokenizer, texts, max_len):
    return tokenizer(
        list(texts),
        padding=False,
        truncation=True,
        max_length=max_len,
        return_token_type_ids=True,
    )
```

### Transformer Model Head

Use the first token embedding, dropout, and one linear 6-label head. Train with `BCEWithLogitsLoss`.

```python
import torch.nn as nn
from transformers import AutoModel

class ToxicModel(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, len(LABELS))
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        h = self.backbone(**kwargs).last_hidden_state[:, 0]
        return self.head(self.dropout(h))
```

### Transformer Training Loop Skeleton

Do not fall back to CPU. The strong run used CUDA, bfloat16 autocast, 1 epoch, gradient accumulation, and per-fold OOF/test saving.

```python
import gc
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

model_name = "microsoft/deberta-v3-large"
max_len = 192
epochs = 1
batch_size = 12
eval_batch_size = 64
grad_accum = 2
lr = 1.5e-5
warmup = 0.06
num_workers = 4

assert torch.cuda.is_available(), "CUDA unavailable; do not run transformer training on CPU"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)
pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
enc_tr = tokenize_texts(tokenizer, train["comment_text"].astype(str).values, max_len)
enc_te = tokenize_texts(tokenizer, test["comment_text"].astype(str).values, max_len)
collate = make_collate(pad_id)

tr_oof = np.zeros((len(train), len(LABELS)), dtype=np.float32)
tr_test = np.zeros((len(test), len(LABELS)), dtype=np.float32)

for f, (tr_idx, va_idx) in enumerate(folds):
    train_ds = ToxicDataset(enc_tr, y, tr_idx)
    valid_ds = ToxicDataset(enc_tr, y, va_idx)
    test_ds = ToxicDataset(enc_te, None, None)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=num_workers > 0,
    )

    model = ToxicModel(model_name).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = int(np.ceil(len(train_loader) / grad_accum) * epochs)
    sched = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(total_steps * warmup),
        num_training_steps=total_steps,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    opt.zero_grad(set_to_none=True)

    for ep in range(epochs):
        for it, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch)
                loss = loss_fn(logits.float(), labels) / grad_accum

            loss.backward()

            if (it + 1) % grad_accum == 0 or (it + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

    tr_oof[va_idx] = infer_model(model, valid_loader, device)
    tr_test += infer_model(model, test_loader, device) / len(folds)

    print(f"fold={f+1} transformer_auc={mean_auc(y[va_idx], tr_oof[va_idx]):.6f}")

    del model, opt, sched, train_loader, valid_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()
```

Inference helper:

```python
def infer_model(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "labels"}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch).float()
            preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(preds).astype(np.float32)
```

### Blend Search And Submission

Search both raw and rank-normalized blends from `0.00` to `1.00` in steps of `0.025`. Pick by OOF AUC, then write test predictions in sample-submission order.

```python
def validate_submission(sample, sub):
    assert list(sub.columns) == list(sample.columns)
    assert len(sub) == len(sample)
    assert sub["id"].equals(sample["id"])

    vals = sub[LABELS].to_numpy()
    assert np.isfinite(vals).all()
    assert vals.min() >= 0.0 and vals.max() <= 1.0

    for c in LABELS:
        assert sub[c].nunique() > 100, c

candidates = []

nb_rank_oof = rank_cols(nb_oof)
nb_rank_test = rank_cols(nb_test)
tr_rank_oof = rank_cols(tr_oof)
tr_rank_test = rank_cols(tr_test)

candidates.append(("nb_raw", mean_auc(y, nb_oof), nb_test))
candidates.append(("nb_rank", mean_auc(y, nb_rank_oof), nb_rank_test))
candidates.append(("tr_raw", mean_auc(y, tr_oof), tr_test))
candidates.append(("tr_rank", mean_auc(y, tr_rank_oof), tr_rank_test))

for w in np.linspace(0.0, 1.0, 41):
    oof = w * nb_rank_oof + (1.0 - w) * tr_rank_oof
    te = w * nb_rank_test + (1.0 - w) * tr_rank_test
    candidates.append((f"rank_nb{w:.2f}_tr{1-w:.2f}", mean_auc(y, oof), te))

for w in np.linspace(0.0, 1.0, 41):
    oof = w * nb_oof + (1.0 - w) * tr_oof
    te = w * nb_test + (1.0 - w) * tr_test
    candidates.append((f"raw_nb{w:.2f}_tr{1-w:.2f}", mean_auc(y, oof), te))

candidates.sort(key=lambda x: x[1], reverse=True)
for name, auc, _ in candidates[:12]:
    print(name, auc)

best_name, best_auc, best_test = candidates[0]

sub = sample.copy()
sub[LABELS] = np.clip(best_test, 1e-6, 1 - 1e-6)
validate_submission(sample, sub)
sub.to_csv("submission.csv", index=False)
print(f"wrote submission.csv using {best_name} auc={best_auc:.6f}")
```

## Operational Checklist

Before submitting:

1. Confirm `test` is merged onto `sample_submission[["id"]]` and submission `id` order exactly matches `sample_submission.csv`.
2. Confirm all five transformer folds finished and filled their validation slice in `tr_oof`.
3. Confirm NB-SVM and transformer OOF arrays use the same fold split and row order.
4. Print `mean_auc(y, nb_oof)`, `mean_auc(y, tr_oof)`, and the top blend candidates.
5. Validate every submission label column is finite, clipped to `[0, 1]`, and non-constant.
6. Save intermediate arrays: `nb_oof.npy`, `nb_test.npy`, `tr_oof_*.npy`, `tr_test_*.npy`, plus a blend summary.

## Score Milestones

Expected score ladder (relative — see below):

```text
NB-SVM only:                         reasonable linear floor
Early NB-SVM + transfer-style blend: clear step up
Bare single DeBERTa run:             not useful alone
Strict 5-fold DeBERTa + NB blend:    strongest; the calibrated NB-SVM x DeBERTa blend is the decisive lever
```

Self-check against the `criterion.json` pass line, not a fixed threshold:

```text
Self-check against the `criterion.json` pass line for this run, not fixed leaderboard thresholds.
Submit only when the strict OOF blend is clearly at its strongest.
```

What stays invariant is the ORDERING: NB-SVM alone tops out below the pass line; adding a strict, fold-complete DeBERTa-v3-large member and selecting the blend by OOF AUC (with rank normalization) is the decisive lift; partial/non-strict transformer results and single naked models regress.
