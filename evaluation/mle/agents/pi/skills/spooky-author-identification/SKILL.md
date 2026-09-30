# SKILL: Spooky Author Identification (three-author text classification · multi-class logloss)

**Task type**: Classic NLP three-way text classification (authors EAP / HPL / MWS); given an English sentence excerpt from a novel, predict author probabilities.
**Evaluation metric**: multi-class **log loss** (lower is better).
**Pass line**: the pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.
**Measured with this recipe**: a pure linear stack plateaus at a single-model-family ceiling; adding DeBERTa-v3 probability fusion is the single biggest jump, and more seeds / temperature tuning gives a further improvement.

---

## 1. Winning recipe

The core is a **two-legged fusion**: (1) a linear TF-IDF stack (3 models) + (2) a DeBERTa-v3-base fine-tune, with the OOF probabilities fed to a **temperature-aware softmax weight optimizer** for ensembling.

Data path: `prepared/public/{train.csv,test.csv,sample_submission.csv}`.
Column conventions: `train=[id,text,author]`, `test=[id,text]`, `sample=[id,EAP,HPL,MWS]`.
`CLASSES = ["EAP","HPL","MWS"]`, **use `LabelEncoder().fit(CLASSES)` to lock the class order**; the submission column order must equal that of sample.

**CV**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, **the linear stack and the transformer share the same folds** (aligned OOF is required for fusion).

### Leg 1: linear TF-IDF stack
- Features: a `FeatureUnion` concatenating **word 1-2gram** (`min_df=2, sublinear_tf=True, strip_accents="unicode", max_features=240000`) + **char 3-5gram** (`min_df=2, sublinear_tf=True, max_features=360000`). **fit_transform on train+test text together** (transductive, key point).
- Each fold trains 3 heads, each producing OOF + test probabilities:
  - `LogisticRegression(C=3.0, solver="saga", max_iter=700)`
  - **NB-SVM**: first compute the per-class NB log-count-ratio, multiply the features by the ratio, then feed to `LogisticRegression(C=4.0, solver="liblinear")`, one-vs-rest three binary classifiers, decision_function → softmax.
  - `ComplementNB(alpha=0.08)`

### Leg 2: DeBERTa-v3-base fine-tune
- backbone: `microsoft/deberta-v3-base`, `local_files_only=True`, **force-loaded with `dtype=torch.float32`** (avoids the half-precision loading pitfall), with **bf16 autocast** during training.
- **Custom CLS head** (do not use `AutoModelForSequenceClassification`, to avoid the pooler dimension pitfall): take `last_hidden_state[:,0]` → `Dropout(0.2)` → `Linear(hidden, 3)`, classification head weights `normal_(std=0.02)`, bias zeroed.
- Hyperparameters: `epochs=4, lr=1.6e-5, batch=32, max_len=192, warmup_ratio=0.08, weight_decay=0.01`, AdamW (`eps=1e-6`, no decay on bias/LayerNorm), `get_cosine_schedule_with_warmup`, grad clip 1.0.
- For each fold, save the state_dict of the best epoch by valid_loss, then reload it to produce OOF/test probabilities. When pushing for score, you can run deberta with multiple seeds and add them all to the ensemble.

### Ensemble (temperature-aware softmax weights)
- Collect the (oof,test) of all models, use Nelder-Mead to optimize **softmax weights + a temperature T** (`T∈[0.55,2.0]`) minimizing OOF logloss.
- Multiple starting points: an all-zeros start + a "bet heavily on deberta" start. Finally, apply the best weights to weight oof/test and then temperature-scale.

---

## 2. The points that actually move the metric (with lessons learned the hard way)

1. **[Biggest jump] The linear stack caps at a single-model-family ceiling, and tuning the linear stack will never close the gap to the pass line — you must bring in a transformer.** Adding a DeBERTa-v3 fine-tune and fusing its probabilities with the linear stack is the single biggest jump in one step. The trajectory shows five consecutive attempts repeatedly tweaking NB-SVM/min_df/C on the linear side with no qualitative change; **this is the most valuable lesson: recognize that the task-type bottleneck is "needs a semantic model" rather than "the linear stack can still be squeezed", and switch tracks in time.**
2. **[Two loading/head pitfalls to avoid]** Two mines repeatedly stepped on the transformer side: (1) the backbone must be force-loaded with `dtype=torch.float32` (half-precision loading causes problems); (2) **use a custom CLS head**, not HF's built-in SequenceClassification head (pooler/dimension pitfall). The code comments specifically note "float32 force-load, custom CLS head to avoid pooler".
3. **[Fusion needs temperature scaling]** A plain weighted average of probabilities is not enough; **temperature T scaling** can further reduce logloss — logloss penalizes over-confident probabilities heavily, and T>1 flattening the probabilities often directly lowers the score. Multi-seed deberta + temperature tuning is the final incremental leg.
4. **Lesson learned (environment layer)**: several "transformer failures" in early attempts were actually an **environment problem (API balance 403), not a method problem** — do not misjudge an environment failure as "the transformer route doesn't work" and abandon the correct direction. Once the balance was restored, the same recipe worked in one shot.
5. **CV alignment is a prerequisite for fusion**: the linear stack and the transformer must use **the same StratifiedKFold folds**, so the OOF can be aligned sample-by-sample when fed to the ensembler.

---

## 3. Key code snippets (skeleton only)

**Class locking + shared folds**
```python
CLASSES = ["EAP","HPL","MWS"]
le = LabelEncoder(); le.fit(CLASSES)
y = le.transform(train["author"].values)
assert list(le.classes_) == CLASSES
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(train["text"].values, y))   # shared by linear and transformer
```

**TF-IDF union (fit on train+test together)**
```python
word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=2,
                           sublinear_tf=True, strip_accents="unicode",
                           lowercase=True, max_features=240000)
char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2,
                           sublinear_tf=True, lowercase=True, max_features=360000)
union = FeatureUnion([("word",word_vec),("char",char_vec)], n_jobs=1)
x_all = union.fit_transform(np.concatenate([texts, test_texts]))
x_train, x_test = x_all[:len(train)], x_all[len(train):]
```

**NB-SVM (log-count-ratio × features → LR)**
```python
def nb_log_count_ratio(x, y, alpha=1.0):
    ratios = []
    for c in range(len(CLASSES)):
        pos = np.asarray(x[y==c].sum(0)).ravel() + alpha
        neg = np.asarray(x[y!=c].sum(0)).ravel() + alpha
        ratios.append(np.log(pos/pos.sum()) - np.log(neg/neg.sum()))
    return np.asarray(ratios)

def fit_predict_nbsvm(train_x, train_y, valid_x, test_x, c=4.0):
    ratios = nb_log_count_ratio(train_x, train_y)
    vlog = np.zeros((valid_x.shape[0],3)); tlog = np.zeros((test_x.shape[0],3))
    for k in range(3):
        clf = LogisticRegression(C=c, solver="liblinear", max_iter=1000, random_state=13)
        clf.fit(train_x.multiply(ratios[k]), (train_y==k).astype(int))
        vlog[:,k] = clf.decision_function(valid_x.multiply(ratios[k]))
        tlog[:,k] = clf.decision_function(test_x.multiply(ratios[k]))
    return normalize_probs(np.exp(vlog - vlog.max(1,keepdims=True))), \
           normalize_probs(np.exp(tlog - tlog.max(1,keepdims=True)))
```

**DeBERTa custom CLS head (avoid pooler, float32 force-load)**
```python
class ClsModel(nn.Module):
    def __init__(self, model_name, dropout=0.2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name, local_files_only=True, dtype=torch.float32)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, len(CLASSES))
        nn.init.normal_(self.classifier.weight, std=0.02); nn.init.zeros_(self.classifier.bias)
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.drop(out.last_hidden_state[:,0]))   # CLS token
```
Training uses `torch.amp.autocast("cuda", dtype=torch.bfloat16)`; save the best epoch by valid_loss; AdamW(lr=1.6e-5, eps=1e-6), no weight_decay on bias/LayerNorm; cosine + warmup 0.08; grad clip 1.0.

**Temperature-aware softmax weight ensemble**
```python
def temperature_scale(p, t):
    z = np.log(np.clip(p,1e-12,1.0))/t; z -= z.max(1,keepdims=True)
    return normalize_probs(np.exp(z))

def objective(z):                       # z = [logits_per_model..., temp_param]
    w = softmax(z[:n]); t = 0.55 + 1.45/(1+np.exp(-z[-1]))
    p = temperature_scale(sum(wi*oi for wi,oi in zip(w, oofs)), t)
    return log_loss(y, p, labels=[0,1,2])
# Nelder-Mead, multiple starts (all-zeros + bet heavily on deberta), take the minimum OOF logloss
```
Assert before submission: column order == sample.columns, id sets match, probabilities all finite and non-NaN.

---

## 4. Score milestones (relative — what each step buys)

| Stage | logloss | Note |
|---|---|---|
| Pure linear TF-IDF stack (LR/NB-SVM/CNB ensemble) | single-model-family ceiling | Not enough on its own; even squeezed to the max, tuning the linear stack cannot close the gap to the pass line |
| **+ DeBERTa-v3-base fine-tune fusion** | large drop | **The single biggest jump — the qualitative move** |
| + multi-seed deberta + temperature tuning | further drop | Additional improvement on top of fusion |
| Further push | lower still | Requires more on the transformer side (larger model / more seeds / stronger ensemble); unreachable by tuning the linear stack |

**One-sentence decision**: once the linear stack stalls at its ceiling, don't cling to it — bring in DeBERTa-v3 fusion right away; this is the only qualitative move that crosses the pass line, and it is the single biggest lever.

Self-check against the `criterion.json` pass line, not a fixed threshold. What stays invariant is the ORDERING: the linear stack plateaus, DeBERTa-v3 probability fusion is the decisive jump, and temperature-scaled multi-seed ensembling adds the final increment.
