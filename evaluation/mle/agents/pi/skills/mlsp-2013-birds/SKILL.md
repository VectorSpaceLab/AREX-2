MLSP 2013 Birds: multi-label bird-call audio recognition with 19 labels, evaluated by mean column-wise ROC AUC (higher is better). The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

## Approach

Use the provided per-segment/tabular features rather than raw audio. The reliable solution is a conservative one-vs-rest tabular ensemble over 19 bird classes, with per-class stratified CV and OOF-based blending.

1. Read `essential_data/rec_labels_test_hidden.txt`.
   - Rows with `labels == "?"` are the hidden test set.
   - Other rows are training rows.
   - Labels are comma-separated integer class ids from `0` to `18`.

2. Build one row per `rec_id` using only provided metadata/supplemental files:
   - `supplemental_data/segment_features.txt`
   - `supplemental_data/histogram_of_segments.txt`
   - `supplemental_data/segment_rectangles.txt`
   - `essential_data/rec_id2filename.txt`

3. Aggregate `segment_features.txt` by `rec_id`.
   - For every segment feature column, compute:
     - mean
     - std, filled with 0
     - min
     - max
     - median
     - 25% quantile
     - 75% quantile
     - skew, filled with 0
   - Add `segment_count`.

4. Add `histogram_of_segments.txt` directly by `rec_id`.

5. Aggregate `segment_rectangles.txt`.
   - Keep columns:
     - `rec_id`
     - `segment_id`
     - `x1`
     - `x2`
     - `y1`
     - `y2`
   - Derive:
     - `width = x2 - x1`
     - `height = y2 - y1`
     - `area = width * height`
   - Aggregate by `rec_id`:
     - mean and std for `x1, x2, y1, y2, width, height, area`
     - max for `width, height, area`

6. Add filename-derived metadata from `rec_id2filename.txt`.
   - Split `filename` on `_`.
   - Parse:
     - `site`
     - `date`
     - `time`
     - `year`
     - `month`
     - `day`
     - `hour`
     - `minute`
     - `dayofyear`
     - `tod_min = hour * 60 + minute`
   - Add cyclic encodings:
     - `sin/cos(2*pi*dayofyear/366)`
     - `sin/cos(2*pi*tod_min/1440)`
   - One-hot encode `site`.

7. Fill all missing feature values with 0.

8. Train two model families independently, one binary classifier per class:
   - scaled logistic regression
   - sklearn gradient boosting

9. Use per-class stratified CV:
   - `n_splits = min(5, positive_count, negative_count)`
   - `shuffle=True`
   - seed `7`
   - if a class cannot support at least 2 folds, use the smoothed class prior.

10. Evaluate OOF predictions with mean column-wise AUC.

11. Blend logistic regression and gradient boosting using OOF predictions.
   - Search global weights:
     - `logreg_weight` in `np.linspace(0.0, 1.0, 21)`
     - `gb_weight = 1.0 - logreg_weight`
   - Also consider each single model alone.
   - Select the blend with best OOF mean AUC.
   - Apply the same weights to hidden-test predictions.

12. Create `submission.csv`.
   - Use `sample_submission.csv` as the template.
   - Submission id mapping is:
     - `Id = rec_id * 100 + class_id`
   - Fill `Probability`.
   - Clip predictions to `[1e-5, 1 - 1e-5]`.

## What Actually Moved The Metric

The most important pieces, in priority order, were:

1. Strong aggregation of official `segment_features.txt`.
   - The reliable solution did not attempt expensive raw-audio modeling.
   - Mean/std/min/max/median/quantile/skew statistics over segment-level features created a stable recording-level representation.
   - This was the core of the result.

2. Adding the cheap supplemental structured features.
   - `histogram_of_segments.txt` was merged directly.
   - `segment_rectangles.txt` was converted into width/height/area statistics.
   - Filename metadata supplied site and time-of-day/season signals.
   - These features are low risk and helped stabilize the tabular baseline.

3. OOF-selected blending of linear and shallow tree models.
   - Logistic regression with balanced class weights handled sparse/imbalanced labels well.
   - Gradient boosting captured small nonlinear effects.
   - The final blend was chosen by OOF mean column-wise AUC, matching the public metric.

Pitfalls and lessons:

- Do not optimize accuracy, F1, or row-wise metrics. The target is mean column-wise AUC, so rare labels matter equally to common labels.
- Do not use a single multi-output split without checking per-class positives. Some classes are rare; use `min(5, pos, neg)` per class.
- Do not emit hard labels. AUC needs ranked probabilities.
- Do not spend the first pass building raw spectrogram/audio models. The reliable solution came from provided tabular features and robust CV.
- The most damaging failure mode is submitting predictions with the wrong `Id` mapping. The expected mapping is `rec_id * 100 + class_id`; preserve sample row order exactly.
- Follow-up attempts to pursue more via spectrogram features and larger fusions did not run to completion and did not contribute. Treat them as optional exploration, not required for the reliable tabular route.

## Key Code Snippets

### Label Parsing

```python
N_CLASSES = 19

def parse_labels(path):
    rows = []
    for line in path.read_text().splitlines()[1:]:
        rec, labels = line.split(",", 1) if "," in line else (line, "")
        rows.append((int(rec), labels))

    df = pd.DataFrame(rows, columns=["rec_id", "labels"])
    y = np.zeros((len(df), N_CLASSES), dtype=np.int8)

    for i, labels in enumerate(df["labels"]):
        if labels in ("", "?"):
            continue
        for label in labels.split(","):
            y[i, int(label)] = 1

    return df, y
```

### Segment Feature Aggregation

```python
seg = pd.read_csv(DATA / "supplemental_data" / "segment_features.txt",
                  header=None, skiprows=1)
seg.columns = ["rec_id", "segment_id"] + [
    f"seg_f{i}" for i in range(seg.shape[1] - 2)
]

feature_cols = [c for c in seg.columns if c.startswith("seg_f")]
grouped = seg.groupby("rec_id")[feature_cols]

seg_agg = pd.concat(
    [
        grouped.mean().add_prefix("seg_mean_"),
        grouped.std().fillna(0).add_prefix("seg_std_"),
        grouped.min().add_prefix("seg_min_"),
        grouped.max().add_prefix("seg_max_"),
        grouped.median().add_prefix("seg_med_"),
        grouped.quantile(0.25).add_prefix("seg_q25_"),
        grouped.quantile(0.75).add_prefix("seg_q75_"),
        grouped.skew().fillna(0).add_prefix("seg_skew_"),
        grouped.size().rename("segment_count").to_frame(),
    ],
    axis=1,
).reset_index()
```

### Histogram And Rectangle Features

```python
hist = pd.read_csv(DATA / "supplemental_data" / "histogram_of_segments.txt",
                   header=None, skiprows=1)
hist.columns = ["rec_id"] + [f"hist_{i}" for i in range(hist.shape[1] - 1)]

rect = pd.read_csv(DATA / "supplemental_data" / "segment_rectangles.txt",
                   header=None, skiprows=1)
rect = rect.iloc[:, :6]
rect.columns = ["rec_id", "segment_id", "x1", "x2", "y1", "y2"]

rect["width"] = rect["x2"] - rect["x1"]
rect["height"] = rect["y2"] - rect["y1"]
rect["area"] = rect["width"] * rect["height"]

rect_cols = ["x1", "x2", "y1", "y2", "width", "height", "area"]

rect_agg = pd.concat(
    [
        rect.groupby("rec_id")[rect_cols].mean().add_prefix("rect_mean_"),
        rect.groupby("rec_id")[rect_cols].std().fillna(0).add_prefix("rect_std_"),
        rect.groupby("rec_id")[["width", "height", "area"]]
            .max()
            .add_prefix("rect_max_"),
    ],
    axis=1,
).reset_index()
```

### Filename Metadata

```python
fn = pd.read_csv(DATA / "essential_data" / "rec_id2filename.txt")
meta = fn.copy()

split = meta["filename"].str.split("_", expand=True)
meta["site"] = split[0]
meta["date"] = split[1]
meta["time"] = split[2]

meta["year"] = meta["date"].str[:4].astype(int)
meta["month"] = meta["date"].str[4:6].astype(int)
meta["day"] = meta["date"].str[6:8].astype(int)
meta["hour"] = meta["time"].str[:2].astype(int)
meta["minute"] = meta["time"].str[2:4].astype(int)

meta["dayofyear"] = pd.to_datetime(meta["date"], format="%Y%m%d").dt.dayofyear
meta["tod_min"] = meta["hour"] * 60 + meta["minute"]

for col, period in [("dayofyear", 366), ("tod_min", 24 * 60)]:
    meta[f"{col}_sin"] = np.sin(2 * np.pi * meta[col] / period)
    meta[f"{col}_cos"] = np.cos(2 * np.pi * meta[col] / period)

site_oh = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore",
).fit_transform(meta[["site"]])

site_df = pd.DataFrame(
    site_oh,
    columns=[f"site_{c}" for c in sorted(meta["site"].unique())],
)

meta = pd.concat(
    [
        meta[
            [
                "rec_id",
                "year",
                "month",
                "day",
                "hour",
                "minute",
                "dayofyear",
                "tod_min",
                "dayofyear_sin",
                "dayofyear_cos",
                "tod_min_sin",
                "tod_min_cos",
            ]
        ],
        site_df,
    ],
    axis=1,
)
```

### Final Feature Matrix

```python
labels_df, all_y = parse_labels(DATA / "essential_data" / "rec_labels_test_hidden.txt")

features = labels_df[["rec_id"]].copy()
features = features.merge(seg_agg, on="rec_id", how="left")
features = features.merge(hist, on="rec_id", how="left")
features = features.merge(rect_agg, on="rec_id", how="left")
features = features.merge(meta, on="rec_id", how="left")
features = features.fillna(0)

hidden_mask = labels_df["labels"].eq("?").to_numpy()
train_mask = ~hidden_mask

x = features.loc[train_mask].drop(columns=["rec_id"]).to_numpy(dtype=np.float32)
y = all_y[train_mask]
test_x = features.loc[hidden_mask].drop(columns=["rec_id"]).to_numpy(dtype=np.float32)
test_rec_ids = features.loc[hidden_mask, "rec_id"].to_numpy()
```

### Metric

```python
def mean_col_auc(y_true, pred):
    aucs = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) < 2:
            aucs.append(float("nan"))
        else:
            aucs.append(float(roc_auc_score(y_true[:, c], pred[:, c])))
    return float(np.nanmean(aucs)), aucs
```

### Models

```python
SEEDS = [7]

def get_model(kind, seed):
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.35,
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=seed,
            ),
        )

    if kind == "gb":
        return GradientBoostingClassifier(
            n_estimators=140,
            learning_rate=0.035,
            max_depth=2,
            subsample=0.75,
            random_state=seed,
        )

    raise ValueError(kind)
```

### Per-Class Stratified CV

```python
def predict_positive(model, x):
    proba = model.predict_proba(x)
    if isinstance(proba, list):
        proba = proba[0]
    return proba[:, 1]

def cv_predict(kind, x, y, test_x):
    oof = np.zeros_like(y, dtype=float)
    test_pred = np.zeros((test_x.shape[0], y.shape[1]), dtype=float)

    for c in range(y.shape[1]):
        yc = y[:, c]
        pos = int(yc.sum())
        neg = int(len(yc) - pos)
        n_splits = min(5, pos, neg)

        if n_splits < 2:
            prior = (pos + 0.5) / (len(yc) + 1.0)
            oof[:, c] = prior
            test_pred[:, c] = prior
            continue

        pred_sum = np.zeros(test_x.shape[0], dtype=float)
        n_models = 0

        for seed in SEEDS:
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )

            for tr, va in splitter.split(x, yc):
                model = get_model(kind, seed + c * 100)
                model.fit(x[tr], yc[tr])

                oof[va, c] += predict_positive(model, x[va]) / len(SEEDS)
                pred_sum += predict_positive(model, test_x)
                n_models += 1

        test_pred[:, c] = pred_sum / max(n_models, 1)

    return oof, test_pred
```

### OOF Blend Selection

```python
kinds = ["logreg", "gb"]

oofs = {}
tests = {}
scores = {}

for kind in kinds:
    oof, test_pred = cv_predict(kind, x, y, test_x)
    score, aucs = mean_col_auc(y, oof)

    oofs[kind] = oof
    tests[kind] = test_pred
    scores[kind] = score

candidates = []

for k in kinds:
    candidates.append((k, {k: 1.0}))

for a in np.linspace(0.0, 1.0, 21):
    candidates.append(("blend", {"logreg": a, "gb": 1.0 - a}))

best_score = -1.0
best_weights = None

for name, weights in candidates:
    weights = {k: v for k, v in weights.items() if v > 0}
    s = sum(weights.values())
    weights = {k: v / s for k, v in weights.items()}

    blend = sum(oofs[k] * w for k, w in weights.items())
    score, _ = mean_col_auc(y, blend)

    if score > best_score:
        best_score = score
        best_weights = weights

final_pred = sum(tests[k] * w for k, w in best_weights.items())
final_pred = np.clip(final_pred, 1e-5, 1 - 1e-5)
```

### Submission Mapping

```python
sample = pd.read_csv(DATA / "sample_submission.csv")

pred_map = {
    int(rec) * 100 + c: float(final_pred[i, c])
    for i, rec in enumerate(test_rec_ids)
    for c in range(N_CLASSES)
}

sub = sample.copy()
sub["Probability"] = sub["Id"].map(pred_map)

if sub["Probability"].isna().any():
    missing = sub.loc[sub["Probability"].isna(), "Id"].head().tolist()
    raise RuntimeError(f"missing predictions for ids {missing}")

assert list(sub.columns) == list(sample.columns)
assert len(sub) == len(sample)
assert sub["Id"].equals(sample["Id"])
assert np.isfinite(sub["Probability"]).all()
assert sub["Probability"].between(0, 1).all()

sub.to_csv("submission.csv", index=False)
```

## Score Milestones (relative)

A naive or underbuilt baseline sits near the median/lower boundary if it ignores the rich segment aggregations and supplemental metadata.

The robust tabular configuration is the reliable strong route:
- Features: aggregated `segment_features` + histogram + rectangle stats + filename metadata
- Models: per-class `logreg` and `gb`
- CV: per-class stratified 5-fold where possible
- Blend: OOF-selected global logreg/GB weight grid

Self-check against the `criterion.json` pass line, not a fixed threshold. What stays invariant is the ORDERING: strong segment-feature aggregation is the core, supplemental structured features stabilize it, and OOF-selected linear/tree blending gives the final lift — while raw-audio modeling is not needed for the reliable route.
