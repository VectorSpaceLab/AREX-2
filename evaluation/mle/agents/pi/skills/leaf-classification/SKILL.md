Task: 99-class leaf species classification; metric: multiclass log loss (lower is better). The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

## Approach

Use the competition-provided 192 tabular features as the core signal, then add small image-derived morphology metadata and train a conservative probability ensemble. The strong solution did not use deep learning; it relied on well-calibrated linear models, multi-seed CV, temperature sharpening, and safe probability clipping.

### 1. Load the official tabular data

Use:

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `images/{id}.jpg`

The tabular feature columns are all columns in `test.csv` except `id`. They are the precomputed `margin`, `shape`, and `texture` features, 64 each, for 192 total features.

Label-encode `train["species"]` into integer classes and preserve the sample submission column order for output.

```python
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

feature_cols = [c for c in test.columns if c != "id"]

le = LabelEncoder()
y = le.fit_transform(train["species"])
labels = np.arange(len(le.classes_))

x_train_tab = train[feature_cols].to_numpy(np.float64)
x_test_tab = test[feature_cols].to_numpy(np.float64)
```

### 2. Add compact image morphology metadata

The tabular features are already very strong, but adding simple metadata from the leaf images improved the solution. Do not build a CNN for this dataset unless you have a very controlled validation signal; there are only about 990 training samples.

Use grayscale thresholding to create a foreground mask, then extract bbox, area, contour, centroid/spread, and Hu moment features. The successful solution produced 24 image metadata features per image and concatenated them to the 192 tabular features.

```python
def image_meta_features(ids):
    feats = []
    img_dir = os.path.join(DATA_DIR, "images")

    for leaf_id in ids:
        path = os.path.join(img_dir, f"{int(leaf_id)}.jpg")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)

        mask = (img < 245).astype(np.uint8)
        ys, xs = np.where(mask > 0)
        h, w = img.shape

        if len(xs) == 0:
            feats.append(np.zeros(24, dtype=np.float64))
            continue

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        bw = max(1, x1 - x0 + 1)
        bh = max(1, y1 - y0 + 1)
        crop = mask[y0:y1 + 1, x0:x1 + 1]

        area = float(mask.sum())
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            contour_area = float(cv2.contourArea(cnt))
            perimeter = float(cv2.arcLength(cnt, True))
            hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
        else:
            contour_area = perimeter = hull_area = 0.0

        moments = cv2.moments(mask)
        hu = cv2.HuMoments(moments).reshape(-1)
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

        row = np.array([
            h,
            w,
            area / (h * w),
            bw / w,
            bh / h,
            bw / bh,
            area / (bw * bh),
            contour_area / (area + 1e-9),
            perimeter / (np.sqrt(area) + 1e-9),
            hull_area / (area + 1e-9),
            (x0 + x1) / (2 * w),
            (y0 + y1) / (2 * h),
            xs.mean() / w,
            ys.mean() / h,
            xs.std() / w,
            ys.std() / h,
            crop.mean(),
        ], dtype=np.float64)

        feats.append(np.r_[row, hu.astype(np.float64)])

    return np.vstack(feats)
```

Concatenate metadata to train/test in one pass so preprocessing is identical:

```python
ids_all = pd.concat([train["id"], test["id"]], ignore_index=True).to_numpy()
img_meta = image_meta_features(ids_all)

x_train_meta = np.hstack([x_train_tab, img_meta[:len(train)]])
x_test_meta = np.hstack([x_test_tab, img_meta[len(train):]])
```

### 3. Train the exact model family that worked

The strongest and safest models were logistic regression pipelines. The final ensemble evaluated several candidates, but selected the two best metadata logistic-regression models:

- `PowerTransformer(method="yeo-johnson", standardize=True)` + `LogisticRegression(C=2.0, solver="lbfgs", max_iter=5000)`
- `PowerTransformer(method="yeo-johnson", standardize=True)` + `LogisticRegression(C=3.0, solver="lbfgs", max_iter=5000)`

Additional candidates were evaluated for safety, but not used in the final fixed blend unless they passed a strict OOF gate:

- `PowerTransformer` + LR `C=5.0` on metadata features
- `StandardScaler` + LR `C=2.0` on tabular-only features
- `StandardScaler` + RBF SVC `C=10`, `gamma=0.003`, `probability=True` on tabular-only features

```python
models = [
    (
        "power_lr_meta_C2",
        make_pipeline(
            PowerTransformer(method="yeo-johnson", standardize=True),
            LogisticRegression(C=2.0, solver="lbfgs", max_iter=5000),
        ),
        x_train_meta,
        x_test_meta,
    ),
    (
        "power_lr_meta_C3",
        make_pipeline(
            PowerTransformer(method="yeo-johnson", standardize=True),
            LogisticRegression(C=3.0, solver="lbfgs", max_iter=5000),
        ),
        x_train_meta,
        x_test_meta,
    ),
    (
        "power_lr_meta_C5",
        make_pipeline(
            PowerTransformer(method="yeo-johnson", standardize=True),
            LogisticRegression(C=5.0, solver="lbfgs", max_iter=5000),
        ),
        x_train_meta,
        x_test_meta,
    ),
    (
        "std_lr_tab_C2",
        make_pipeline(
            StandardScaler(),
            LogisticRegression(C=2.0, solver="lbfgs", max_iter=5000),
        ),
        x_train_tab,
        x_test_tab,
    ),
    (
        "svc_tab_C10_g003",
        make_pipeline(
            StandardScaler(),
            SVC(C=10, gamma=0.003, kernel="rbf", probability=True, random_state=17),
        ),
        x_train_tab,
        x_test_tab,
    ),
]
```

### 4. Use 6-fold stratified OOF across multiple seeds

The successful CV scheme was:

- `StratifiedKFold`
- `n_splits=6`
- `shuffle=True`
- seeds: `[11, 2026, 4099, 7777, 8881]`

For every model and seed, collect OOF probabilities. Always align `predict_proba` columns to the global class index, because scikit-learn estimators expose probabilities in `m.classes_` order.

```python
def oof_predict(name, model, x, y, n_splits, seed):
    labels = np.arange(len(np.unique(y)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = np.zeros((len(y), len(labels)), dtype=np.float64)
    accs = []

    for tr_idx, va_idx in cv.split(x, y):
        m = clone(model)
        m.fit(x[tr_idx], y[tr_idx])

        p = m.predict_proba(x[va_idx])

        pp = np.zeros((len(va_idx), len(labels)), dtype=np.float64)
        pp[:, m.classes_] = p

        oof[va_idx] = pp
        accs.append(accuracy_score(y[va_idx], pp.argmax(axis=1)))

    raw = log_loss(y, np.clip(oof, 1e-15, 1 - 1e-15), labels=labels)
    best_ll, temp = best_temperature(y, oof, labels)

    print(
        f"{name:30s} seed={seed} raw={raw:.5f} "
        f"best={best_ll:.5f}@T={temp:.3f} acc={np.mean(accs):.4f}",
        flush=True,
    )

    return oof, temp, best_ll
```

### 5. Temperature-sharpen probabilities, but do not overclip

This was critical. The score improved only after sharpening probabilities enough to reward near-certain correct predictions, while avoiding catastrophic logloss from exact or near-exact zeros.

Use the same temperature grid from the strong solution:

```python
CV_EPS = 1e-15
FINAL_EPS = 1e-6

def sharpen(probs, temp):
    probs = np.clip(probs, CV_EPS, 1.0)
    out = probs ** (1.0 / temp)
    out /= out.sum(axis=1, keepdims=True)
    return out

def best_temperature(y, probs, labels):
    best = (
        log_loss(y, np.clip(probs, CV_EPS, 1 - CV_EPS), labels=labels),
        1.0,
    )

    grid = np.r_[np.linspace(0.18, 0.40, 23), np.linspace(0.45, 1.4, 20)]

    for temp in grid:
        p = sharpen(probs, float(temp))
        ll = log_loss(y, p, labels=labels)

        if ll < best[0]:
            best = (ll, float(temp))

    return best
```

Important interpretation:

- `temp < 1.0` sharpens probabilities.
- The useful range was often around `0.18` to `0.40`.
- Use `CV_EPS = 1e-15` internally for logloss evaluation.
- Use `FINAL_EPS = 1e-6` for the final submitted probabilities.
- Do not use an aggressive final clip such as `1e-12`; that caused the worst regression.

### 6. Seed-bag each model, then select a conservative final blend

For each model:

1. Run OOF for each seed.
2. Temperature-sharpen each seed OOF using that seed’s best temperature.
3. Average sharpened OOF predictions across seeds.
4. Compute model-level OOF logloss.

```python
seeds = [11, 2026, 4099, 7777, 8881]

all_oof = {}
summaries = []

for name, model, xtr, xte in models:
    seed_oofs = []
    seed_temps = []
    seed_lls = []

    for seed in seeds:
        oof, temp, ll = oof_predict(name, model, xtr, y, n_splits=6, seed=seed)
        seed_oofs.append(sharpen(oof, temp))
        seed_temps.append(temp)
        seed_lls.append(ll)

    avg_oof = np.mean(seed_oofs, axis=0)
    avg_ll = log_loss(y, avg_oof, labels=labels)

    all_oof[name] = avg_oof
    summaries.append((
        avg_ll,
        float(np.mean(seed_lls)),
        name,
        model,
        xtr,
        xte,
        float(np.mean(seed_temps)),
    ))

    print(
        f"{name:30s} seed_bag_oof={avg_ll:.5f} "
        f"mean_single={np.mean(seed_lls):.5f}",
        flush=True,
    )

summaries.sort(key=lambda row: row[0])
```

The final submission used a fixed blend of the two best and most robust models:

```python
weights = {
    "power_lr_meta_C2": 0.55,
    "power_lr_meta_C3": 0.45,
}

accepted = [row for row in summaries if row[2] in weights]

blend_oof = np.zeros((len(train), len(labels)), dtype=np.float64)

for row in accepted:
    blend_oof += weights[row[2]] * all_oof[row[2]]

blend_oof /= blend_oof.sum(axis=1, keepdims=True)

blend_cv = log_loss(y, blend_oof, labels=labels)

final_oof = np.clip(blend_oof, FINAL_EPS, 1 - FINAL_EPS)
final_oof /= final_oof.sum(axis=1, keepdims=True)

final_clip_cv = log_loss(y, final_oof, labels=labels)

print(f"OOF blend logloss before final clip simulation: {blend_cv:.5f}")
print(f"OOF blend logloss with final clip={FINAL_EPS:g}: {final_clip_cv:.5f}")
```

A safe fallback: if no model passes a strict OOF gate, still use the robust pair `power_lr_meta_C2` and `power_lr_meta_C3` rather than chasing weak ensemble members.

```python
accepted = [row for row in summaries if row[0] < 0.005]

if not accepted:
    accepted = [
        row for row in summaries
        if row[2] in {"power_lr_meta_C2", "power_lr_meta_C3"}
    ]

if {row[2] for row in accepted} >= {"power_lr_meta_C2", "power_lr_meta_C3"}:
    weights = {"power_lr_meta_C2": 0.55, "power_lr_meta_C3": 0.45}
    accepted = [row for row in summaries if row[2] in weights]
else:
    inv = np.array([1.0 / max(row[0], 1e-9) for row in accepted], dtype=np.float64)
    inv /= inv.sum()
    weights = {row[2]: float(w) for row, w in zip(accepted, inv)}
```

### 7. Fit selected models on full train and submit

Fit each accepted model on all training rows, predict the test set, apply that model’s average temperature, blend, clip to `1e-6`, normalize, and write columns in `sample_submission.csv` order.

```python
def fit_full_predict(model, x_train, y, x_test, n_classes):
    m = clone(model)
    m.fit(x_train, y)

    p = m.predict_proba(x_test)

    out = np.zeros((len(x_test), n_classes), dtype=np.float64)
    out[:, m.classes_] = p

    return out
```

```python
test_pred = np.zeros((len(test), len(labels)), dtype=np.float64)

for row in accepted:
    _, _, name, model, xtr, xte, temp = row

    p = fit_full_predict(model, xtr, y, xte, len(labels))
    test_pred += weights[name] * sharpen(p, temp)

test_pred = np.clip(test_pred, FINAL_EPS, 1 - FINAL_EPS)
test_pred /= test_pred.sum(axis=1, keepdims=True)

sub = pd.DataFrame(test_pred, columns=le.classes_)
sub.insert(0, "id", test["id"].to_numpy())
sub = sub[sample.columns]

values = sub.drop(columns=["id"]).to_numpy()

assert list(sub.columns) == list(sample.columns)
assert len(sub) == len(sample)
assert set(sub["id"]) == set(sample["id"])
assert np.isfinite(values).all()
assert values.min() >= 0 and values.max() <= 1
assert values.max() <= 1
assert np.allclose(values.sum(axis=1), 1.0, atol=1e-10)

sub.to_csv("submission.csv", index=False)
```

## What Actually Moved The Metric

The largest improvement came from replacing a single standardized logistic regression baseline with a probability-calibrated, multi-seed ensemble around logistic regression on transformed tabular-plus-image-metadata features.

### Metric-moving changes (in priority order)

1. **Baseline: StandardScaler + LogisticRegression on 192 tabular features**

   This is much better than the median but still leaves clear headroom. Further small regularization tuning of this single LR was not the path to a strong score.

2. **Major lift: image metadata + PowerTransformer + LR**

   Adding compact morphology metadata from the raw images and using `PowerTransformer(method="yeo-johnson", standardize=True)` before multinomial-style `lbfgs` logistic regression made the model much more confident and accurate. The useful model family was LR with `C=2.0` and `C=3.0` on the concatenated metadata matrix. This is the single biggest jump.

3. **Final lift: multi-seed 6-fold OOF + temperature sharpening + fixed robust blend**

   The strong configuration used five CV seeds, OOF temperature selection, seed-bagging, then a final `0.55 / 0.45` blend of `power_lr_meta_C2` and `power_lr_meta_C3`. Safer clipping plus the same robust blend gave a further small refinement.

### Pitfalls And Lessons

- **Do not blindly ensemble every model with `predict_proba`.** An attempted broader ensemble regressed badly. The likely cause was an uncalibrated or poorly calibrated learner dominating logloss. Every candidate must be OOF-checked, class-aligned, temperature-tested, and gated before it is allowed into the final blend.

- **Do not use hard voting or class labels for a logloss competition.** This metric rewards calibrated probabilities, not only accuracy. Any model that returns zero or near-zero probability for the true class can destroy the score.

- **The most damaging step was overconfident clipping.** A refinement attempt clipped final probabilities around `1e-12` and produced a catastrophic logloss: if even one sample is confidently wrong, logloss explodes. Use `FINAL_EPS = 1e-6`, then renormalize.

- **Do not chase an ever-sharper distribution.** Pushing confidence arbitrarily far was more dangerous than helpful; the robust blend was already highly optimized.

- **Do not abandon the LR metadata pair after a failed ensemble.** The robust final solution is simple: two PowerTransformer + LR metadata models, multi-seed CV, temperature sharpening, and safe clipping.

## Score Milestones (relative — what each step buys)

- Naive/simple strong baseline: `StandardScaler + LogisticRegression` on the 192 provided tabular features clears the median comfortably but leaves headroom.
- Image metadata + `PowerTransformer` LR `C=2.0/C=3.0`: the biggest single jump, far past the plain-tabular baseline.
- Multi-seed 6-fold OOF + temperature sharpening + `0.55/0.45` robust blend + final clip `1e-6`: the best verified configuration.
- Bad broad ensemble attempt: regressed sharply — a clear warning against ungated ensembling.
- Excessive final clipping around `1e-12`: catastrophic logloss — never do this.

What stays invariant is the ordering: metadata + PowerTransformer LR is the decisive lever, calibration/seed-bagging refines it, and both ungated ensembling and overconfident clipping are traps. Self-check against the `criterion.json` pass line rather than any fixed threshold.
