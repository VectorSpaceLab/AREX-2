Task type: binary audio classification of 2-second AIFF clips for right-whale up-calls; evaluation metric: ROC AUC over submitted probabilities (higher is better). The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

## Approach

Use a robust CPU-friendly tabular audio pipeline rather than a fragile deep-learning run. The reliable solution extracts fixed-length handcrafted STFT features from every 2-second clip, trains several scikit-learn classifiers with 5-fold stratified CV, and selects the best OOF blend by ROC AUC.

1. Read `train2.zip`, `test2.zip`, and `sampleSubmission.csv` from the prepared public data directory.

2. Parse labels directly from train filenames:
   `..._0.aif` means negative and `..._1.aif` means positive.

3. Decode AIFF files from the zip without extracting them to disk.

4. Force every waveform to exactly `4000` samples, corresponding to 2 seconds at `SR = 2000`.

5. Normalize each waveform by subtracting its mean and dividing by its standard deviation.

6. Extract handcrafted spectral features using `scipy.signal.stft`:
   - `fs=2000`
   - Hann window
   - `nperseg=256`
   - `noverlap=224`
   - `nfft=512`
   - no padding
   - focus most features on the biologically relevant call band, roughly `50-400 Hz`

7. Cache extracted matrices as compressed `.npz` files because feature extraction over all zip members is the slow part.

8. Train these base models:
   - Logistic regression with standardized features:
     - `C=0.35`
     - `class_weight="balanced"`
     - `max_iter=2000`
     - `solver="lbfgs"`
   - HistGradientBoostingClassifier:
     - `learning_rate=0.035`
     - `max_iter=420`
     - `max_leaf_nodes=31`
     - `l2_regularization=0.08`
     - `early_stopping=True`
     - `random_state=11`
   - ExtraTreesClassifier:
     - `n_estimators=650`
     - `max_features="sqrt"`
     - `min_samples_leaf=2`
     - `class_weight="balanced_subsample"`
     - `n_jobs=-1`
     - `random_state=23`
   - RandomForestClassifier:
     - `n_estimators=450`
     - `max_features="sqrt"`
     - `min_samples_leaf=3`
     - `class_weight="balanced_subsample"`
     - `n_jobs=-1`
     - `random_state=37`

9. Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)` for every base model and for stack/blend evaluation.

10. Generate OOF predictions and test predictions for each model. Average fold-level test predictions per base model.

11. Evaluate:
   - each single model
   - mean ensemble
   - rank ensemble
   - logistic stacking
   - small non-negative grid blends over raw predictions and rank-transformed predictions

12. Select the candidate with the highest OOF ROC AUC and write `submission.csv` with columns from `sampleSubmission.csv`, replacing `probability`.

13. Clip final probabilities to `[1e-6, 1 - 1e-6]`.

## What Actually Moved The Metric

The strong result was reached by the simple, reliable feature-engineering plus tabular-ensemble solution, not by a completed deep-learning upgrade. The key lift came from turning each short audio clip into many stable spectral summaries around the right-whale call frequency range and then blending diverse sklearn models.

The most important metric-moving choices were:

1. Domain-focused STFT features over raw audio:
   The solution summarizes power in low-frequency bands, especially the `50-400 Hz` call band. Right-whale up-calls are frequency-sweep events, so features like call-band energy ratio, peak frequency, centroid, spread, and dominant-frequency slope matter more than generic full-band audio statistics.

2. Diversity across tabular models:
   Logistic regression, histogram gradient boosting, ExtraTrees, and RandomForest capture different aspects of the engineered features. The selected submission is chosen by OOF AUC from singles, raw blends, rank blends, and stacking, rather than assuming one model family is best.

3. Conservative CV-based model selection:
   The winning code uses one consistent 5-fold stratified split and selects the final blend using OOF ROC AUC. This avoided making blind score-chasing changes.

Pitfalls and lessons:

- The trajectory explored ideas for a log-mel CNN and richer spectrogram features to push higher, but those attempts did not produce a better confirmed submission. Some later attempts failed from infrastructure or quota interruption, not because the modeling idea was validated or invalidated.

- Do not spend the whole budget on a heavy CNN before producing a reliable submission. The single most damaging pattern was over-scoping toward a deep-learning experiment that could fail to run, leaving no improved artifact.

- Do not rely only on global audio statistics. The reliable solution’s useful signal comes from time-frequency summaries, call/noise ratios, frequency-bin snapshots, and upward-sweep descriptors.

- Do not skip filename/order handling. Test predictions must be mapped by AIFF basename into `sampleSubmission.csv`; zip order alone is not a safe submission order.

## Key Code Snippets

### AIFF Loading And Normalization

Use the AIFF reader from the standard library and handle big-endian 16-bit samples correctly. Keep clips at exactly 4000 samples.

```python
import aifc
import io
import zipfile
import numpy as np

SR = 2000

def read_aiff_from_zip(zf, name):
    with aifc.open(io.BytesIO(zf.read(name)), "rb") as f:
        raw = f.readframes(f.getnframes())
        width = f.getsampwidth()

        if width == 2:
            x = np.frombuffer(raw, dtype=">i2").astype(np.float32)
        elif width == 1:
            x = np.frombuffer(raw, dtype=np.int8).astype(np.float32)
        else:
            raise ValueError(f"unsupported sample width {width}")

    if len(x) < 4000:
        x = np.pad(x, (0, 4000 - len(x)))
    elif len(x) > 4000:
        x = x[:4000]

    x -= x.mean()
    s = x.std()
    if s > 1e-6:
        x /= s

    return x
```

### Core STFT Feature Extraction

This is the core of the solution. Preserve the STFT parameters and the low-frequency band design.

```python
from scipy import signal, stats
import numpy as np

def spectral_features(x):
    feats = []

    feats.extend([
        float(x.mean()),
        float(x.std()),
        float(np.sqrt(np.mean(x * x))),
        float(np.max(x)),
        float(np.min(x)),
        float(np.ptp(x)),
        float(np.mean(np.abs(x))),
        float(stats.skew(x)),
        float(stats.kurtosis(x)),
        float(np.mean(np.diff(np.signbit(x)))),
    ])

    freqs, times, zxx = signal.stft(
        x,
        fs=SR,
        window="hann",
        nperseg=256,
        noverlap=224,
        nfft=512,
        boundary=None,
        padded=False,
    )

    power = (np.abs(zxx) ** 2).astype(np.float32) + 1e-9
    logp = np.log1p(power)
    total = power.sum(axis=0) + 1e-9

    bands = [
        (0, 30), (30, 50), (50, 80), (80, 110),
        (110, 140), (140, 180), (180, 230),
        (230, 300), (300, 400), (400, 600),
        (600, 900), (900, 1000),
    ]

    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        b = power[m].sum(axis=0) + 1e-9
        r = b / total

        feats.extend([
            float(np.mean(np.log1p(b))),
            float(np.std(np.log1p(b))),
            float(np.max(np.log1p(b))),
            float(np.mean(r)),
            float(np.std(r)),
            float(np.max(r)),
        ])

    call_mask = (freqs >= 50) & (freqs <= 400)
    noise_mask = ~call_mask

    call_energy = power[call_mask].sum(axis=0) + 1e-9
    noise_energy = power[noise_mask].sum(axis=0) + 1e-9
    ratio = np.log(call_energy / noise_energy)

    feats.extend([
        float(ratio.mean()),
        float(ratio.std()),
        float(ratio.max()),
        float(np.percentile(ratio, 75)),
        float(np.percentile(ratio, 90)),
        float(np.mean(ratio > np.percentile(ratio, 75))),
    ])

    call_power = power[call_mask]
    call_freqs = freqs[call_mask]

    peak_idx = np.argmax(call_power, axis=0)
    peak_freq = call_freqs[peak_idx]

    centroid = (
        (call_power * call_freqs[:, None]).sum(axis=0)
        / (call_power.sum(axis=0) + 1e-9)
    )

    spread = np.sqrt(
        ((call_freqs[:, None] - centroid) ** 2 * call_power).sum(axis=0)
        / (call_power.sum(axis=0) + 1e-9)
    )

    feats.extend([
        float(peak_freq.mean()),
        float(peak_freq.std()),
        float(np.percentile(peak_freq, 10)),
        float(np.percentile(peak_freq, 90)),
        float(centroid.mean()),
        float(centroid.std()),
        float(spread.mean()),
        float(spread.std()),
    ])

    if len(peak_freq) > 1:
        slope = np.diff(peak_freq)
        feats.extend([
            float(slope.mean()),
            float(slope.std()),
            float(slope.max()),
            float(np.sum(slope > 0) / len(slope)),
            float((peak_freq[-1] - peak_freq[0]) / max(1, len(peak_freq) - 1)),
        ])
    else:
        feats.extend([0.0] * 5)

    arr = np.asarray(feats, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr
```

### Coarse Time-Frequency Snapshot Features

Add these to the end of `spectral_features`. They give the tree models useful coarse spectrogram structure without training a CNN.

```python
freq_bins = np.array_split(np.where((freqs >= 40) & (freqs <= 450))[0], 24)
time_bins = np.array_split(np.arange(logp.shape[1]), 8)

for fb in freq_bins:
    vals = logp[fb]
    feats.append(float(vals.mean()))
    feats.append(float(vals.std()))

for tb in time_bins:
    vals = logp[call_mask][:, tb]
    feats.append(float(vals.mean()))
    feats.append(float(vals.max()))

for fb in freq_bins:
    row = []
    for tb in time_bins:
        row.append(logp[fb][:, tb].mean())

    feats.extend([
        float(np.mean(row)),
        float(np.std(row)),
        float(np.max(row) - np.min(row)),
    ])
```

### Feature Matrix Caching

Cache both train and test features. This lets you rerun model/blend experiments quickly.

```python
import os
import zipfile
import numpy as np

def load_names(data_dir, zip_name):
    with zipfile.ZipFile(os.path.join(data_dir, zip_name)) as zf:
        return [i.filename for i in zf.infolist() if not i.is_dir()]

def extract_matrix(data_dir, work_dir, zip_name, cache_name):
    cache_path = os.path.join(work_dir, cache_name)

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return data["names"].tolist(), data["X"]

    names = load_names(data_dir, zip_name)
    X = []

    with zipfile.ZipFile(os.path.join(data_dir, zip_name)) as zf:
        for idx, name in enumerate(names, 1):
            X.append(spectral_features(read_aiff_from_zip(zf, name)))

            if idx % 2000 == 0:
                print(f"{zip_name}: {idx}/{len(names)}", flush=True)

    X = np.vstack(X).astype(np.float32)
    np.savez_compressed(cache_path, names=np.array(names, dtype=object), X=X)

    return names, X
```

### Base Models And CV

Use exactly this family of models and the same CV split for reproducibility.

```python
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

models = [
    (
        "logreg",
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.35,
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
            ),
        ),
    ),
    (
        "hgb",
        HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=420,
            max_leaf_nodes=31,
            l2_regularization=0.08,
            early_stopping=True,
            random_state=11,
        ),
    ),
    (
        "extratrees",
        ExtraTreesClassifier(
            n_estimators=650,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=23,
        ),
    ),
    (
        "rf",
        RandomForestClassifier(
            n_estimators=450,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=37,
        ),
    ),
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)

oofs = []
test_preds = []

for model_name, model in models:
    oof = np.zeros(len(y), dtype=np.float32)
    fold_test = []

    for fold, (tr, va) in enumerate(cv.split(X, y), 1):
        model.fit(X[tr], y[tr])

        oof[va] = model.predict_proba(X[va])[:, 1]
        fold_test.append(model.predict_proba(Xt)[:, 1])

        print(
            f"{model_name} fold {fold} auc "
            f"{roc_auc_score(y[va], oof[va]):.6f}",
            flush=True,
        )

    print(f"{model_name} OOF AUC {roc_auc_score(y, oof):.6f}", flush=True)

    oofs.append(oof)
    test_preds.append(np.mean(fold_test, axis=0))
```

### Stacking And Blend Selection

The final prediction should be selected from OOF AUC, including raw and rank-space blends. Keep the blend grid small to reduce fold overfitting.

```python
from itertools import product
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

stack_X = np.column_stack(oofs)
stack_T = np.column_stack(test_preds)

meta_oof = np.zeros(len(y), dtype=np.float32)
meta_test = []

for fold, (tr, va) in enumerate(cv.split(stack_X, y), 1):
    meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    meta.fit(stack_X[tr], y[tr])

    meta_oof[va] = meta.predict_proba(stack_X[va])[:, 1]
    meta_test.append(meta.predict_proba(stack_T)[:, 1])

mean_oof = np.mean(np.column_stack(oofs), axis=1)
rank_oof = np.mean(
    np.column_stack([pd.Series(o).rank(pct=True).to_numpy() for o in oofs]),
    axis=1,
)

print(f"mean ensemble OOF AUC {roc_auc_score(y, mean_oof):.6f}")
print(f"rank ensemble OOF AUC {roc_auc_score(y, rank_oof):.6f}")
print(f"stack ensemble OOF AUC {roc_auc_score(y, meta_oof):.6f}")

raw_oof = np.column_stack(oofs)
raw_test = np.column_stack(test_preds)

ranked_oof = np.column_stack([
    pd.Series(o).rank(pct=True).to_numpy()
    for o in oofs
])
ranked_test = np.column_stack([
    pd.Series(p).rank(pct=True).to_numpy()
    for p in test_preds
])

candidates = []
model_names = [m[0] for m in models]

for i, name in enumerate(model_names):
    candidates.append((roc_auc_score(y, raw_oof[:, i]), f"single_{name}", raw_test[:, i]))
    candidates.append((roc_auc_score(y, ranked_oof[:, i]), f"single_rank_{name}", ranked_test[:, i]))

grid = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]

for weights in product(grid, repeat=len(models)):
    w = np.asarray(weights, dtype=np.float32)

    if w.sum() <= 0:
        continue

    w /= w.sum()

    for prefix, oo, tt in [
        ("raw", raw_oof, raw_test),
        ("rank", ranked_oof, ranked_test),
    ]:
        blend_oof = oo @ w
        auc = roc_auc_score(y, blend_oof)
        candidates.append((auc, f"{prefix}_blend_{','.join(f'{x:.2f}' for x in w)}", tt @ w))

best_auc, best_name, pred = max(candidates, key=lambda x: x[0])

print(f"selected {best_name} OOF AUC {best_auc:.6f}", flush=True)
```

### Submission Mapping

Map by basename into the sample submission. Do not assume test zip order matches submission order.

```python
import os
import re
import pandas as pd
import numpy as np

y = np.array(
    [int(re.search(r"_([01])\.aif$", n).group(1)) for n in train_names],
    dtype=np.int64,
)

clip_to_pred = {
    os.path.basename(n): float(p)
    for n, p in zip(test_names, pred)
}

sub = pd.read_csv(os.path.join(DATA_DIR, "sampleSubmission.csv"))
sub["probability"] = sub["clip"].map(clip_to_pred).astype(float)

if sub["probability"].isna().any():
    missing = sub.loc[sub["probability"].isna(), "clip"].head().tolist()
    raise RuntimeError(f"missing predictions for {missing}")

eps = 1e-6
sub["probability"] = np.clip(sub["probability"], eps, 1 - eps)
sub.to_csv("submission.csv", index=False)
```

## Score Milestones (relative — what each step buys)

Naive or weak baselines sit around the competition median and leave clear headroom. The strong configuration — handcrafted STFT/domain features plus a 5-fold sklearn ensemble/blend selection — is a clear step up from those baselines and is the reliable target.

What stays invariant is the ordering: domain-focused STFT features are the biggest lever, model diversity plus OOF-selected blending adds the rest, and over-scoping to an unfinished CNN is a trap that leaves no artifact. Self-check against the `criterion.json` pass line rather than any fixed threshold.
