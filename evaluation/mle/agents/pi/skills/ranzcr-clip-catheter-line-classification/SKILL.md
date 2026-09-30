RANZCR CLiP catheter/line multi-label chest X-ray classification; metric is macro ROC-AUC over 9 scored targets (higher is better). The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

**Task Objective**

Build a high-quality submission for `ranzcr-clip-catheter-line-classification`. The public labels contain 11 targets, but the score is driven by the first 9 catheter/line position targets:

```python
TARGETS = [
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
]
```

The best known trajectory landed just short of the pass line. The next solver should not restart from a plain classifier. Start from annotation-aware segmentation-auxiliary models plus complete OOF endpoint ensembling, then add genuinely new full-fold signal.

**Best Starting Approach**

1. Use `train.csv`, `sample_submission.csv`, and `train_annotations.csv`.

2. Train annotation-aware segmentation-auxiliary image classifiers, not plain classification-only EfficientNet.

3. Use PatientID-grouped 5-fold CV. The winning trajectory explicitly checked that fold validation patients do not overlap train patients.

4. Use the scored 9 targets for submission optimization. Keep the 11-label head only if the training script already supports it, but always evaluate/report `AUC9` on the 9 scored targets.

5. Render `train_annotations.csv` into line masks for auxiliary supervision. Prior attempts used a line-width around `60` and grouped masks by catheter family such as ETT / NGT / CVC. The important point is not a perfect mask architecture; it is that the backbone learns catheter localization.

6. Strong base endpoint:
   - Backbone: `resnet200d`
   - Image size: `768`
   - 5 folds complete
   - Segmentation auxiliary branch enabled
   - Test-time augmentation enabled, at minimum horizontal flip
   - Output files following the proven pattern:
     - `oof_seg_aux.npy`
     - `oof_done.npy`
     - `test_pred_fold*_best.npy`

7. Add diverse complete endpoints:
   - `seresnet152d_640_s42`, complete 5-fold endpoint
   - `dm_nfnet_f1`, complete 5-fold endpoint if possible
   - A second independently trained `resnet200d` seed can help, but it is not enough by itself
   - `tf_efficientnet_b7.ap_in1k@640` was tried for only fold0 / 3 epochs and was too weak; it should be continued to 8-10 epochs and completed across folds before being trusted

8. Use OOF-driven rank ensembling rather than probability averaging alone. Preserve the probability marginals of the best known high-scoring floor submission and only remap ranks using complete OOF-supported endpoints.

9. Keep CVC columns conservative. Prior CVC OOF gains repeatedly failed to transfer to real feedback. The best postprocessing locked CVC unchanged and only moved ETT/NGT ranks slightly.

**Known Good Hyperparameters / Training Evidence**

The actual retained final diagnostic code records this fresh endpoint:

```text
model: tf_efficientnet_b7.ap_in1k
size: 640
fold: 0
epochs run: 3
seed: 3031
batch_size: 10
loss style: Focal + Dice for segmentation auxiliary training
CV: PatientID grouped
AMP: enabled
scheduler: cosine LR
TTA: horizontal flip
```

This B7 endpoint was not strong enough as trained:

```text
fold0 epoch-best AUC9: first working single-fold level
fold0 TTA partial OOF (AUC9/AUC11): TTA adds a small consistent lift
```

So do not blend a weak partial endpoint heavily. The next agent should continue this model or train a better fresh endpoint until partial/full OOF quality is at least around the median-quality regime, about `AUC9 >= 0.9675`, before using it materially.

**What Actually Moved The Metric**

The important lifts were:

1. Plain `tf_efficientnet_b5_ns@512` single model plateaued at a low single-model ceiling. Many attempts repeated this and did not help.

2. The first real breakthrough came from completing the annotation-aware segmentation-auxiliary route instead of undertraining it:
   - undertrained seg-aux endpoints around 1-5 epochs had OOF below the B5 baseline and were ignored by blending
   - a properly trained seg-aux `resnet200d` endpoint was the single biggest modeling jump
   - completing / improving that route added a further step

3. Ensembling and TTA pushed the trained seg-aux pool further:
   - seg-aux ensemble/TTA lifted the pool
   - adding a diverse `seresnet152d_640_s42` style endpoint and careful rank blending added a small increment
   - the best trajectory landed just short of the pass line

The real lesson: the solution is signal-limited, not postprocessing-limited. Once a strong floor exists, tiny rank nudges add almost nothing. To cross the pass line, add a genuinely new, complete, held-out-compatible endpoint.

**Pitfalls / Lessons**

Do not repeat these failed paths:

1. Do not rerun a plain B5/B7 classification model without annotation auxiliary training. This repeatedly scored at a low ceiling, far below the passing range.

2. Do not build a multi-stage teacher-student pseudo-label pipeline that spends CPU time generating soft masks or predictions without GPU utilization. Earlier versions repeatedly stalled at GPU 0% for long periods and were killed.

3. Do not submit partial-fold models as if they were full endpoints. Partial OOF is not comparable to complete 5-fold OOF and caused blend weights to collapse or regress.

4. Do not endlessly tune blend weights, rank shrink, column picks, or positive/negative nudges against the same endpoint pool. Once a strong floor existed, post-hoc probability/rank perturbations were capped and could not cross the pass line.

5. The single most damaging pattern was restarting new pipelines every attempt and training only 1-5 epochs. That created weak endpoints, misleading OOF comparisons, and forced repeated fallback to the same floor submission.

6. Treat CVC with suspicion. Multiple OOF-guided CVC moves did not transfer. Prefer leaving CVC unchanged unless a new complete endpoint shows strong OOF and real feedback confirms it.

**Implementation Skeleton**

Use the following as the core target/validation structure.

```python
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA_DIR = Path("./input")

TARGETS = [
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
]

train = pd.read_csv(DATA_DIR / "train.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
y = train[TARGETS].to_numpy(np.float64)
```

Patient-grouped folds are mandatory. The exact split implementation was not present in the final postprocessing code, but the retained diagnostic report states that all folds asserted zero `PatientID` overlap. Use this pattern in the training script:

```python
from sklearn.model_selection import GroupKFold

train = pd.read_csv(DATA_DIR / "train.csv")
groups = train["PatientID"].values

folds = np.full(len(train), -1, dtype=np.int32)
gkf = GroupKFold(n_splits=5)

for fold, (_, val_idx) in enumerate(gkf.split(train, train[TARGETS], groups)):
    folds[val_idx] = fold

for fold in range(5):
    tr_patients = set(train.loc[folds != fold, "PatientID"])
    va_patients = set(train.loc[folds == fold, "PatientID"])
    assert not (tr_patients & va_patients), f"PatientID leakage in fold {fold}"
```

The model should be a classifier with a segmentation auxiliary branch. Keep this minimal and reliable; the historical winning path benefited from annotation localization, not from complicated teacher stages.

```python
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class SegAuxClassifier(nn.Module):
    def __init__(self, model_name="resnet200d", num_targets=11, seg_channels=3, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,),
        )
        ch = self.encoder.feature_info.channels()[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(ch, num_targets)

        self.seg_head = nn.Sequential(
            nn.Conv2d(ch, ch // 2, 3, padding=1),
            nn.BatchNorm2d(ch // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch // 2, seg_channels, 1),
        )

    def forward(self, x):
        feat = self.encoder(x)[-1]
        logits = self.cls_head(self.pool(feat).flatten(1))
        seg_logits = self.seg_head(feat)
        seg_logits = F.interpolate(seg_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits, seg_logits
```

Use BCE/focal classification plus Dice-style segmentation loss. The final diagnostic explicitly records Focal+Dice for the B7AP seg-aux endpoint.

```python
def focal_bce_with_logits(logits, targets, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    return (bce * (1 - pt).pow(gamma)).mean()

def dice_loss_with_logits(seg_logits, masks, eps=1e-6):
    prob = torch.sigmoid(seg_logits)
    dims = (2, 3)
    inter = (prob * masks).sum(dims)
    union = prob.sum(dims) + masks.sum(dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()

def total_loss(cls_logits, y, seg_logits, masks, seg_weight=1.0):
    cls_loss = focal_bce_with_logits(cls_logits, y)
    seg_loss = dice_loss_with_logits(seg_logits, masks)
    return cls_loss + seg_weight * seg_loss
```

For each complete endpoint, save OOF predictions, a boolean done mask, and per-fold test predictions:

```python
# For one endpoint directory:
# oof shape: (len(train), >=9)
# done shape: (len(train),)
# test preds: one file per fold, shape (len(sample), >=9)
np.save("oof_seg_aux.npy", oof)
np.save("oof_done.npy", done)
np.save(f"test_pred_fold{fold}_best.npy", test_pred)
```

Compute OOF AUC on the scored 9 targets. Refuse to use incomplete OOF as a major endpoint.

```python
done = np.load("oof_done.npy").astype(bool)
oof = np.load("oof_seg_aux.npy")[:, :len(TARGETS)]

assert done.all(), "Do not major-blend incomplete OOF endpoints"

auc_by_col = {
    col: float(roc_auc_score(train[col].values, oof[:, j]))
    for j, col in enumerate(TARGETS)
}
auc9 = float(np.mean(list(auc_by_col.values())))
print("AUC9", auc9, auc_by_col)
```

**Rank Utility Code**

This condensed code is directly from the final best postprocessing logic. It converts predictions to stable per-column ranks.

```python
def rank01_1d(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), dtype=np.float64)
    out[order] = (np.arange(len(x), dtype=np.float64) + 0.5) / len(x)
    return out

def rank_cols(a: np.ndarray) -> np.ndarray:
    return np.stack([rank01_1d(a[:, j]) for j in range(a.shape[1])], axis=1)

def avg_test(paths):
    paths = list(paths)
    assert paths, "empty test paths"
    return np.mean([np.load(p)[:, :len(TARGETS)] for p in paths], axis=0)
```

**Endpoint Loader**

This is the proven pattern for loading only complete 5-fold endpoints.

```python
def load_endpoint(name: str, directory: str, oof_name: str, done_name: str, test_glob: str):
    d = Path(directory)
    oof = np.load(d / oof_name)[:, :len(TARGETS)]
    done = np.load(d / done_name).astype(bool)
    assert done.all(), f"{name} OOF is not complete"

    test = avg_test(sorted(d.glob(test_glob)))

    return {
        "name": name,
        "dir": str(d),
        "oof": oof.astype(np.float64),
        "test": test.astype(np.float64),
        "oof_rank": rank_cols(oof),
        "test_rank": rank_cols(test),
    }
```

Known useful endpoint pool from the best trajectory:

```python
endpoints = [
    load_endpoint(
        "a30_segaux",
        "./a30_resnet200d_segaux",            # a directory an earlier stage of this recipe wrote
        "oof_seg_aux.npy",
        "oof_done.npy",
        "test_pred_fold*_best.npy",
    ),
    load_endpoint(
        "a34_seresnet_s42",
        "./a34_seresnet152d_640_s42",            # a directory an earlier stage of this recipe wrote
        "oof_seresnet152d_640_s42.npy",
        "oof_seresnet152d_640_s42_done.npy",
        "test_pred_seresnet152d_640_s42_fold*_best.npy",
    ),
    load_endpoint(
        "a49_nfnet_f1",
        "./a49_nfnet_f1",            # a directory an earlier stage of this recipe wrote
        "oof_seg_aux.npy",
        "oof_done.npy",
        "test_pred_nfnet_f1_fold*_best.npy",
    ),
    load_endpoint(
        "a28_resnet",
        "./a28_resnet200d_segaux",            # a directory an earlier stage of this recipe wrote
        "oof_seg_aux.npy",
        "oof_done.npy",
        "test_pred_fold*_best.npy",
    ),
]
```

**Best Known Rank-Remap Ensemble**

The best code did not replace the trusted floor probabilities. It preserved each floor column’s sorted probability distribution and only changed ordering slightly using rank consensus from complete OOF endpoints.

```python
floor = pd.read_csv(FLOOR_PATH).set_index("StudyInstanceUID").loc[
    sample.StudyInstanceUID
].reset_index()

assert list(floor.columns) == list(sample.columns)
assert floor.StudyInstanceUID.equals(sample.StudyInstanceUID)

base_vals = floor[TARGETS].to_numpy(np.float64)
base_rank = rank_cols(base_vals)
out = floor.copy()

by_name = {e["name"]: e for e in endpoints}

column_sources = {
    "ETT - Abnormal": ["a30_segaux", "a49_nfnet_f1"],
    "ETT - Borderline": ["a49_nfnet_f1", "a28_resnet"],
    "ETT - Normal": ["a49_nfnet_f1", "a28_resnet"],
    "NGT - Abnormal": ["a34_seresnet_s42", "a30_segaux"],
    "NGT - Borderline": ["a34_seresnet_s42", "a30_segaux"],
    "NGT - Incompletely Imaged": ["a28_resnet"],
    "NGT - Normal": ["a28_resnet"],
    "CVC - Abnormal": [],
    "CVC - Borderline": [],
}

weights = {
    "ETT - Abnormal": 0.050,
    "ETT - Borderline": 0.040,
    "ETT - Normal": 0.020,
    "NGT - Abnormal": 0.050,
    "NGT - Borderline": 0.050,
    "NGT - Incompletely Imaged": 0.020,
    "NGT - Normal": 0.020,
    "CVC - Abnormal": 0.0,
    "CVC - Borderline": 0.0,
}

selected_oof = np.zeros((len(train), len(TARGETS)), dtype=np.float64)
selected_test = np.zeros((len(sample), len(TARGETS)), dtype=np.float64)

for j, col in enumerate(TARGETS):
    src = column_sources[col]

    if src:
        selected_oof[:, j] = np.mean([by_name[n]["oof_rank"][:, j] for n in src], axis=0)
        selected_test[:, j] = np.mean([by_name[n]["test_rank"][:, j] for n in src], axis=0)
    else:
        selected_test[:, j] = base_rank[:, j]

    w = weights[col]
    if w > 0:
        mixed = (1.0 - w) * base_rank[:, j] + w * selected_test[:, j]
        order = np.argsort(mixed, kind="mergesort")

        remapped = np.empty(len(order), dtype=np.float64)
        remapped[order] = np.sort(base_vals[:, j])

        out[col] = remapped

out[TARGETS] = np.clip(out[TARGETS].to_numpy(np.float64), 1e-6, 1 - 1e-6)
out.to_csv("submission.csv", index=False)
```

This exact strategy was intended as a conservative refinement over a known high-scoring floor. It is useful only after strong endpoints exist. It is not a substitute for training a new strong endpoint.

**Tiny Partial-Endpoint NGT Remap**

A fresh `tf_efficientnet_b7.ap_in1k@640` fold0 endpoint was trained, but its global partial OOF was weak. The only attempted safe use was a tiny NGT-only rank remap.

```python
weights = {
    "ETT - Abnormal": 0.0,
    "ETT - Borderline": 0.0,
    "ETT - Normal": 0.0,
    "NGT - Abnormal": 0.0040,
    "NGT - Borderline": 0.0040,
    "NGT - Incompletely Imaged": 0.0020,
    "NGT - Normal": 0.0020,
    "CVC - Abnormal": 0.0,
    "CVC - Borderline": 0.0,
}

base_vals = floor[TARGETS].to_numpy(np.float64)
new_vals = raw_new[TARGETS].to_numpy(np.float64)

for j, col in enumerate(TARGETS):
    w = weights[col]
    if w <= 0:
        continue

    blended_rank = (1.0 - w) * rank01_1d(base_vals[:, j]) + w * rank01_1d(new_vals[:, j])
    order = np.argsort(blended_rank, kind="mergesort")

    remapped = np.empty(len(order), dtype=np.float64)
    remapped[order] = np.sort(base_vals[:, j])

    out[col] = remapped
```

This should be treated as a diagnostic fallback, not the main plan. If the new endpoint has `AUC9 < 0.9675`, keep weights tiny or do not use it.

**Submission Checks**

Always verify format before submission.

```python
import hashlib

def check_submission(path: Path, sample: pd.DataFrame) -> dict:
    sub = pd.read_csv(path)
    vals = sub[TARGETS].to_numpy(np.float64)

    return {
        "shape": list(sub.shape),
        "columns_equal_sample": list(sub.columns) == list(sample.columns),
        "ids_equal_sample": bool(sub.StudyInstanceUID.equals(sample.StudyInstanceUID)),
        "finite": bool(np.isfinite(vals).all()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "nunique_min": int(sub[TARGETS].nunique().min()),
        "sha256_12": hashlib.sha256(path.read_bytes()).hexdigest()[:12],
    }

report = check_submission(Path("submission.csv"), sample)
assert report["columns_equal_sample"]
assert report["ids_equal_sample"]
assert report["finite"]
```

**Recommended Next Push**

The best known path is short of the pass line by a small margin, but postprocessing is exhausted. The next solver should do this:

1. Keep the known high-scoring floor submission as an anchor.

2. Continue or retrain a genuinely diverse endpoint:
   - `tf_efficientnet_b7.ap_in1k@640` from the existing fold0 checkpoint if available
   - train fold0 to 8-10 epochs, not 3
   - increase batch size if GPU memory allows; the diagnostic saw only about 1.15GB allocated on an A100 for batch 10, so there was headroom
   - consider `704` or `768` resolution after throughput is confirmed
   - keep PatientID-group CV, annotation masks, Focal+Dice, AMP, cosine LR, and hflip TTA

3. Do not complete folds 1-4 until fold0 reaches a credible sanity target. Aim for fold0 TTA `AUC9 >= 0.9675`.

4. Once the new endpoint is credible, complete all 5 folds and save:
   - full OOF
   - full OOF done mask
   - fold test predictions
   - per-column OOF AUC report

5. Blend only with complete OOF support. First test ETT/NGT rank remap and keep CVC locked. Only move CVC if OOF and real submissions both confirm it.

6. If compute allows, train another full diverse seed rather than tuning CSV weights. A new complete endpoint with even small independent signal is more valuable than another rank-remap variant.

**Score Milestones (relative — what each step buys)**

Naive / early baseline:
```text
tf_efficientnet_b5_ns@512 single model: a strong single-backbone level
```

First useful annotation-aware breakthrough:
```text
properly trained resnet200d seg-aux endpoint: a clear step up (segmentation-aux is the single biggest modeling lever)
improved/continued seg-aux: marginal further gain
```

Strong ensemble range:
```text
seg-aux ensemble + TTA: higher
resnet200d + seresnet152d/NFNet-style endpoint rank refinements: further small gain
best known level: the task is signal-limited near here
```

Pass line:
```text
Self-check against the `criterion.json` pass line for this run — not fixed leaderboard thresholds.
```

Practical self-estimate:
```text
If an endpoint's OOF AUC9 is clearly below the ensemble level: it is not ready for major blending.
Once the base ensemble works but lacks diversity: postprocessing gains are nearly exhausted.
To push further: train a new complete, annotation-aware, private-compatible endpoint and blend it conservatively by OOF-supported ranks.
```

Self-check against the `criterion.json` pass line, not a fixed threshold. What stays invariant is the ORDERING: a plain classifier plateaus at a low ceiling; the annotation-aware segmentation-auxiliary `resnet200d` route is the single biggest modeling jump; ensemble/TTA and diverse complete endpoints add smaller steps; and once a strong floor exists the solution is signal-limited, so crossing requires a genuinely new complete endpoint, not more rank-remap tuning. Keep CVC locked unless a new endpoint's OOF and real feedback both confirm a move.
