Task: APTOS 2019 Blindness Detection — grade diabetic retinopathy severity `0..4` (ordinal) from a single retinal fundus image. Metric: **Quadratic Weighted Kappa (QWK), HIGHER is better**. The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

This recipe is SELF-CONTAINED: it trains every fold from scratch. It does NOT load cached artifacts or an `oof.npy` from some earlier attempt — an earlier version of this skill did exactly that, and **that is why the run crashed every time**: the path does not exist on a fresh machine. Build every array yourself.

Data lives ONLY under `prepared/public/`. NEVER read `prepared/private/`. NEVER join test labels. (Training on external 2015 Diabetic-Retinopathy images is allowed; never use any external data for test-label lookup.)

```python
import os
from pathlib import Path
DATA_DIR = Path("./input")
# Files: train.csv (id_code,diagnosis), sample_submission.csv (id_code,diagnosis),
#        train_images/<id_code>.png , test_images/<id_code>.png
```

## Approach

The method is a **multi-seed/backbone ENSEMBLE of `tf_efficientnet_b4/b5 @ 380px+, Ben-Graham preprocessed, trained as scalar REGRESSION over 5 folds, with OOF-optimized thresholds re-fit on the combined OOF**. A *single* such model plateaus at a single-model ceiling — that is exactly where a single B4 + good thresholds stalls, short of the pass line. Crossing needs the *combination* — Ben-Graham + strong backbone + OOF-optimized thresholds + a k-fold/seed ENSEMBLE (and optionally 2015 DR pretraining). The plan below STAGES that ensemble: build one solid model, then add a fresh seed/backbone variant and re-optimize thresholds on the growing OOF. Do NOT chase further gains with heavy distribution-mismatched fusion — earlier attempts tried SE-ResNeXt50 + rank-fusion twice and regressed despite a self-estimated OOF that looked much higher. The ensemble here means *same-family seed/backbone averaging of continuous preds*, which is stable, not exotic rank-fusion of mismatched models.

Numbered recipe (each stage MUST end with a valid `submission.csv`; keep the previous best as fallback and only overwrite when OOF QWK improves):

1. Read `train.csv`, `sample_submission.csv`, `train_images/`, `test_images/`. Preserve `train.csv` row order so `oof[i]` aligns with `train.iloc[i]`.
2. **Preprocessing = Ben-Graham** (circle-crop to the retina, then weighted Gaussian-blur subtraction for illumination normalization), then resize to the model resolution. This is the single biggest non-model lever for APTOS; see snippet. Cache cropped arrays to disk once if I/O is slow.
3. **Backbone:** `timm.create_model("tf_efficientnet_b4.ns_jft_in1k", pretrained=True, num_classes=1)`. **ROBUSTNESS:** wrap `pretrained=True` in try/except. If the weight download fails (offline box — a likely crash), fall back through `tf_efficientnet_b4` → `tf_efficientnet_b3` → `efficientnet_b3` and finally `pretrained=False`. A from-scratch B3 still clears the median; never let a weight download hard-crash the run.
4. **Resolution:** 380 for B4. If GPU memory is tight at batch-size ≥ 8, drop to 300-320px or B3 rather than OOM-crashing. Use `batch_size` you can fit; enable AMP (`torch.cuda.amp`).
5. **Target = scalar regression** on labels `0..4` (NOT 5-way softmax). Single-output head, `MSELoss` (SmoothL1 also fine). Regression + threshold-opt beats softmax+argmax here.
6. **5-fold** `StratifiedKFold(5, shuffle=True, random_state=42)` on `diagnosis`. For each fold save the val preds into `oof`, and predict the full test set; average test preds over folds.
7. **Threshold optimization (MANDATORY):** optimize exactly 4 cut points on the *combined-ensemble* OOF preds for QWK (Nelder-Mead from `[0.5,1.5,2.5,3.5]`, sort after each step). Apply the SAME fixed thresholds to the combined test preds. This alone is worth several thousandths of QWK — it is NOT optional, and it must be re-fit on the growing OOF, not carried over stale. Never tune thresholds on public feedback.
8. **ENSEMBLE across variants (the lever most often missed).** Each variant saves its own continuous OOF/test `.npy`; the submission is built from the **mean of ALL variants' continuous preds** (TTA-averaged), with thresholds re-optimized on that combined OOF. Persist per-variant `.npy` so a later run reads earlier variants' saved preds and averages in its new one.
9. **Save artifacts every stage:** per-variant `oof_v{k}.npy` + `test_v{k}.npy` (continuous), the combined `oof.npy`/`test_preds.npy`, `thresholds.npy`, `report.json` (with `oof_fixed_threshold_qwk`, `oof_optimized_threshold_qwk`, and the per-variant + combined OOF QWK). Always write `submission.csv` from the current best combined `test_preds`.

STAGED PLAN — the plan front-loads thresholds and stages ONE NEW seed/backbone variant per stage so the ensemble actually grows; under-building the ensemble is exactly why a single calibrated model stalls:
- **Stage 1 (must not crash):** ONE solid variant end-to-end — Ben-Graham + B4-380 (or B3 fallback) regression, 5-fold, ~5-6 epochs/fold, **with hflip TTA on val+test**, then **OOF threshold-opt (mandatory, not deferred)**, valid `submission.csv`. Save `oof_v0.npy`/`test_v0.npy` and the combined `.npy`. If the 5-fold B4 won't finish in budget, train fewer epochs or use B3@300; a finished valid submission beats a timed-out B4. *Threshold optimization belongs in this first stage — it is worth real QWK and must not be skipped.*
- **Stage 2 (add variant 1):** train a **second seed** (`random_state=43`) B4-380, OR a **different backbone** (`tf_efficientnet_b5.ns_jft_in1k` @ 456 if memory allows, else B3-380). Save `oof_v1.npy`/`test_v1.npy`. **Average continuous OOF/test over v0+v1**, re-optimize thresholds on the combined OOF. The seed/backbone diversity averaging — not TTA — is what breaks the single-model plateau. Only overwrite submission if combined OOF QWK improved.
- **Stage 3 (add variant 2):** add a **third variant** — another seed or backbone not yet used (B5-456, B4 seed 44, or B3-456). Save `oof_v2`/`test_v2`, average over v0+v1+v2, re-optimize thresholds. Three-variant averaging is the realistic crossing point (a single calibrated B4 sits just below it; a 3-variant ensemble should match-or-beat it).
- **Stage 4 (add the 2015-pretrain variant):** train one more variant whose backbone is **2015-Diabetic-Retinopathy pretrained then APTOS-fine-tuned** (this lifted the original standalone score and adds real diversity to the ensemble). **ROBUSTNESS:** wrap the 2015 download/read in try/except; if the external data is unavailable, fall back to an ImageNet-pretrained extra seed instead and log it — never crash. Save `oof_v3`/`test_v3`, average all variants, re-optimize thresholds.

**Every stage: keep the previous combined `test_preds.npy` + `thresholds.npy` as fallback; re-fit thresholds on the new combined OOF; only overwrite `submission.csv` when the new combined OOF QWK ≥ saved best.**

## What Actually Moved The Metric

From the recorded trajectory (relative ordering of what each lever buys):

1. **B3 single regression** — below the pass line. Baseline.
2. **B3 + APTOS-2015 pretraining/data** — a small lift. The 2015 DR data is the classic APTOS lever; still short on its own.
3. **B4 @ 380px + Ben-Graham + 5-fold regression + OOF-optimized thresholds** — the decisive jump. This combination — not any single piece — is what crosses.
4. **Threshold optimization was critical:** the final used calibrated cut points like `[0.5737,1.4076,2.3139,3.1389]`, NOT the naive `[0.5,1.5,2.5,3.5]`. Calibrated cut points alone are worth several thousandths of QWK on an ordinal task.
5. **Fusion HURT:** post-crossing, SE-ResNeXt50-456 + rank-fusion scored high on OOF but dropped sharply on public; a second fusion attempt also failed back. Distribution-mismatched ensembles overfit the OOF. Prefer a clean calibrated B4 + light TTA/seed-averaging.
6. **Most early failures were infra (codex 403/502), not method** — the B4-380 plan is sound; just make it actually finish.

Repro-specific warnings: a plain single B4/B3 caps at a single-model ceiling — good thresholds on a single model still fall short. **The missing lever is the ENSEMBLE, not the thresholds (which were already applied) and not 2015-pretrain alone.** Thresholds + 2015-pretrain each buy thousandths but a single model still falls short; only stacking 2-3 seed/backbone variants (continuous-pred averaging) + re-optimized thresholds reliably crosses the pass line. A 3-4 variant ensemble of comparable backbones should match-or-exceed the best single calibrated B4.

## Key Code Snippets

### Ben-Graham preprocessing (circle-crop + illumination normalization)

```python
import cv2, numpy as np

def crop_to_circle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > 7
    if mask.sum() == 0:
        return img
    ys, xs = np.where(mask)
    img = img[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = img.shape[:2]
    s = min(h, w)
    cy, cx = h // 2, w // 2
    img = img[max(0,cy-s//2):cy+s//2, max(0,cx-s//2):cx+s//2]
    return img

def ben_graham(path, size=380, sigma_scale=10):
    img = cv2.imread(str(path))
    if img is None:
        return np.zeros((size, size, 3), np.uint8)   # never crash on a bad image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_to_circle(img)
    img = cv2.resize(img, (size, size))
    blur = cv2.GaussianBlur(img, (0, 0), size / sigma_scale)
    img = cv2.addWeighted(img, 4, blur, -4, 128)
    # circular mask to kill corner artifacts
    m = np.zeros((size, size), np.uint8)
    cv2.circle(m, (size//2, size//2), int(size//2*0.9), 1, -1, 8)
    img = img * m[..., None] + 128 * (1 - m[..., None])
    return img.astype(np.uint8)
```

### Model with offline-safe pretrained fallback (fixes the other likely crash)

```python
import timm, torch.nn as nn

def build_model():
    for name in ["tf_efficientnet_b4.ns_jft_in1k", "tf_efficientnet_b4",
                 "tf_efficientnet_b3", "efficientnet_b3"]:
        try:
            return timm.create_model(name, pretrained=True, num_classes=1), name
        except Exception as e:
            print(f"[warn] pretrained {name} failed: {e}")
    print("[warn] all pretrained downloads failed -> from scratch")
    return timm.create_model("tf_efficientnet_b3", pretrained=False, num_classes=1), "scratch_b3"

criterion = nn.MSELoss()   # regression target = float(diagnosis)
```

### QWK threshold optimization (on OOF only)

```python
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score

def labels_from_thresholds(preds, t):
    return np.digitize(np.clip(preds, 0, 4), np.sort(t)).astype(np.int64)

def optimize_thresholds(y_true, oof):
    def neg_qwk(t):
        return -cohen_kappa_score(y_true, labels_from_thresholds(oof, t), weights="quadratic")
    res = minimize(neg_qwk, [0.5,1.5,2.5,3.5], method="Nelder-Mead")
    return np.sort(res.x).astype(np.float32)
# Reference cut points from the source solution (good warm-start / sanity check):
# [0.5736805746173068, 1.4075554674018627, 2.313896735299941, 3.13888165029665]
```

### TTA — average continuous preds over flips

```python
import torch
@torch.no_grad()
def predict_tta(model, x):                  # x: (B,3,H,W) normalized
    model.eval()
    outs = [model(x), model(torch.flip(x, [3])), model(torch.flip(x, [2]))]
    return torch.stack(outs).mean(0).squeeze(1)
```

### Per-variant training + cross-variant ensemble (THE missing lever)

Each stage trains ONE variant (a 5-fold seed or a backbone) and saves its own
continuous preds; the submission is the mean of ALL variants saved so far.

```python
from sklearn.model_selection import StratifiedKFold

# --- train one variant this round (k=round index): a single seed/backbone, 5-fold ---
def train_variant(k, seed, backbone):
    oof = np.zeros(len(train), np.float32)
    test_preds = np.zeros(len(sample), np.float32)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for f, (tr, va) in enumerate(skf.split(train, train["diagnosis"])):
        model = train_one_fold(tr, va, backbone)         # returns trained model
        oof[va]    = predict_tta_continuous(model, va_loader)   # TTA-averaged
        test_preds += predict_tta_continuous(model, test_loader) / 5
    np.save(f"oof_v{k}.npy", oof); np.save(f"test_v{k}.npy", test_preds)
    return oof, test_preds

# Round 1: train_variant(0, 42, "tf_efficientnet_b4.ns_jft_in1k")
# Round 2: train_variant(1, 43, "tf_efficientnet_b4.ns_jft_in1k")  # or b5/b3
# Round 3: train_variant(2, 44, "tf_efficientnet_b5.ns_jft_in1k")  # or another seed
# Round 4: a 2015-pretrained variant (see maybe_pretrain_2015), else extra seed

# --- combine EVERY round: average all variant preds present on disk ---
def combine_variants():
    ks = sorted(int(p.stem.split("v")[1]) for p in Path(".").glob("oof_v*.npy"))
    oof  = np.mean([np.load(f"oof_v{k}.npy")  for k in ks], axis=0).astype(np.float32)
    test = np.mean([np.load(f"test_v{k}.npy") for k in ks], axis=0).astype(np.float32)
    np.save("oof.npy", oof); np.save("test_preds.npy", test)
    return oof, test
# Then re-optimize thresholds on the COMBINED oof (mandatory) and build submission.
# (rank-average is a safe alt if a backbone's scale differs; plain mean is fine for same family)
```

### Submission — self-contained, validated, best-kept-as-fallback

```python
import json, pandas as pd
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
train  = pd.read_csv(DATA_DIR / "train.csv")
assert list(sample.columns) == ["id_code", "diagnosis"]

oof, test_preds = combine_variants()        # mean of all variants saved this & prior rounds
thr = optimize_thresholds(train["diagnosis"].to_numpy(), oof)   # MANDATORY, re-fit each round
oof_qwk = cohen_kappa_score(train["diagnosis"], labels_from_thresholds(oof, thr), weights="quadratic")

# only overwrite if we improved on carried-forward best
best_path = Path("best_oof_qwk.json")
prev = json.loads(best_path.read_text())["qwk"] if best_path.exists() else -1.0
if oof_qwk >= prev:
    labels = labels_from_thresholds(test_preds, thr)
    sub = sample.copy(); sub["diagnosis"] = labels.astype(int)
    assert sub["id_code"].equals(sample["id_code"])
    assert not sub.isna().any().any() and sub["diagnosis"].between(0,4).all()
    sub.to_csv("submission.csv", index=False)
    np.save("oof.npy", oof); np.save("test_preds.npy", test_preds); np.save("thresholds.npy", thr)
    best_path.write_text(json.dumps({"qwk": float(oof_qwk),
        "thresholds": thr.tolist(),
        "test_label_counts": sub["diagnosis"].value_counts().sort_index().to_dict()}))
    print(f"[ok] new best OOF QWK={oof_qwk:.5f}; wrote submission.csv")
else:
    print(f"[keep] OOF QWK={oof_qwk:.5f} < best {prev:.5f}; kept previous submission.csv")
```

### Optional 2015 DR pretraining — wrapped, skip-if-unavailable

```python
def maybe_pretrain_2015(model):
    try:
        # obtain the 2015 Diabetic-Retinopathy train images + trainLabels.csv
        # (kaggle competition 'diabetic-retinopathy-detection') if reachable, else skip
        if not Path("dr2015/trainLabels.csv").exists():
            raise FileNotFoundError("2015 DR data not present")
        # ... Ben-Graham preprocess + a few regression epochs on 2015 labels, then return ...
        return model, True
    except Exception as e:
        print(f"[warn] 2015 DR pretrain skipped: {e}")
        return model, False     # NEVER crash the round on this
```

## Score Milestones

```text
B3 single regression (no Ben-Graham/TTA) ........... weak, well short
B3 + 2015 DR pretraining ........................... closer, still short
plain single B4/B3 + good OOF thresholds ........... single-model level, still short
B4-380 + Ben-Graham + 5-fold reg + OOF thresholds .. strong single calibrated model
2-3 variant ENSEMBLE (seeds/backbones) + OOF thr ... best, stable <-- target: the ensemble + OOF thresholds is the decisive lever
SE-ResNeXt + rank fusion ........................... REGRESSED — avoid heavy mismatched fusion
```

The decisive lever is the **ENSEMBLE**: after one calibrated B4-380 Ben-Graham regression with mandatory OOF thresholds, add a seed/backbone variant per stage, average continuous preds, and re-optimize thresholds — that is what crosses. Threshold-opt and 2015-pretrain are real but small on their own; under-ensembling is the gap. Treat 2015 pretraining as a robustness-wrapped extra *variant*, not a dependency. Self-check against the `criterion.json` pass line, not a fixed threshold.
