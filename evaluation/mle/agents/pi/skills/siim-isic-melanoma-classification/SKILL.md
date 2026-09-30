Task: SIIM-ISIC Melanoma Classification — binary skin-lesion classification (`target` malignant=1), scored by **ROC-AUC (HIGHER is better)**. The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

## Critical context (read first)

The strong reference submission was a **rank blend of three separately-trained artifacts** (a strong image-CNN "anchor", an external-data EfficientNet-B3, and a B5-embedding GBDT). Historical solve scripts only *blended cached `.pt`/`.csv`/`.npy` from prior runs* under personal/mounted paths — **those caches DO NOT EXIST in a fresh run.** If you only "preserve an anchor and add 18% of an external B3", with nothing actually built, you train a single plain CNN and **plateau clearly short of a competitive score** (this is exactly the observed failure). You must **BUILD every source from scratch.**

What actually crosses into competitive territory is **NOT one CNN.** It is: a **patient-grouped k-fold EfficientNet ensemble at 384px** + **multi-crop TTA** + **a second, externally-augmented backbone (B3 trained on 2020 + ISIC-2019, which lifts the malignant rate from 1.76% → ~17%)** + **rank-averaged blend across folds/backbones**. A single plain CNN cannot get there. Stage the work so each phase writes a valid `submission.csv` and only overwrites the previous best when **patient-grouped OOF AUC improves**.

Data lives ONLY in `prepared/public/`: `train.csv`, `test.csv`, `sample_submission.csv`, `jpeg/train/<image_name>.jpg`, `jpeg/test/<image_name>.jpg`. NEVER read `prepared/private/`, never join test labels. External ISIC-2019 data for *training* is allowed (it is genuinely external and contains no 2020 test labels) — only use it to add malignant training examples, never to look up test ids.

## Approach

1. **Folds — patient-grouped, fixed once and carried forward.**
   `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)`, stratify by `target`, **group by `patient_id`**. Patients have multiple lesions; lesion-level random CV is optimistic by ~0.01 AUC and will mislead blend selection. Save `folds.csv` and reuse it verbatim throughout.

2. **One solid k-fold image CNN, end-to-end, VALID submission.**
   - Backbone `tf_efficientnet_b3_ns` (timm, `pretrained=True`, `num_classes=1`), **image size 384**, GAP head → single logit, BCE-with-logits loss.
   - Resize each jpg with `INTER_AREA` to 384; augment (train only): horizontal+vertical flip, `ShiftScaleRotate`, brightness/contrast, hue/sat, `CoarseDropout` (microscope-style), light `RandomResizedCrop(scale=(0.8,1.0))`. Validation/test: resize only + ImageNet normalize.
   - Optimizer Adam/AdamW `lr=3e-4` (head/backbone same), cosine schedule, **AMP bf16**, batch 32–48, **8–12 epochs/fold**. Class imbalance: BCE `pos_weight≈ (neg/pos)` capped to ~5, OR sampler oversampling positives ~3×. Keep EMA of weights; pick best-epoch by **fold AUC** (save both raw and EMA, keep whichever scores higher on the val fold).
   - For each fold save: checkpoint `b3_384_fold{f}.pt`, OOF preds for its val rows, and test preds (mean of 2 flips = cheap 2× TTA). Build `oof_b3.npy` (length=len(train)) and `test_b3.npy` (mean over 5 folds).
   - **Time:** at 384px / B3, per-fold-epoch cost is high, so 5 folds × 10ep can be tight. To fit a limited budget: either run a subset of folds first and add the rest in a later pass, or **cache 384px resized jpgs once to disk** (`cache_resized/384/...`) so epochs are I/O-cheap and 5×8 fits. Prefer caching. Write a submission from whatever folds finished (`pct_rank(test_b3)`), so this phase is always valid even if cut short.
   - Expected single-backbone OOF AUC is a floor, NOT a competitive score.

3. **Finish folds, add TTA, blend across folds.**
   - Inherit `b3_384_fold*.pt`, `oof_b3.npy`, `test_b3.npy`. Train any missing folds; recompute OOF.
   - **Heavy TTA at inference (this is a real lever):** `tta=8` — original + hflip + vflip + hflip∘vflip, each at center + one scale jitter, **rank-average** the TTA views per image, then **rank-average across the 5 folds**. TTA alone typically adds a small but real AUC gain over single-crop.
   - Re-evaluate patient-grouped OOF AUC; only overwrite `submission.csv` if it beats the carried-forward best. Still short of the strongest models — expected.

4. **Second backbone + EXTERNAL DATA (the decisive crossing lever).**
   - Train `tf_efficientnet_b3_ns` (and/or `b5`) **with ISIC-2019 external malignant images merged into training folds** (concat external rows into the train split of each fold; external rows have no `patient_id` overlap so grouping is safe; keep validation = 2020 public only). External data raises malignant rate from `1.76% → ~17%`, which sharply improves recall calibration and OOF AUC stability. Same 384px, 8 epochs, 8× TTA on the 2020 test set.
     - If ISIC-2019 jpgs are not available in the environment, instead train a **second seed/backbone** (`seed=2027`, or `tf_efficientnet_b4_ns`@384) to get an orthogonal model; the external route is preferred when data exists.
   - Build `oof_ext.npy`, `test_ext.npy`. The external/second model on its own may be slightly weaker per-fold but is **decorrelated** — that is what the blend exploits.

5. **OOF-tuned rank blend + optional metadata fusion; final submission.**
   - Combine all available test-prediction vectors by **rank-normalizing each, then weighted-averaging with weights chosen to maximize patient-grouped OOF AUC** (grid search weights on the stacked OOF vectors, not on grader feedback). The reference final used roughly a dominant strong model plus ~15–25% of a decorrelated external model; reproduce that *spirit*, but **let OOF pick the weights**.
   - **Optional metadata/GBDT branch** (orthogonal, often weight 0.0–0.1): extract GAP embeddings from the trained B3/B5 OOF checkpoints, `StandardScaler+PCA(128)`, concat patient/site/age/jpeg-size meta features, train 5-fold LightGBM+XGBoost with `scale_pos_weight`, rank-average. **Only add it if OOF improves** — in the original its final weight was `0.00`. Do not force-include a new branch.
   - Final: rank-normalize the blended vector, write `submission.csv` in **exact `sample_submission.csv` order**. Keep previous best as fallback; overwrite only if OOF AUC improves.

## What Actually Moved The Metric

From the real trajectory:
- **Single plain CNN / metadata-GBDT alone:** plateaus clearly short. Speculative architecture swaps regressed (B5/B6@512 external attempt, Swin/ViT/BEiT all landed below the strong CNN). **Architecture churn did NOT help.**
- **Strong patient-grouped k-fold EfficientNet ensemble + heavy TTA** is the bulk of the score: this is the anchor a fresh agent must rebuild from scratch — it is NOT free, it is the main work.
- **Adding a decorrelated, externally-trained EfficientNet-B3 (2020 + ISIC-2019) at ~18% rank weight** provided the thin margin that crosses from the anchor floor into competitive territory. The lift comes from *decorrelation + external malignant signal*, not from the external model being individually better.
- **Rank-normalization before AND after blending** matters because CNNs/external-models/GBDTs are differently calibrated; raw-probability blends underperform.

## Key Code Snippets

Patient-grouped folds (build once, carry forward):
```python
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def build_folds(train, n_splits=5, seed=2026):
    y = train["target"].values.astype(int)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(sgkf.split(train, y, groups=train["patient_id"].values))
    fold_id = np.full(len(train), -1, np.int16)
    for f, (_, va) in enumerate(folds): fold_id[va] = f
    pd.DataFrame({"image_name": train["image_name"], "patient_id": train["patient_id"],
                  "target": y, "fold": fold_id}).to_csv("folds.csv", index=False)
    return folds, fold_id
```

Data + augmentation (train) / eval transforms:
```python
import albumentations as A, cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

MEAN, STD = (0.485,0.456,0.406), (0.229,0.224,0.225)
def train_tf(sz=384):
    return A.Compose([
        A.RandomResizedCrop(size=(sz,sz), scale=(0.8,1.0), ratio=(0.9,1.1), p=0.5),
        A.Resize(sz, sz, interpolation=cv2.INTER_AREA),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=20, p=0.5),
        A.RandomBrightnessContrast(0.1,0.1,p=0.5), A.HueSaturationValue(10,15,10,p=0.4),
        A.CoarseDropout(p=0.4),
        A.Normalize(MEAN, STD), ToTensorV2()])
def eval_tf(sz=384):
    return A.Compose([A.Resize(sz,sz,interpolation=cv2.INTER_AREA), A.Normalize(MEAN,STD), ToTensorV2()])

class MelDS(Dataset):
    def __init__(self, df, tf, has_y=True):
        self.p=df["image_path"].values; self.tf=tf
        self.y=df["target"].values.astype("float32") if has_y else None
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        img=cv2.cvtColor(cv2.imread(self.p[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        x=self.tf(image=img)["image"]
        return (x, self.y[i]) if self.y is not None else (x, i)
```

Model + AUC-aware fold training (BCE + pos_weight, EMA, pick best by fold AUC):
```python
import timm, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

def make_model(name="tf_efficientnet_b3_ns"):
    m = timm.create_model(name, pretrained=True, num_classes=1, drop_rate=0.2)
    return m

def train_one_fold(model, tr_loader, va_loader, y_va, epochs=8, lr=3e-4, pos_weight=5.0, dev="cuda"):
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs*len(tr_loader))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=dev))
    best_auc, best_state = -1, None
    for ep in range(epochs):
        model.train()
        for x,y in tr_loader:
            x,y=x.to(dev),y.to(dev).unsqueeze(1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss=lossf(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval(); preds=[]
        with torch.no_grad():
            for x,_ in va_loader:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    preds.append(torch.sigmoid(model(x.to(dev))).float().cpu().numpy())
        auc=roc_auc_score(y_va, np.concatenate(preds).ravel())
        if auc>best_auc: best_auc, best_state = auc, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"ep{ep} auc={auc:.5f}")
    model.load_state_dict(best_state); return model, best_auc
```

Heavy TTA inference (8 views, rank-averaged) — the per-fold test predictor:
```python
import numpy as np
from scipy.stats import rankdata
def pct_rank(x): x=np.asarray(x,float); return rankdata(x,method="average")/(len(x)+1.0)

@torch.no_grad()
def infer_tta(model, loader, n=len(test), dev="cuda"):
    model.eval(); acc=np.zeros(n)
    flips=[(False,False),(True,False),(False,True),(True,True)]  # ×2 scales below = 8 views
    for hf,vf in flips:
        for scale in (1.0, 0.9):
            p=np.zeros(n)
            for x,idx in loader:
                xb=x.to(dev)
                if hf: xb=torch.flip(xb,[3])
                if vf: xb=torch.flip(xb,[2])
                if scale!=1.0:
                    xb=torch.nn.functional.interpolate(xb, scale_factor=scale, mode="bilinear", align_corners=False)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o=torch.sigmoid(model(xb)).float().cpu().numpy().ravel()
                p[idx.numpy()]=o
            acc += pct_rank(p)      # rank-average the views
    return acc/(len(flips)*2)
```

OOF k-fold ensemble + OOF-tuned rank blend (across backbones; choose weight by OOF, not grader):
```python
from sklearn.metrics import roc_auc_score
# oof_*  : length len(train), aligned to train order; test_* : length len(test)
def blend_by_oof(y, oof_list, test_list):
    # grid over the second model's weight; first model is the dominant strong ensemble
    base_oof, base_test = pct_rank(oof_list[0]), pct_rank(test_list[0])
    best = (1.0, base_oof, base_test, roc_auc_score(y, base_oof))
    for o2, t2 in zip(oof_list[1:], test_list[1:]):
        ro, rt = pct_rank(o2), pct_rank(t2)
        for w in np.linspace(0.05, 0.40, 36):
            cand_oof = (1-w)*best[1] + w*ro
            auc = roc_auc_score(y, cand_oof)
            if auc > best[3]:
                best = (w, cand_oof, (1-w)*best[2]+w*rt, auc)
    return best  # (w, blended_oof, blended_test, oof_auc)
```

Final submission in sample order (overwrite only if OOF improved):
```python
sample = pd.read_csv(DATA/"sample_submission.csv")
out = sample[["image_name"]].copy()
out["target"] = pct_rank(blended_test)           # final rank-normalize
assert out["image_name"].equals(sample["image_name"])
assert out["target"].isna().sum()==0 and out["target"].nunique()>1
out.to_csv("submission.csv", index=False)        # only after confirming OOF AUC >= previous best
```

## Score Milestones (relative — what each step buys)

- Naive CNN / metadata-only: around the competition median line.
- Single B3@384 k-fold ensemble (light TTA): the initial floor.
- + heavy 8× TTA + all 5 folds rank-averaged: a small step up (still short of the strongest models — expected).
- + second/external EfficientNet-B3 (2020 + ISIC-2019), OOF-tuned ~15–20% rank blend: the crossing into competitive territory, the decisive step.
- Pushing further (more backbones/seeds and 512px) is possible but expensive and beyond a tight budget.

**Do NOT** chase architecture swaps (Swin/ViT/B6@512) — every such attempt in the real trajectory regressed below the strong CNN. The reproducible path is: strong patient-grouped EfficientNet k-fold + heavy TTA + one decorrelated external-data backbone + OOF-tuned rank blend.

The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score. What stays invariant is the ORDERING: the patient-grouped k-fold EfficientNet ensemble + heavy TTA is the bulk of the score and must be built from scratch; a single decorrelated external-data backbone at a modest rank weight is the decisive crossing lever; and rank-normalization before and after blending is what makes the blend valid. Architecture churn never crosses it.
