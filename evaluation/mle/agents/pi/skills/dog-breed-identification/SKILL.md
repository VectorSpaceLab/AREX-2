Task: Dog Breed Identification — 120-class fine-grained classification scored by multiclass log loss (LOWER is better). The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

READ THIS FIRST — the lever is a CORRECTLY-fine-tuned strong backbone:
Dog breeds are a subset of ImageNet / EVA02 pretraining, so a strong pretrained backbone that is fine-tuned correctly gets near-perfect on this task HONESTLY. The original solve wasted 60+ attempts because every learned attempt was EXECUTION-BUGGED, not because classification is the wrong lever:
- Frozen-feature + LogReg / linear head plateaued at log loss `0.15`–`0.21` (a frozen backbone is not adapted to the fine-grained boundaries).
- Direct ImageNet-1000-dog-logit mapping capped at `~0.1479` (Kaggle's 120 breeds do NOT 1:1 align with ImageNet dog synsets).
- End-to-end fine-tune "collapsed at epoch 1" over and over: unfreezing the WHOLE backbone at once + a randomly-init head with too-high head LR (`2e-3`) + mixup/cutmix destroys the pretrained features, val logloss explodes (`4.37→3.51→2.49`...), and the agent reverted to the `0.1479` prior.

The breakthrough and the TEMPLATE for this skill: an honest **EVA02-Large progressive fine-tune** using the timm fine-grained recipe reached a very low log loss. The fix that unlocked it: proper **WARMUP** + **LOWER head LR** (`5e-4`, not `2e-3`) + **fixing the epoch-1 collapse** via two-stage linear-probe-then-fine-tune with **layer-wise LR decay / progressive unfreeze** (never unfreeze everything under a hot random head).

FORBIDDEN (the leak path that later cheated — DO NOT USE):
- NO perceptual-hash / pHash / dHash matching of test images to anything.
- NO downloading Stanford Dogs `images.tar` (or any external dog image set) to look up / recover test labels.
- NO nearest-neighbor / duplicate image matching to assign test breeds.
- Read `prepared/public/` ONLY; NEVER read `prepared/private/`; NEVER join test labels.
- Using timm / open_clip ImageNet-pretrained WEIGHTS is FINE and is the whole point — that is not leakage. Recovering test labels by matching images to an external labeled set IS leakage and is banned.

## Approach

ALWAYS write a valid `submission.csv` and keep the previous best as `best_submission.csv` fallback so a run can never regress.

Data (read `prepared/public/` ONLY): `sample_submission.csv` (column order = the 120 class names AND the exact test id order — submission MUST match both), `labels.csv` (train id→breed), `train/<id>.jpg`, `test/<id>.jpg`.

### Primary model: EVA02-Large two-stage anti-collapse fine-tune

Model: `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` @ 448 (alt: `eva02_large_patch14_clip_336` @ 336). Build via timm with `num_classes=120`, `pretrained=True`. Use the model's own pretrained mean/std and input size.

**Stage A — linear probe (mandatory, makes the head sane BEFORE any unfreeze):**
1. FREEZE the entire backbone; train ONLY the new 120-class head.
2. Head LR `~1e-3`, AdamW, cosine, **2-3 epochs**, label smoothing `0.1`.
3. This gives the head a reasonable init so Stage B's gradients into the backbone are small and well-aligned. Skipping this is the #1 cause of the epoch-1 collapse.

**Stage B — full fine-tune with LAYER-WISE LR DECAY (the anti-collapse core):**
4. Unfreeze the backbone, but with **layer-wise LR decay** `layer_decay ≈ 0.75` (deepest blocks near head get the full LR; early blocks get LR * 0.75^depth — early layers barely move). Equivalent acceptable variant: **progressive unfreeze** (unfreeze last block group first, add more block groups over epochs). NEVER unfreeze all layers at once under a hot head.
5. Backbone base LR `~1e-5`; head LR `~5e-4` (NOT `2e-3` — that is the documented collapse trigger).
6. **WARMUP**: 1 epoch linear warmup (or `warmup_ratio 0.1`) ramping from ~0 to peak LR, then cosine decay to ~0.
7. AdamW (`weight_decay ~0.05`), label smoothing `0.1`, **light or NO mixup/cutmix** (mixup on top of unfreezing wrecks pretrained features — keep it off or very mild, mixup_alpha ≤ 0.1).
8. EMA of weights (decay `~0.9998`); evaluate and predict with the EMA weights.
9. bf16 autocast; gradient clipping (`max_norm 1.0`).
10. **5-8 epochs** Stage B. Augmentations: RandomResizedCrop (scale 0.5-1.0), hflip, light color jitter / RandAugment(m=7). Do NOT over-augment a fine-grained task.

**Cross-validation:** 5-fold stratified by breed (or at least one clean holdout fold + a full-data refit). Save per-fold OOF probabilities (`oof_eva.npy`, shape `(n_train, 120)`) and per-fold test probabilities; average test probs across folds. Save the fine-tuned weights (`eva_fold{k}.pt`).

### Ensemble for the extra push

Add a second strong backbone from a DIFFERENT family, fine-tuned with the SAME two-stage anti-collapse recipe:
- `convnext_large_mlp.clip_laion2b_ft_in1k_384` @ 384, OR `swin_large_patch4_window12_384` @ 384.
- Save its OOF (`oof_cnx.npy`) and test probs.
- Blend OOF-weighted: search a blend weight `w∈[0,1]` minimizing OOF log loss of `w*eva + (1-w)*cnx`; apply that weight to test probs. Family diversity is what pushes a single strong model to the best result.

### TTA (inference)

For each test image average probabilities over: original + horizontal flip, and optionally a 5-crop (center + 4 corners) or a second slightly larger resize. Average the softmax probabilities (not logits). Do TTA for every backbone before blending.

### Calibration + log-loss hygiene

1. **Temperature scaling** on OOF: fit a single scalar `T` minimizing OOF log loss over `softmax(logit_oof / T)`; apply `T` to test logits. (If you only kept probs, fit `T` on `log(probs)`.) Typically `T` slightly > 1 reduces overconfidence and lowers log loss.
2. **Clip** probabilities to `[1e-15, 1-1e-15]` (a small floor — never exact 0), then **renormalize** each row to sum to 1.
3. Write submission in EXACT `sample_submission.csv` column order (120 breed columns) and id order.

## What Actually Moved The Metric

From the real trajectory (log loss, lower better — relative ordering of what each lever buys):
- Frozen bottleneck features + LogReg / dense heads (and ~15 later "concat" retries): stuck `0.15`–`0.21`. A frozen backbone never adapts to fine-grained breed boundaries.
- ImageNet-1000-dog-logit direct mapping: clean ceiling `~0.1479` — because Kaggle's 120 breeds don't 1:1 align with ImageNet dog synsets. Useful conceptually as a "prior" but it is NOT a route to the pass line.
- End-to-end fine-tune attempts (ConvNeXt / EVA02 / Swin / CLIP / ArcFace) `0.148`–`0.33` (CLIP branch `0.58`): these REPEATEDLY "collapsed at epoch 1" and the agent reverted to the `0.1479` prior. **This was an execution bug, not a ceiling** — the cause was unfreezing the whole backbone at once + hot random head (head LR `2e-3`) + mixup/cutmix nuking pretrained features.
- **EVA02-Large progressive fine-tune with the timm fine-grained recipe: the decisive result, a near-perfect log loss.** The decisive fixes vs the failed attempts: WARMUP added, head LR LOWERED from `2e-3` to `5e-4`, and the epoch-1 collapse fixed via two-stage linear-probe-then-LLRD/progressive-unfreeze instead of unfreezing everything under a hot head.

Decisive change: the SAME classification lever the prior attempts kept mis-executing, finally executed correctly (warmup + low head LR + staged unfreeze + LLRD). Dog breeds ⊂ pretraining, so a correctly fine-tuned strong backbone gets near-perfect honestly.

## Pitfalls

- **Epoch-1 collapse (the historical killer):** unfreezing the entire backbone at once, under a randomly-initialized head with a high head LR (`2e-3`) and mixup/cutmix, sends huge gradients through the pretrained backbone → val logloss explodes (`4.37→3.51→2.49`...) and never recovers. FIX (mandatory): Stage A linear probe first, then Stage B with WARMUP + head LR `5e-4` + LLRD `0.75` (or progressive unfreeze) + light/no mixup + grad clip.
- **Reverting to the prior:** if a fold's val logloss is exploding at epoch 1, the unfreeze schedule is WRONG — fix the schedule. Do NOT revert to the `0.1479` ImageNet-logit-mapping prior and call it done (that wasted many attempts). The fine-tune OOF MUST end up clearly `< 0.10`.
- **Frozen-feature plateau:** freezing the backbone and training only a linear/LogReg head caps at `0.15`–`0.21`. The backbone MUST be fine-tuned (Stage B) to cross the pass line.
- **Logit-mapping ceiling:** mapping ImageNet-1000 dog logits to the 120 classes caps at `~0.1479` because the breed sets don't align. Not a viable route.
- **Over-augmentation:** heavy mixup/cutmix/RandAugment on ~85 images/class fine-grained data hurts. Keep augmentation light.
- **Overconfidence in log loss:** without temperature scaling + clipping, a confident-but-occasionally-wrong model is punished hard. Always calibrate on OOF and clip to `[1e-15, 1-1e-15]`.
- **Submission alignment:** wrong column order or id order silently tanks the score. Assert both match `sample_submission.csv` exactly.

## Self-Check Gates

- After Stage B for fold 0, assert OOF (or holdout) log loss is clearly `< 0.10` before accepting the run. If `>= 0.10` or exploding, the unfreeze schedule is broken — fix warmup / head LR / staging; do NOT fall back to a logit-mapping prior.
- Keep the previous best submission as `best_submission.csv`; only overwrite when the new full-pipeline OOF log loss IMPROVES.
- First pass = EVA02-L Stage A→B on ONE fold + valid TTA'd, calibrated submission (should already be well under `0.05`). Later passes add the remaining folds, the 2nd backbone, ensemble blend, and refine calibration, inheriting prior fold weights/probs.

## Key Code Snippets

Self-contained. `DATA = Path(prepared/public)`. All training uses public train labels only.

### Setup, folds, datasets
```python
import numpy as np, pandas as pd, timm, torch
import torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

DATA = Path("prepared/public")
sample  = pd.read_csv(DATA / "sample_submission.csv")
classes = list(sample.columns[1:])                  # 120 breed columns, exact order
cls2idx = {c: i for i, c in enumerate(classes)}
labels  = pd.read_csv(DATA / "labels.csv")
labels["y"] = labels["breed"].map(cls2idx)
y = labels["y"].to_numpy()
test_ids = sample["id"].tolist()

MODEL = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"  # img_size 448
cfg = timm.data.resolve_data_config({}, model=timm.create_model(MODEL, pretrained=False))
MEAN, STD, SIZE = cfg["mean"], cfg["std"], cfg["input_size"][-1]

import torchvision.transforms as T
train_tf = T.Compose([
    T.RandomResizedCrop(SIZE, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
    T.RandAugment(num_ops=2, magnitude=7),
    T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.Resize(int(SIZE*1.14)), T.CenterCrop(SIZE),
    T.ToTensor(), T.Normalize(MEAN, STD)])

class DogDS(Dataset):
    def __init__(self, ids, ys, split, tf):
        self.ids, self.ys, self.root, self.tf = ids, ys, DATA/split, tf
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        im = Image.open(self.root/f"{self.ids[i]}.jpg").convert("RGB")
        x = self.tf(im)
        return (x, self.ys[i]) if self.ys is not None else x

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(labels["id"], y))
```

### Layer-wise LR decay param groups (the anti-collapse core)
```python
def llrd_param_groups(model, base_lr, head_lr, layer_decay=0.75, wd=0.05):
    # EVA02/ViT: blocks live in model.blocks; head in model.head. Deeper -> higher LR.
    blocks = model.blocks
    n = len(blocks)
    groups, seen = [], set()
    # head + final norm: head_lr
    head_params = [p for nm,p in model.named_parameters()
                   if p.requires_grad and (nm.startswith("head") or nm.startswith("fc_norm") or nm.startswith("norm"))]
    groups.append({"params": head_params, "lr": head_lr, "weight_decay": wd})
    seen.update(id(p) for p in head_params)
    # blocks: LR scaled by layer_decay**(depth from top)
    for i, blk in enumerate(blocks):
        scale = layer_decay ** (n - i)              # early layers -> tiny LR
        ps = [p for p in blk.parameters() if p.requires_grad and id(p) not in seen]
        if ps:
            groups.append({"params": ps, "lr": base_lr * scale, "weight_decay": wd})
            seen.update(id(p) for p in ps)
    # patch embed / cls token / pos embed: smallest LR
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    if rest: groups.append({"params": rest, "lr": base_lr*(layer_decay**(n+1)), "weight_decay": wd})
    return groups
```

### Two-stage fine-tune loop (Stage A linear probe -> Stage B LLRD + warmup + EMA)
```python
from torch.optim.lr_scheduler import LambdaLR
import math, copy

def cosine_warmup(opt, total_steps, warmup_steps):
    def f(s):
        if s < warmup_steps: return s / max(1, warmup_steps)
        p = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * p))
    return LambdaLR(opt, f)

def run_epoch(model, loader, opt, sched, scaler, ema, smooth=0.1):
    model.train()
    for x, yb in loader:
        x, yb = x.cuda(non_blocking=True), yb.cuda(non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x), yb, label_smoothing=smooth)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if sched: sched.step()
        if ema: ema.update(model)

class EMA:
    def __init__(self, model, decay=0.9998):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1-self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()): s.copy_(p)

def fit_fold(tr_idx, va_idx, fold):
    ids = labels["id"].to_numpy()
    tr = DataLoader(DogDS(ids[tr_idx], y[tr_idx], "train", train_tf),
                    batch_size=16, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    va = DataLoader(DogDS(ids[va_idx], y[va_idx], "train", eval_tf),
                    batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    model = timm.create_model(MODEL, pretrained=True, num_classes=120).cuda()
    scaler = torch.cuda.amp.GradScaler()

    # ---- Stage A: linear probe (backbone frozen, head only) ----
    for nm, p in model.named_parameters():
        p.requires_grad_(nm.startswith("head"))
    optA = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.05)
    for _ in range(3):
        run_epoch(model, tr, optA, None, scaler, None)

    # ---- Stage B: full FT, LLRD + warmup + cosine + EMA ----
    for p in model.parameters(): p.requires_grad_(True)
    groups = llrd_param_groups(model, base_lr=1e-5, head_lr=5e-4, layer_decay=0.75, wd=0.05)
    optB = torch.optim.AdamW(groups)
    EPOCHS_B = 6
    steps = EPOCHS_B * len(tr); warm = len(tr)         # 1 epoch warmup
    sched = cosine_warmup(optB, steps, warm)
    ema = EMA(model, 0.9998)
    best = 9.9
    for e in range(EPOCHS_B):
        run_epoch(model, tr, optB, sched, scaler, ema)
        vp = predict(ema.shadow, va)                   # eval with EMA weights
        ll = log_loss(y[va_idx], vp, labels=list(range(120)))
        print(f"fold{fold} ep{e} val_logloss={ll:.4f}", flush=True)
        # SELF-CHECK GATE: epoch-1 collapse detector
        if e == 0:
            assert ll < 1.0, f"epoch-1 collapse (ll={ll:.3f}) -> fix warmup/head_lr/staging"
        if ll < best: best = ll; torch.save(ema.shadow.state_dict(), f"eva_fold{fold}.pt")
    assert best < 0.10, f"fold{fold} OOF {best:.3f} >= 0.10 -> schedule wrong, do NOT revert to prior"
    return best

@torch.no_grad()
def predict(model, loader):
    model.eval(); out = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(F.softmax(model(x.cuda()).float(), 1).cpu().numpy())
    return np.concatenate(out)
```

### TTA inference (orig + hflip, average probabilities)
```python
@torch.no_grad()
def predict_tta(model, loader):
    model.eval(); out = []
    for batch in loader:
        x = (batch[0] if isinstance(batch,(list,tuple)) else batch).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p  = F.softmax(model(x).float(), 1)
            p += F.softmax(model(torch.flip(x, dims=[3])).float(), 1)
        out.append((p/2).cpu().numpy())
    return np.concatenate(out)
```

### Temperature scaling on OOF + log-loss-safe submission
```python
def fit_temperature(oof_probs, y_true):
    logit = np.log(np.clip(oof_probs, 1e-15, 1.0))
    t = torch.nn.Parameter(torch.ones(1))
    L = torch.tensor(logit, dtype=torch.float32); yy = torch.tensor(y_true)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=60)
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(L / t, yy); loss.backward(); return loss
    opt.step(closure); return float(t.detach())

def write_submission(test_probs, T=1.0, path="submission.csv"):
    p = np.exp(np.log(np.clip(test_probs, 1e-15, 1.0)) / T)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    p /= p.sum(1, keepdims=True)
    sub = pd.DataFrame(p, columns=classes); sub.insert(0, "id", test_ids)
    assert list(sub.columns) == list(sample.columns)
    assert sub["id"].tolist() == sample["id"].tolist()
    assert np.isfinite(p).all() and (p > 0).all()
    sub.to_csv(path, index=False); return sub
```

### OOF-weighted ensemble blend (the extra push)
```python
def best_blend(oof_a, oof_b, y_true):
    best_w, best_ll = 0.5, 9.9
    for w in np.linspace(0, 1, 21):
        ll = log_loss(y_true, np.clip(w*oof_a + (1-w)*oof_b, 1e-15, 1), labels=list(range(120)))
        if ll < best_ll: best_ll, best_w = ll, w
    return best_w, best_ll        # apply best_w to test probs: w*test_a + (1-w)*test_b
```

## Score Milestones
- Frozen bottleneck + linear/dense head (buggy plateau): `~0.15-0.21`
- ImageNet dog-logit mapping ceiling: `~0.1479`
- End-to-end FT with epoch-1 collapse (reverted to prior): `~0.148-0.33` (CLIP branch `0.58`) — the EXECUTION BUG, not a real ceiling
- EVA02-L correct two-stage FT (warmup + head LR `5e-4` + LLRD/progressive unfreeze): a near-perfect log loss — the decisive result

A single well-fine-tuned EVA02-L already gets very low log loss honestly. A calibrated 2-family ensemble pushes lower still but approaches near-perfect and is risky — never overwrite the fallback unless OOF improves. Self-check against the `criterion.json` pass line, not a fixed threshold.
