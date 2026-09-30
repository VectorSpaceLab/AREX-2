Denoising Dirty Documents is a grayscale image-to-image regression task: predict cleaned pixel values in `[0, 1]`, evaluated by RMSE (lower is better). The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

## Approach

Use a small supervised U-Net trained on random image patches, with a classical document-cleaning baseline as an extra input channel. The reliable configuration was not a pure CV filter and not a large model; it was a compact U-Net learning residual/document cleanup behavior from paired dirty and clean pages.

1. Load train images from `train/*.png`, clean targets from `train_cleaned/*.png`, and test images from `test/*.png`. Read all images as grayscale float32 in `[0, 1]`.

2. Build a classical baseline per image:
   - Convert the dirty image to uint8.
   - Estimate local background with `cv2.medianBlur(..., ksize=15)`.
   - Divide the dirty image by this background.
   - Clip to `[0, 1]`.

3. Use two model input channels:
   - channel 0: raw dirty grayscale image
   - channel 1: median-background-divided CV baseline

4. Use one holdout split to tune training and post-processing:
   - Collect all train ids.
   - Shuffle with seed `2026`.
   - Use the first `18` ids as validation.
   - Use the rest as training.
   - After tuning, retrain on all train images for the final submission.

5. Train on random `128x128` patches:
   - Oversample ink-containing patches about `35%` of the time using target pixels `< 0.92`.
   - Horizontal flip augmentation with probability `0.5`.
   - Dataset length per epoch: `4096` random patches.
   - Batch size: `96`.
   - Optimizer: `AdamW(lr=2e-3, weight_decay=1e-4)`.
   - Loss: pixelwise MSE.
   - AMP enabled on CUDA.
   - Validation training: `40` epochs with early stopping after epoch `12` if no improvement for `7` epochs.
   - Final model: train on all public training images for `45` epochs.

6. Predict full test images at native resolution with reflective padding inside the model so dimensions divisible by `8` work cleanly.

7. Tune a simple post-process on the validation split:
   - Blend model output with the CV baseline:
     `z = (1 - alpha) * model_pred + alpha * cv_baseline`
   - Grid search `alpha` over `0.0, 0.05, ..., 0.5`.
   - Grid search white threshold over `[0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.01]`.
   - If `white_thr <= 1.0`, force pixels above it to exactly `1.0`.
   - Reuse the best validation `alpha` and `white_thr` for final test submission.

8. Write submission exactly in `sampleSubmission.csv` order. The sample id format is `image_row_col`, with row and column 1-indexed. Convert to zero-indexed indices when extracting predicted values.

## What Actually Moved The Metric

The biggest lift came from turning the classical denoising result into a learned input feature instead of relying on it alone. The `medianBlur(ksize=15)` background division gives a strong document-cleaning prior, but the U-Net learns where that prior damages ink strokes, over-whitens texture, or leaves residual dirt. The winning model used both raw dirty pixels and the CV baseline as channels; dropping either channel is likely worse.

The second important change was patch sampling that does not drown the loss in blank background. Dirty document images are mostly near-white page area, so uniform random patches make the model optimize background RMSE while under-learning text strokes. Oversampling target pixels where `clean < 0.92` for `35%` of patches helped keep ink reconstruction important.

The third meaningful gain was validation-tuned post-processing. A small blend back toward the CV baseline plus optional hard-whitening of very bright pixels reduces background noise without needing a larger ensemble. This is cheap and should always be tuned on the holdout split before final all-data training.

Pitfalls and lessons:

- A pure thresholding or morphology solution is useful as a sanity baseline but is not enough on its own. Use it as an input feature and post-process component, not as the final method.
- Do not train only on full images or uniformly sampled patches without ink oversampling; the model can get deceptively low train loss by predicting clean white background while missing text/detail.
- Do not tune post-processing on the final test distribution or by eyeballing. The validation grid search for `alpha` and `white_thr` is simple and less fragile.
- The most damaging likely mistake is breaking submission indexing. `sampleSubmission.csv` uses `image_row_col` with 1-indexed pixel coordinates. Use `row - 1` and `col - 1`, preserve sample row order exactly, and assert all values are finite and within `[0, 1]`.

## Key Code Snippets

### Reproducibility And Image Loading

```python
from pathlib import Path
import random
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path("./input")

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0
```

### Classical Baseline Channel

This baseline was used both as the second model input channel and as a post-processing blend target.

```python
def cv_baseline(img: np.ndarray, ksize: int = 15) -> np.ndarray:
    x8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    bg = cv2.medianBlur(x8, ksize).astype(np.float32) / 255.0
    return np.clip(img / (bg + 1e-3), 0.0, 1.0).astype(np.float32)

def load_pairs(ids):
    dirty, clean, base = {}, {}, {}
    for image_id in ids:
        x = read_gray(DATA_DIR / "train" / f"{image_id}.png")
        y = read_gray(DATA_DIR / "train_cleaned" / f"{image_id}.png")
        dirty[image_id] = x
        clean[image_id] = y
        base[image_id] = cv_baseline(x)
    return dirty, clean, base
```

### Holdout Split

Use this split to tune the model and post-processing. Then retrain on all ids.

```python
seed = 2026
ids = sorted(int(p.stem) for p in (DATA_DIR / "train").glob("*.png"))

random.Random(seed).shuffle(ids)
val_ids = sorted(ids[:18])
train_ids = sorted(ids[18:])
```

### Patch Dataset With Ink Oversampling

The input tensor has shape `[2, 128, 128]`: raw dirty image plus CV baseline. The target has shape `[1, 128, 128]`.

```python
class PatchDataset(Dataset):
    def __init__(self, ids, dirty, clean, base, patch=128, length=4096, augment=True):
        self.ids = ids
        self.dirty = dirty
        self.clean = clean
        self.base = base
        self.patch = patch
        self.length = length
        self.augment = augment

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image_id = self.ids[random.randrange(len(self.ids))]
        x = self.dirty[image_id]
        y = self.clean[image_id]
        b = self.base[image_id]
        h, w = x.shape

        if random.random() < 0.35:
            ink = np.argwhere(y < 0.92)
            if len(ink):
                cy, cx = ink[random.randrange(len(ink))]
                r0 = int(np.clip(cy - self.patch // 2, 0, max(0, h - self.patch)))
                c0 = int(np.clip(cx - self.patch // 2, 0, max(0, w - self.patch)))
            else:
                r0 = random.randrange(max(1, h - self.patch + 1))
                c0 = random.randrange(max(1, w - self.patch + 1))
        else:
            r0 = random.randrange(max(1, h - self.patch + 1))
            c0 = random.randrange(max(1, w - self.patch + 1))

        xp = x[r0:r0 + self.patch, c0:c0 + self.patch]
        bp = b[r0:r0 + self.patch, c0:c0 + self.patch]
        yp = y[r0:r0 + self.patch, c0:c0 + self.patch]

        if self.augment and random.random() < 0.5:
            xp = np.ascontiguousarray(xp[:, ::-1])
            bp = np.ascontiguousarray(bp[:, ::-1])
            yp = np.ascontiguousarray(yp[:, ::-1])

        inp = np.stack([xp, bp], axis=0).astype(np.float32)
        tgt = yp[None].astype(np.float32)
        return torch.from_numpy(inp), torch.from_numpy(tgt)
```

### Tiny U-Net

The model is intentionally small. Width `32` was sufficient.

```python
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class TinyUNet(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.c1 = ConvBlock(2, width)
        self.c2 = ConvBlock(width, width * 2)
        self.c3 = ConvBlock(width * 2, width * 4)
        self.c4 = ConvBlock(width * 4, width * 8)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(width * 8, width * 4, 2, stride=2)
        self.d3 = ConvBlock(width * 8, width * 4)
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.d2 = ConvBlock(width * 4, width * 2)
        self.up1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.d1 = ConvBlock(width * 2, width)
        self.out = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        orig_h, orig_w = x.shape[-2:]
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        e1 = self.c1(x)
        e2 = self.c2(self.pool(e1))
        e3 = self.c3(self.pool(e2))
        z = self.c4(self.pool(e3))

        z = self.up3(z)
        z = self.d3(torch.cat([z, e3], dim=1))
        z = self.up2(z)
        z = self.d2(torch.cat([z, e2], dim=1))
        z = self.up1(z)
        z = self.d1(torch.cat([z, e1], dim=1))

        z = torch.sigmoid(self.out(z))
        return z[..., :orig_h, :orig_w]
```

### Training Loop Hyperparameters

Use MSE loss because the metric is RMSE; minimizing MSE is equivalent.

```python
def train_model(train_ids, val_ids, dirty, clean, base, device, epochs, seed):
    seed_everything(seed)

    model = TinyUNet(width=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    ds = PatchDataset(train_ids, dirty, clean, base, patch=128, length=4096, augment=True)
    dl = DataLoader(
        ds,
        batch_size=96,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = None
    best_val = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(xb)
                loss = loss_fn(pred, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        if val_ids:
            val_rmse, _ = evaluate(model, val_ids, dirty, clean, device)
            if best_val is None or val_rmse < best_val:
                best_val = val_rmse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch >= 12 and no_improve >= 7:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val
```

### Full-Image Prediction And Validation RMSE

```python
@torch.no_grad()
def predict_image(model, img, device):
    model.eval()
    b = cv_baseline(img)
    inp = np.stack([img, b], axis=0)[None].astype(np.float32)
    x = torch.from_numpy(inp).to(device)
    pred = model(x).squeeze().float().cpu().numpy()
    return np.clip(pred, 0.0, 1.0).astype(np.float32)

@torch.no_grad()
def evaluate(model, ids, dirty, clean, device):
    preds = []
    se_sum = 0.0
    n_sum = 0

    for image_id in ids:
        p = predict_image(model, dirty[image_id], device)
        y = clean[image_id]
        preds.append(p)
        se_sum += float(np.sum((p - y) ** 2))
        n_sum += y.size

    return float(np.sqrt(se_sum / n_sum)), preds
```

### Validation-Tuned Post-Processing

This small grid search is worth keeping. It is cheap and directly optimizes RMSE on the same pixel scale as the metric.

```python
def tune_postprocess(preds, bases, targets):
    best = (10.0, 1.0, 1.01)

    for alpha in np.linspace(0.0, 0.5, 11):
        for white_thr in [0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.01]:
            se = 0.0
            n = 0

            for p, b, y in zip(preds, bases, targets):
                z = (1.0 - alpha) * p + alpha * b

                if white_thr <= 1.0:
                    z = z.copy()
                    z[z > white_thr] = 1.0

                z = np.clip(z, 0.0, 1.0)
                se += float(np.sum((z - y) ** 2))
                n += y.size

            score = float(np.sqrt(se / n))
            if score < best[0]:
                best = (score, float(alpha), float(white_thr))

    return best
```

### Final Training Pattern

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ids = sorted(int(p.stem) for p in (DATA_DIR / "train").glob("*.png"))
dirty, clean, base = load_pairs(ids)

random.Random(2026).shuffle(ids)
val_ids = sorted(ids[:18])
train_ids = sorted(ids[18:])

model, best_val = train_model(
    train_ids=train_ids,
    val_ids=val_ids,
    dirty=dirty,
    clean=clean,
    base=base,
    device=device,
    epochs=40,
    seed=2026,
)

val_rmse, val_preds = evaluate(model, val_ids, dirty, clean, device)
val_bases = [base[i] for i in val_ids]
val_targets = [clean[i] for i in val_ids]
tuned_rmse, alpha, white_thr = tune_postprocess(val_preds, val_bases, val_targets)

final_model, _ = train_model(
    train_ids=sorted(ids),
    val_ids=None,
    dirty=dirty,
    clean=clean,
    base=base,
    device=device,
    epochs=45,
    seed=2027,
)
```

### Submission Writing

Keep the `sampleSubmission.csv` order. Parse ids into image, row, and column, converting row and column to zero-indexed coordinates.

```python
def write_submission(model, alpha, white_thr, device, out_path):
    sample = pd.read_csv(DATA_DIR / "sampleSubmission.csv")

    id_parts = sample["id"].str.split("_", expand=True).astype(int)
    sample["_image"] = id_parts[0]
    sample["_row"] = id_parts[1] - 1
    sample["_col"] = id_parts[2] - 1

    pred_cache = {}

    for image_id in sorted(sample["_image"].unique()):
        img = read_gray(DATA_DIR / "test" / f"{image_id}.png")
        pred = predict_image(model, img, device)
        base = cv_baseline(img)

        z = (1.0 - alpha) * pred + alpha * base
        if white_thr <= 1.0:
            z[z > white_thr] = 1.0

        pred_cache[int(image_id)] = np.clip(z, 0.0, 1.0).astype(np.float32)

    values = np.empty(len(sample), dtype=np.float32)

    for image_id, idx in sample.groupby("_image", sort=False).groups.items():
        rows = sample.loc[idx, "_row"].to_numpy()
        cols = sample.loc[idx, "_col"].to_numpy()
        values[idx] = pred_cache[int(image_id)][rows, cols]

    out = pd.DataFrame({"id": sample["id"].to_numpy(), "value": values})
    out.to_csv(out_path, index=False)

    check = pd.read_csv(out_path, usecols=["id", "value"])
    assert check.shape == (len(sample), 2)
    assert check.columns.tolist() == ["id", "value"]
    assert check["id"].equals(sample["id"])
    assert np.isfinite(check["value"]).all()
    assert check["value"].between(0, 1).all()
```

## Score Milestones (relative — what each step buys)

Naive dirty-image or weak CV-only baselines should be treated as sanity checks; if RMSE is no better than a plain classical filter, the method is underpowered. The largest single jump comes from feeding the CV baseline in as a second channel to the U-Net rather than using it alone. Ink-biased patch sampling and validation-tuned post-processing (CV blend + hard-whitening) each add smaller but reliable gains on top. What stays invariant is the ordering: two-channel Tiny U-Net > single-channel U-Net > classical filter alone, and ink oversampling and tuned post-processing further tighten RMSE. Self-check against the `criterion.json` pass line rather than any fixed threshold.
