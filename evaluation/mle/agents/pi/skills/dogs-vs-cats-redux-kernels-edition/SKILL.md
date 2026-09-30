Dogs-vs-cats binary image classification scored by log loss (lower is better). The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

**Approach**

Use transfer learning with a pretrained `timm` CNN. A single stratified holdout (not k-fold CV) is sufficient here; a strong single EfficientNet-B0 already lands far below the naive baseline log loss.

1. Prepare labels from filenames:
   - `dog.*.jpg` -> `1`
   - `cat.*.jpg` -> `0`
   - Test ids are numeric stems: `1.jpg`, `2.jpg`, etc.
   - Build the final submission by merging predictions back to `sample_submission.csv` on `id`.

2. Split training data once:
   - `train_test_split(..., test_size=0.12, stratify=labels, random_state=473685)`
   - Use this validation split to select the best epoch by clipped validation log loss.

3. Model:
   - `timm.create_model("efficientnet_b0.ra_in1k", pretrained=True, num_classes=1)`
   - Single sigmoid logit for dog probability.
   - Loss: `torch.nn.BCEWithLogitsLoss()`

4. Training hyperparameters:
   - `epochs=3`
   - `image_size=224`
   - `batch_size=128` on A100-class GPU
   - validation/test batch size: `batch_size * 2`
   - optimizer: `AdamW(lr=2e-4, weight_decay=1e-4)`
   - scheduler: per-batch `CosineAnnealingLR`, `T_max=epochs * len(train_loader)`
   - AMP enabled on CUDA with `GradScaler`
   - `num_workers=6`, `pin_memory=True`, `persistent_workers=True`
   - seed: `473685`
   - probability clip for validation/submission: `clip=0.01`

5. Augmentation and preprocessing:
   - Train:
     - `RandomResizedCrop(224, scale=(0.78, 1.0), ratio=(0.8, 1.25))`
     - `RandomHorizontalFlip()`
     - mild `ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02)`
     - ImageNet normalization
   - Validation/test:
     - `Resize(int(image_size * 1.14))`
     - `CenterCrop(image_size)`
     - ImageNet normalization

6. Inference:
   - Load the best validation checkpoint.
   - Predict test images in the exact `sample_submission.csv` id order.
   - Use horizontal flip TTA:
     - `p = 0.5 * (sigmoid(model(x)) + sigmoid(model(flip(x))))`
   - Clip final probabilities to `[0.01, 0.99]`.

**What Actually Moved The Metric**

The strong trajectory was short: the first serious transfer-learning run already reached a very low log loss. The high-value decisions were:

1. Pretrained CNN transfer learning was the main jump.
   - This task is visually simple but log-loss sensitive. A pretrained ImageNet EfficientNet-B0 is strong enough with only 3 epochs.
   - Do not start with handcrafted features, shallow ML, or a CNN trained from scratch. Those waste time and are very unlikely to approach a competitive score.

2. The validation and loss setup matched the scoring metric.
   - Use `BCEWithLogitsLoss` during training and compute validation `log_loss` on sigmoid probabilities.
   - Save the best epoch by validation log loss, not by accuracy.
   - Accuracy can look saturated while log loss still improves or regresses.

3. Conservative probability handling mattered.
   - Horizontal flip TTA improved stability at almost no implementation cost.
   - Clipping predictions to `0.01..0.99` protects log loss from a few overconfident mistakes.
   - The most damaging avoidable error is submitting unbounded/extreme probabilities or mismatched test ids; either can destroy log loss even if the model is good.

Pitfalls and lessons:

- Do not copy tiny-GPU settings blindly. On A100/x86, use a large batch such as `128`, `workers=4..8`, and AMP.
- Do not overcomplicate this competition before establishing the EfficientNet-B0 baseline. This single model already clears a competitive score by a large margin.
- Do not use random non-stratified validation. The dataset is balanced, but stratification keeps validation log loss reliable.
- Do not trust filesystem sorting for test submission order unless you explicitly map ids. Always read `sample_submission.csv` and build `test_paths = test_dir / f"{id}.jpg"` from it.
- Do not optimize for validation accuracy. The scoring metric is log loss, and calibration/confidence matters.
- The single most damaging step would be a submission/id-order bug or missing sigmoid/clipping, because log loss heavily penalizes confident wrong labels.

**Key Code Snippets**

Dataset and labels:

```python
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class CatDogDataset(Dataset):
    def __init__(self, paths, labels=None, transform=None):
        self.paths = list(paths)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.labels is None:
            return image, int(Path(path).stem)

        return image, torch.tensor(self.labels[idx], dtype=torch.float32)


train_paths = sorted(train_dir.glob("*.jpg"))
labels = np.asarray(
    [1 if p.name.startswith("dog.") else 0 for p in train_paths],
    dtype=np.int64,
)
```

Stratified validation split:

```python
from sklearn.model_selection import train_test_split

tr_paths, va_paths, tr_y, va_y = train_test_split(
    train_paths,
    labels,
    test_size=0.12,
    random_state=473685,
    stratify=labels,
)
```

Transforms:

```python
from torchvision import transforms

def make_transforms(image_size=224):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.78, 1.0),
            ratio=(0.8, 1.25),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.12,
            contrast=0.12,
            saturation=0.08,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, eval_tf
```

Model, loss, optimizer, scheduler:

```python
import timm
import torch.nn as nn
import torch

model = timm.create_model(
    "efficientnet_b0.ra_in1k",
    pretrained=True,
    num_classes=1,
).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max(1, epochs * len(train_loader)),
)
```

Training loop skeleton with AMP and best-checkpoint selection:

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler(enabled=device.type == "cuda")
best_loss = float("inf")

for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0

    for images, y in train_loader:
        images = images.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            logits = model(images).flatten()
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_loss += loss.item() * images.size(0)

    val_loss, val_probs, val_labels = evaluate(
        model,
        val_loader,
        device,
        clip=0.01,
    )

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "artifacts/best_model.pt")
```

Validation log loss with clipping:

```python
from sklearn.metrics import log_loss
import numpy as np
import torch

@torch.no_grad()
def evaluate(model, loader, device, clip=0.01):
    model.eval()
    labels = []
    probs = []

    for images, y in loader:
        images = images.to(device, non_blocking=True)

        with autocast(enabled=device.type == "cuda"):
            p = torch.sigmoid(model(images).flatten())

        labels.extend(y.numpy().tolist())
        probs.extend(p.float().cpu().numpy().tolist())

    probs = np.clip(np.asarray(probs, dtype=np.float64), clip, 1.0 - clip)
    labels = np.asarray(labels, dtype=np.float64)

    return log_loss(labels, probs), probs, labels
```

Test-time horizontal flip TTA:

```python
@torch.no_grad()
def predict(model, loader, device, tta=True, clip=0.01):
    model.eval()
    ids = []
    probs = []

    for images, batch_ids in loader:
        images = images.to(device, non_blocking=True)

        with autocast(enabled=device.type == "cuda"):
            logits = model(images).flatten()
            p = torch.sigmoid(logits)

            if tta:
                logits_flip = model(torch.flip(images, dims=[3])).flatten()
                p = 0.5 * (p + torch.sigmoid(logits_flip))

        probs.extend(p.float().cpu().numpy().tolist())

        if torch.is_tensor(batch_ids):
            ids.extend(batch_ids.cpu().numpy().tolist())
        else:
            ids.extend(batch_ids)

    probs = np.clip(np.asarray(probs, dtype=np.float64), clip, 1.0 - clip)
    return np.asarray(ids), probs
```

Submission construction that preserves sample order:

```python
import pandas as pd

sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

test_paths = [test_dir / f"{int(i)}.jpg" for i in sample["id"].tolist()]
missing = [str(p) for p in test_paths if not p.exists()]
if missing:
    raise FileNotFoundError(f"missing test files: {missing[:5]}")

ids, probs = predict(model, test_loader, device, tta=True, clip=0.01)

pred_df = pd.DataFrame({
    "id": ids.astype(int),
    "label": probs,
})

sub = sample[["id"]].merge(pred_df, on="id", how="left")
if sub["label"].isna().any():
    raise RuntimeError("submission contains missing predictions")

sub = sub[["id", "label"]]
sub.to_csv("submission.csv", index=False)
```

Recommended DataLoader settings:

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    CatDogDataset(tr_paths, tr_y, train_tf),
    batch_size=128,
    shuffle=True,
    num_workers=6,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True,
)

val_loader = DataLoader(
    CatDogDataset(va_paths, va_y, eval_tf),
    batch_size=256,
    shuffle=False,
    num_workers=6,
    pin_memory=True,
    persistent_workers=True,
)

test_loader = DataLoader(
    CatDogDataset(test_paths, labels=None, transform=eval_tf),
    batch_size=256,
    shuffle=False,
    num_workers=6,
    pin_memory=True,
    persistent_workers=True,
)
```

**Score Milestones (relative)**

- Naive constant `0.5` prediction: log loss ≈ `0.693` — the trivial upper bound.
- A pretrained EfficientNet-B0 transfer-learning run is the single decisive jump and lands far below that baseline.
- The configuration that reaches a strong score:
  - `efficientnet_b0.ra_in1k`
  - 224px images
  - 3 epochs
  - stratified 12% validation
  - AdamW `2e-4`
  - cosine LR
  - AMP
  - horizontal flip TTA
  - prediction clipping to `[0.01, 0.99]`

Self-check against the `criterion.json` pass line, not a fixed threshold. What stays invariant is that pretrained transfer learning plus metric-aligned validation and probability clipping is what carries the score.
