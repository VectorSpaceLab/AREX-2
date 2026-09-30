Histopathologic cancer detection: binary classification on 96x96 PCam pathology patches, evaluated by ROC AUC (higher is better). The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

## Approach

Use a compact ImageNet-pretrained CNN, not a large custom pipeline. The strongest run used `timm` `tf_efficientnet_b0_ns` with a single sigmoid logit head, trained on one stratified fold and submitted test predictions from the best-validation checkpoint.

Executable recipe:

1. Read:
   - `train_labels.csv`
   - `sample_submission.csv`
   - image folders `train/{id}.tif` and `test/{id}.tif`

2. Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on `label`.

3. Train fold `0` by default:
   - model: `tf_efficientnet_b0_ns`
   - pretrained: `True`
   - `num_classes=1`
   - image size: `96`
   - epochs: `3`
   - batch size: `256`
   - validation/test batch size: `512`
   - optimizer: `AdamW`
   - learning rate: `3e-4`
   - weight decay: `1e-4`
   - scheduler: `CosineAnnealingLR`
   - loss: `BCEWithLogitsLoss`
   - AMP: CUDA autocast fp16 + `GradScaler`
   - workers: `8`
   - seed: `42`

4. Save the checkpoint with the best validation ROC AUC.

5. Predict test using the best checkpoint with simple flip TTA:
   - original
   - horizontal flip
   - vertical flip
   - average logits, then sigmoid

6. Clip probabilities to `[1e-6, 1 - 1e-6]` and preserve exact `sample_submission.csv` row order.

## What Actually Moved The Metric

The available trajectory is sparse, but the strong code shows the high-leverage pieces clearly.

1. **Pretrained EfficientNet-B0 was sufficient.**
   The winning submission did not need a heavy ensemble, custom MIL logic, segmentation, or high-resolution tiling. A `tf_efficientnet_b0_ns` ImageNet-pretrained backbone trained directly on the 96x96 patches reached a strong AUC on its own.

2. **Stratified validation and best-checkpoint selection mattered.**
   The code uses 5-fold stratification but trains/submits one selected fold. This keeps validation AUC meaningful for a balanced model-selection signal. Do not use a random unstratified split; PCam is simple enough that leakage or split noise can mislead you.

3. **Patch-appropriate augmentation plus flip TTA was enough.**
   The useful augmentations were pathology-safe symmetries and mild stain/color variation:
   - horizontal flip
   - vertical flip
   - 90/180/270 degree rotations
   - mild brightness/contrast/saturation/hue jitter

   Test-time augmentation only averages original, horizontal-flipped, and vertical-flipped logits. Keep TTA simple; this task benefits from orientation invariance, but complex postprocessing is unnecessary.

Pitfalls and lessons:

- **Most damaging mistake:** breaking submission row order. Always start from `sample_submission.csv`, assign predictions into its `label` column, and assert the output `id` column equals the sample IDs exactly.
- **Do not train on CPU.** The strong code explicitly refuses CPU training. This dataset has many image files, and CPU-only training will waste the attempt.
- **Do not use `BCELoss` on sigmoid probabilities during training.** Use `BCEWithLogitsLoss` on raw logits for numerical stability.
- **Do not resize upward unnecessarily.** The successful run used the native `96x96` size. Larger image sizes cost time and are not needed.
- **Do not overbuild an ensemble before getting the single-fold baseline.** The strong result came from one EfficientNet-B0 fold plus TTA. Multi-fold ensembling can help, but it is not required.
- **Do not use aggressive color or geometric transforms.** Heavy stain perturbation, arbitrary rotations with interpolation, blur, cutout, or strong augmentations may damage small tumor-region cues. Use mild color jitter and exact right-angle/flip symmetries.

## Key Code Snippets

### Dataset

Use PIL to load `.tif` patches and preserve IDs from the dataframe.

```python
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class PcamDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, labels=True):
        self.ids = df["id"].to_numpy()
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.labels = labels
        self.y = df["label"].to_numpy(dtype=np.float32) if labels else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = Image.open(self.image_dir / f"{self.ids[idx]}.tif").convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.labels:
            return image, torch.tensor(self.y[idx], dtype=torch.float32)
        return image
```

### Transforms

Use ImageNet normalization because the model is ImageNet-pretrained.

```python
from torchvision import transforms

def build_transforms(image_size=96):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.25),
        transforms.RandomApply([transforms.RandomRotation((180, 180))], p=0.25),
        transforms.RandomApply([transforms.RandomRotation((270, 270))], p=0.25),
        transforms.ColorJitter(
            brightness=0.12,
            contrast=0.12,
            saturation=0.10,
            hue=0.02,
        ),
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    valid_tf = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, valid_tf
```

### Split

Use stratified folds and train one fold first. The code defaults to fold `0`.

```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DATA_DIR = Path("./input")

train_df = pd.read_csv(DATA_DIR / "train_labels.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, valid_idx = list(skf.split(train_df["id"], train_df["label"]))[0]

trn_df = train_df.iloc[train_idx].reset_index(drop=True)
val_df = train_df.iloc[valid_idx].reset_index(drop=True)

print(
    f"train={len(trn_df)} valid={len(val_df)} "
    f"test={len(sample)} pos_rate={train_df.label.mean():.4f}"
)
```

### DataLoaders

Use large batches; these are tiny 96x96 images.

```python
from torch.utils.data import DataLoader

train_tf, valid_tf = build_transforms(image_size=96)

train_ds = PcamDataset(trn_df, DATA_DIR / "train", train_tf, labels=True)
valid_ds = PcamDataset(val_df, DATA_DIR / "train", valid_tf, labels=True)
test_ds = PcamDataset(sample, DATA_DIR / "test", valid_tf, labels=False)

train_loader = DataLoader(
    train_ds,
    batch_size=256,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)

valid_loader = DataLoader(
    valid_ds,
    batch_size=512,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

test_loader = DataLoader(
    test_ds,
    batch_size=512,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)
```

### Model, Loss, Optimizer, Scheduler

Use `num_classes=1` and raw logits.

```python
import timm
import torch
import torch.nn as nn

assert torch.cuda.is_available(), "CUDA is required"
device = torch.device("cuda")

model = timm.create_model(
    "tf_efficientnet_b0_ns",
    pretrained=True,
    num_classes=1,
).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=3 * len(train_loader),
    eta_min=3e-4 * 0.03,
)

scaler = torch.cuda.amp.GradScaler()
```

If pretrained weights fail to download in the environment, retrying with `pretrained=False` keeps the code runnable, but expect a lower score. For serious attempts, make sure pretrained weights are available.

```python
def make_model(model_name="tf_efficientnet_b0_ns", pretrained=True):
    try:
        return timm.create_model(model_name, pretrained=pretrained, num_classes=1)
    except Exception:
        if pretrained:
            return timm.create_model(model_name, pretrained=False, num_classes=1)
        raise
```

### Training Loop

Step the cosine scheduler every batch, not every epoch.

```python
def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, device):
    model.train()
    losses = []
    optimizer.zero_grad(set_to_none=True)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).view(-1, 1)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses))
```

### Validation And TTA Prediction

Compute AUC on sigmoid probabilities. For TTA, average logits before sigmoid.

```python
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

@torch.no_grad()
def predict_loader(model, loader, device, tta=False):
    model.eval()
    preds = []

    for x in loader:
        if isinstance(x, (list, tuple)):
            x = x[0]

        x = x.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)

            if tta:
                logits = logits + model(torch.flip(x, dims=[3]))  # horizontal
                logits = logits + model(torch.flip(x, dims=[2]))  # vertical
                logits = logits / 3.0

        preds.append(torch.sigmoid(logits).float().cpu().numpy().reshape(-1))

    return np.concatenate(preds)


best_auc = -1.0
val_y = val_df["label"].to_numpy()

for epoch in range(1, 4):
    loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, scaler, criterion, device
    )

    val_pred = predict_loader(model, valid_loader, device, tta=False)
    auc = roc_auc_score(val_y, val_pred)

    if auc > best_auc:
        best_auc = auc
        torch.save(
            {"model": model.state_dict(), "auc": best_auc},
            "artifacts/tf_efficientnet_b0_ns_fold0_best.pt",
        )
```

### Submission

Reload the best checkpoint, predict test with TTA, clip probabilities, and validate the file shape/order.

```python
ckpt = torch.load(
    "artifacts/tf_efficientnet_b0_ns_fold0_best.pt",
    map_location=device,
)
model.load_state_dict(ckpt["model"])

test_pred = predict_loader(model, test_loader, device, tta=True)

submission = sample.copy()
submission["label"] = np.clip(test_pred, 1e-6, 1 - 1e-6)
submission.to_csv("submission.csv", index=False)

check = pd.read_csv("submission.csv")
assert list(check.columns) == list(sample.columns)
assert len(check) == len(sample)
assert check["id"].equals(sample["id"])
assert check["label"].between(0, 1).all()
```

## Recommended Run Order

1. First run a small smoke test with `max_train`/`max_valid` equivalents if you add them, just to verify image loading, CUDA, and submission formatting.
2. Run the full fold-0 training for 3 epochs with `tf_efficientnet_b0_ns`.
3. Check validation AUC. A healthy run should be comfortably in the high `0.98x` range locally.
4. Submit the TTA prediction file.
5. Only if the score is unexpectedly low, try:
   - another fold and average test predictions
   - `efficientnet_b1` or another small `timm` model
   - slightly longer training, such as 4-5 epochs

Do not spend early time on complicated pathology-specific tricks; the simple pretrained EfficientNet pipeline is already strong.

## Score Milestones (relative — what each step buys)

- Naive/random probability baseline: about `0.5` ROC AUC.
- Basic competent CNN without the full recipe: should clear the competition median if trained correctly.
- Strongest configuration from this trajectory: `tf_efficientnet_b0_ns`, 5-fold stratified split using fold 0, 3 epochs, batch size 256, AdamW `3e-4`, cosine schedule, mild flip/rotation/color augmentation, flip TTA — lands in the high `0.98x` range.

The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score. What stays invariant is the ORDERING: a pretrained EfficientNet-B0 on native-resolution patches with stratified selection is the decisive baseline, and simple flip TTA is the cheap finisher — heavy ensembles, upscaling, and aggressive augmentation are not needed to reach a strong score.
