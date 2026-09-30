Plant Pathology 2020 FGVC7 image classification: predict probabilities for `healthy`, `multiple_diseases`, `rust`, `scab`; evaluation is mean ROC-AUC over the 4 class columns. The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

## Approach

Use this as a high-confidence recipe for this competition.

1. Treat the task as **4-class single-label classification**, not independent multilabel classification.
   - The train labels are one-hot across `healthy`, `multiple_diseases`, `rust`, `scab`.
   - Train with `CrossEntropyLoss` on `argmax(label_columns)`.
   - Submit `softmax(logits)` probabilities for the four columns.

2. Train a pretrained image classifier from `timm`.
   - Strong default: `tf_efficientnet_b3_ns`
   - Good variants to try if time remains: EfficientNet-B4 or `convnext_tiny`
   - Use `num_classes=4`.

3. Use 5-fold stratified CV on the class index.
   - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
   - Average test predictions across folds.
   - Track OOF mean column ROC-AUC.

4. Use 512px crops/resizing.
   - Training: `RandomResizedCrop(512, scale=(0.72, 1.0), ratio=(1.25, 1.65))`
   - Validation/test: deterministic `Resize((512, 512))`
   - Normalize with ImageNet mean/std.

5. Use moderate regularization.
   - `epochs=8`
   - `batch_size=24`
   - `eval_batch_size=48`
   - `lr=2e-4`
   - `weight_decay=1e-4`
   - `label_smoothing=0.05`
   - `mixup_alpha=0.15`
   - `patience=3`
   - optimizer: `AdamW`
   - scheduler: per-batch cosine annealing to `lr * 0.02`
   - AMP enabled on CUDA
   - gradient clipping at `1.0`

6. Use simple flip TTA for final validation and test inference.
   - Average predictions over original, horizontal flip, vertical flip, and both flips.
   - Apply TTA only for final fold checkpoint inference and test prediction, not for every validation epoch.

## What Actually Moved The Metric

The strong result came from a straightforward but robust setup: pretrained EfficientNet at 512px, stratified 5-fold CV, and fold averaging with TTA. The attempt history only records the successful trajectory, so do not overfit to unrecorded ablations. The known high-impact choices are:

1. **Single-label softmax classification**
   - The winning code converts one-hot targets to class indices and trains `CrossEntropyLoss`.
   - This matters because the labels are mutually exclusive, while the submission still requires one probability per class.
   - A common damaging mistake is to treat the task as four unrelated binary labels with unconstrained probabilities. That can work somewhat with ROC-AUC, but it ignores the strongest label structure and usually gives worse-calibrated rankings.

2. **5-fold stratified ensemble**
   - Public train size is small enough that single splits are noisy.
   - The winning solution trains all 5 folds and averages test predictions.
   - This is likely the main lift from a competent single model to top-level stability.

3. **512px pretrained model with strong but not extreme augmentation**
   - Leaf disease details matter; too-small image sizes discard lesion texture.
   - The successful crop keeps the leaf aspect bias using `ratio=(1.25, 1.65)` rather than generic square-only random crops.
   - Color jitter, flips, rotation, random erasing, label smoothing, and light mixup add useful robustness without making the task too synthetic.

Pitfalls and lessons:

- Do not submit class labels or hard one-hot predictions. Submit probabilities in exactly the sample submission column order.
- Do not accidentally reorder test rows. Preserve `sample_submission.csv` order and assert `image_id` equality.
- Do not optimize only accuracy. Select checkpoints by mean per-column ROC-AUC.
- Do not skip CV if targeting the best result. A single fold/model may be good, but the strongest recorded score used 5-fold averaging.
- The single most damaging step would be mishandling the target formulation or submission columns: wrong column order, missing softmax probabilities, or treating `image_id` order casually can destroy the score even with a strong model.

## Essential Code Skeleton

### Constants And Dataset

```python
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

DATA_DIR = Path("./input")
TARGETS = ["healthy", "multiple_diseases", "rust", "scab"]

class PlantDataset(Dataset):
    def __init__(self, df, image_dir, transform, train: bool):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.image_dir / f"{row.image_id}.jpg").convert("RGB")
        image = self.transform(image)

        if self.train:
            label = int(row[TARGETS].values.astype(np.float32).argmax())
            return image, label

        return image, row.image_id
```

### Transforms

```python
from torchvision import transforms

def get_transforms(image_size=512):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.72, 1.0),
            ratio=(1.25, 1.65),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.18,
            contrast=0.18,
            saturation=0.18,
            hue=0.03,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(
            p=0.18,
            scale=(0.02, 0.12),
            ratio=(0.3, 3.3),
            value="random",
        ),
    ])

    valid_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    return train_tfms, valid_tfms
```

### Metric

```python
from sklearn.metrics import roc_auc_score

def auc_score(y_true, pred):
    scores = []
    for i in range(len(TARGETS)):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        scores.append(roc_auc_score(y_true[:, i], pred[:, i]))
    return float(np.mean(scores))
```

### CV Split

```python
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

labels = train[TARGETS].values.argmax(1)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

oof = np.zeros((len(train), len(TARGETS)), dtype=np.float32)
test_pred = np.zeros((len(test), len(TARGETS)), dtype=np.float32)

for fold, (trn_idx, val_idx) in enumerate(skf.split(train, labels)):
    train_df = train.iloc[trn_idx]
    valid_df = train.iloc[val_idx]
    # train one fold, save best checkpoint by validation AUC
    # fill oof[val_idx]
    # add fold test probabilities / 5
```

### Model, Class Weights, Loss, Optimizer

```python
import torch
import torch.nn as nn
import timm

model = timm.create_model(
    "tf_efficientnet_b3_ns",
    pretrained=True,
    num_classes=len(TARGETS),
).to(device)

counts = train_df[TARGETS].values.argmax(1)
class_counts = np.bincount(counts, minlength=len(TARGETS)).astype(np.float32)
weights = class_counts.sum() / np.maximum(class_counts, 1.0)
weights = weights / weights.mean()

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(weights, dtype=torch.float32, device=device),
    label_smoothing=0.05,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max(1, 8 * len(train_loader)),
    eta_min=2e-4 * 0.02,
)
```

### Mixup Training Step

```python
def mixup(x, y, alpha=0.15):
    if alpha <= 0:
        return x, y, None, 1.0

    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam
```

```python
from torch.amp import GradScaler, autocast

scaler = GradScaler(device.type, enabled=device.type == "cuda")

model.train()
for images, labels in train_loader:
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    images, labels_a, labels_b, lam = mixup(images, labels, alpha=0.15)

    optimizer.zero_grad(set_to_none=True)

    with autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits = model(images)
        if labels_b is None:
            loss = criterion(logits, labels_a)
        else:
            loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
```

### Prediction With Flip TTA

```python
import torch
from torch.amp import autocast

@torch.no_grad()
def predict_loader(model, loader, device, tta: bool):
    model.eval()
    preds = []

    for images, _ in loader:
        images = images.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            prob = torch.softmax(logits, dim=1)

            if tta:
                for dims in [(-1,), (-2,), (-2, -1)]:
                    logits_t = model(torch.flip(images, dims=dims))
                    prob = prob + torch.softmax(logits_t, dim=1)

                prob = prob / 4.0

        preds.append(prob.float().cpu().numpy())

    return np.concatenate(preds, axis=0)
```

### Fold Checkpointing And Ensembling

```python
best_auc = -1.0
best_path = out_dir / f"tf_efficientnet_b3_ns_fold{fold}.pt"
patience = 0
y_valid = valid_df[TARGETS].values.astype(np.float32)

for epoch in range(1, 8 + 1):
    # train one epoch
    pred = predict_loader(model, valid_loader, device, tta=False)
    val_auc = auc_score(y_valid, pred)

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), best_path)
        patience = 0
    else:
        patience += 1
        if patience >= 3:
            break

model.load_state_dict(torch.load(best_path, map_location=device))

valid_pred = predict_loader(model, valid_loader, device, tta=True)
fold_auc = auc_score(y_valid, valid_pred)
oof[valid_indices] = valid_pred

test_pred += predict_loader(model, test_loader, device, tta=True) / 5
```

### Submission Safety Checks

```python
sub = sample.copy()
sub[TARGETS] = test_pred
sub = sub[sample.columns]

assert sub.shape == sample.shape
assert sub["image_id"].tolist() == sample["image_id"].tolist()
assert np.isfinite(sub[TARGETS].values).all()

sub.to_csv("submission.csv", index=False)
```

## Score Milestones

- Naive random or constant-ranking baseline: approximately `0.50` mean ROC-AUC.
- Competent pretrained single model at high resolution: should clear the median comfortably if validation is sane.
- Strongest configuration: `tf_efficientnet_b3_ns`, 512px, 5 stratified folds, CE softmax, class weights, label smoothing, mixup, flip TTA, fold averaging.

The absolute scores above are trajectory anchors; what stays invariant is the ORDERING — single-label softmax over the four mutually-exclusive classes beats independent-binary framing, and 5-fold fold-averaging is the main lift from a competent single model to top stability. Self-check against the `criterion.json` pass line, not a fixed threshold.
