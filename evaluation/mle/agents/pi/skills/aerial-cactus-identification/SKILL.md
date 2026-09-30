# SKILL.md — aerial-cactus-identification

**Task type / metric**: binary classification of 32×32 aerial thumbnails (whether a cactus is present), metric **ROC-AUC**. This is a classic easy CV task where scores cluster extremely tightly near the top — a huge number of near-perfect scores, so the entire separation lives in the relative ordering of a handful of test boundary samples. In practice the only way to clear the pass line is to push AUC essentially to a perfect 1.0; getting the relative ordering wrong on even a few boundary samples costs the run. The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

---

## Winning recipe (executable steps)

The essence is not "swapping in a stronger model" but **heterogeneous multi-backbone 5-fold + D4 TTA + rank/prob grid blending**, correcting the relative ordering of the handful of boundary samples that get flipped.

1. **Data**: `train.csv` / `sample_submission.csv` + extract `train.zip` / `test.zip` into a local cache directory (verify counts after extraction). Label column `has_cactus`, id column `id`.
2. **CV**: `StratifiedKFold(n_splits=5, shuffle=True)`, with the **fold count fixed at 5** (all cached predictions are 5-fold, so this must align). Run each backbone with different seeds (42 / 202 / 777) to increase diversity.
3. **Backbone pool (heterogeneous, complementary)**: a validated set = `efficientnet_b0`, `densenet121`, `resnet34` (each seed 42), plus `densenet121`, `resnet34` (seed 202); add `convnext_nano` (size 96) and `tf_efficientnet_b0` (size 128) for more diversity. All use `timm.create_model(name, pretrained=True, num_classes=1)`, single logit + `BCEWithLogitsLoss`.
4. **Key hyperparameters** (defer to the actual code):
   - input size **96** (upsampling the 32×32 images is an effective trick, see below); `tf_efficientnet_b0` uses 128.
   - `epochs=40, min_epochs=18, patience=8` (early stopping on val-AUC), `batch_size=256`.
   - `AdamW(lr=1e-3, weight_decay=1e-4)` + `CosineAnnealingLR(T_max=epochs)`.
   - `label_smoothing=0.05`, AMP fp16.
   - Augmentation: Resize → random 90/180/270 rotation (p=.5) + H/V flip + light ColorJitter (p=.35) + light RandomResizedCrop (scale .86–1.0, p=.25).
5. **TTA**: inference uses **D4 eight-orientation TTA** (flip × {0,90,180,270}) averaged; both OOF and test go through TTA.
6. **Blending + selection**: store each backbone's OOF (valid) and test `.npy`, and **select the blend on a grid** using OOF-AUC: compare prob-averaging vs **rank-averaging** modes, anchor on the best-known weights and do small-step perturbation over the grid, and write the submission from the candidate with the highest OOF-AUC. Final prediction `np.clip(test_pred, 1e-7, 1-1e-7)`.

---

## The tricks that actually move the metric (+ cautionary lessons)

**Starting point**: a single `resnet18` alone already reaches a near-perfect AUC; 3 backbones 5-fold+TTA gets even closer to perfect. Closing the final tiny gap to a full 1.0 relies on **heterogeneous diversity + upsampling + rank blending** to fix the handful of mis-ordered boundary samples:

1. **rank blending > prob blending (decisive)**. AUC only cares about relative ordering; at this precision, probability averaging gets drowned out by the scale of individual models, whereas applying `rank/len` per model first and then taking a weighted average can fix the boundary pairs where "a minority of models are right and the majority are wrong." The blending script enumerates both prob and rank candidates and adjudicates by OOF-AUC — rank usually wins.
2. **Upsample the small images to 96/128**. Feeding 32×32 directly to the network provides too little information; resizing to 96 (some backbones 128) lets the pretrained CNN shine, and it is the key to separating the last few sample pairs.
3. **Heterogeneous backbones + multiple seeds, not homogeneous stacking**. The trajectory is clear: a homogeneous resnet18 ensemble "tops out." The real breakthrough is introducing structurally different backbones (effb0/densenet/resnet/convnext/tf_effb0) + different seeds, which provide complementary errors so that blending has something to fix.

**Cautionary lessons (don't step on these)**:
- **Don't get fancy with semi-supervised/pseudo-labeling and the like**. Once you're already essentially at the ceiling, a complex pipeline is risk > reward, and **steadily thickening and hardening the already-validated ensemble** is what crosses the pass line. (Note: earlier "semi-supervised self-training" attempts in the trajectory look like failures but were actually interrupted by infra/quota issues and never really ran — so treat them as unproven, not disproven, and still deprioritize them because the ensemble is the safer lever at this precision.)
- **Don't pile on more homogeneous ensembles**, the returns have already topped out.
- **Don't touch the fold count** (must be 5), otherwise it won't align with the cached OOF/test predictions and the blend collapses outright.
- OOF is **extremely fragile** at this precision; deliberately keep the grid low-dimensional, anchor on the best-known weights with small-step perturbations, and don't overfit OOF.

---

## Key code snippets

**Model head + loss (single-logit binary classification)**
```python
model = timm.create_model(model_name, pretrained=True, num_classes=1)
criterion = nn.BCEWithLogitsLoss()
# label smoothing on target
if args.label_smoothing > 0:
    y = y * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
logits = model(x).reshape(-1)
loss = criterion(logits, y)
```

**5-fold CV (seed must match the cached predictions)**
```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df["id"], y)):
    ...
# early stopping: on val-AUC, min_epochs=18, patience=8
auc = roc_auc_score(va_df["has_cactus"].to_numpy(), val_pred)
if auc > best_auc + 1e-8:
    best_auc, best_state, bad_epochs = auc, copy_state(model), 0
else:
    bad_epochs += 1
    if epoch >= args.min_epochs and bad_epochs >= args.patience:
        break
```

**D4 eight-orientation TTA (training size=96)**
```python
def d4_tta_transforms(size):
    ops = []
    for flip in [False, True]:
        for angle in [0, 90, 180, 270]:
            def op(img, flip=flip, angle=angle):
                out = img.resize((size, size), Image.Resampling.BILINEAR)
                if flip: out = out.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if angle: out = out.rotate(angle)
                return out
            ops.append(op)
    return [Compose([Lambda(op), ToTensor(), Normalize(mean, std)]) for op in ops]
# at inference, average the sigmoid probabilities over the 8 transforms
pred = np.mean([predict(model, loader_for(tfm)) for tfm in d4_tta_transforms(size)], axis=0)
```

**rank blending + OOF grid selection (the most valuable part)**
```python
def rank_1d(a):
    return pd.Series(a).rank(method="average").to_numpy(np.float64) / len(a)

def weighted_rank(arrs, w):
    return np.average(np.vstack([rank_1d(a) for a in arrs]), axis=0, weights=w)

# each candidate computes its OOF-AUC; both prob and rank modes enter the candidate pool
if mode == "prob":
    oof_pred, test_pred = np.average(oofs, 0, w), np.average(tests, 0, w)
else:  # rank
    oof_pred, test_pred = weighted_rank(oofs, w), weighted_rank(tests, w)
candidates.append((roc_auc_score(y, oof_pred), name, mode, w, test_pred))

# anchor on the historical best weights a5 with small-step perturbations + let the strongest model dominate per OOF
best = max(candidates, key=lambda x: x[0])     # pick the highest OOF-AUC
sub["has_cactus"] = np.clip(best.test_pred, 1e-7, 1 - 1e-7)
```

**Submission sanity check (guard against silly mistakes)**
```python
assert sub["id"].equals(sample_df["id"])            # id ordering consistent
assert not sub.isna().any().any()
assert not sub["id"].duplicated().any()
assert sub["has_cactus"].nunique() >= 1000          # must be continuous probabilities, not 0/1 hard labels
```

---

## Score milestones (relative — what each step buys)

- A single `resnet18` naive baseline already lands near-perfect AUC.
- 3 backbones 5-fold + D4 TTA closes most of the remaining gap but still leaves a few mis-ordered boundary samples.
- **Heterogeneous multi-backbone + upsampling + rank grid blending** is what fixes those last boundary pairs and reaches a full 1.0.

What stays invariant is the ORDERING: near-perfect single models plateau; only heterogeneous diversity + upsampling + rank blending closes the final gap. Self-check against the `criterion.json` pass line — the remaining gap is entirely in the relative ordering of a few test boundary samples; attack it with rank blending, not a bigger model.
