Forest Cover Type synthetic tabular multiclass classification; metric = accuracy (higher is better). The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

# Approach

Use a single strong LightGBM multiclass model with 5-fold stratified CV, Forest Cover Type feature engineering, and averaged test probabilities. This single-model recipe was sufficient to score well above the useful line.

1. Load `train.csv`, `test.csv`, and `sample_submission.csv`.
2. Inspect class counts. In this dataset, `Cover_Type == 5` has only one training row. Drop it before fitting.
3. Pop `Cover_Type` and `Id`; preserve test IDs and sample submission order for final alignment.
4. Add engineered features:
   - Collapse one-hot `Soil_Type1..40` into integer `Soil_Type`.
   - Collapse one-hot `Wilderness_Area1..4` into integer `Wilderness_Area`.
   - Add hydrology distance, road/fire interactions, elevation-distance interactions, hillshade aggregates/differences, aspect/slope sine/cosine, and elevation-by-category interactions.
5. Downcast numeric columns to reduce memory.
6. Label-encode the remaining target classes after removing class 5.
7. Train `LGBMClassifier` with `objective="multiclass"` and `num_class=len(classes)`.
8. Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=202112)`.
9. Use early stopping on `multi_error`, `stopping_rounds=150`, and average fold test probabilities.
10. Convert predicted encoded classes back to original `Cover_Type` labels with the fitted `LabelEncoder`.
11. Merge predictions back to `sample_submission[["Id"]]` and assert exact row count/order.

Core LightGBM settings that reached the strong result:

```python
params = {
    "objective": "multiclass",
    "num_class": len(classes),
    "learning_rate": 0.045,
    "n_estimators": 6000,
    "num_leaves": 192,
    "max_depth": -1,
    "min_child_samples": 80,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.82,
    "reg_alpha": 0.05,
    "reg_lambda": 4.0,
    "max_bin": 127,
    "device_type": "gpu",
    "gpu_use_dp": False,
    "n_jobs": 48,
    "random_state": 202112,
    "verbosity": -1,
}
```

If GPU is unavailable, switch to CPU by removing `device_type` and `gpu_use_dp`; expect slower training, not necessarily worse accuracy.

# What Actually Moved The Metric

Relative ordering of the levers:

The largest gain came from using a tuned LightGBM multiclass GBDT with stratified CV and probability averaging. For this dataset, a single well-configured LightGBM model was already enough to land comfortably above the useful line — no ensemble was required.

The second important piece was the Forest Cover Type feature engineering. Reconstructing `Soil_Type` and `Wilderness_Area` from their one-hot columns gave LightGBM compact categorical signals, while hydrology, road/fire, elevation, hillshade, aspect, and slope interactions supplied useful geography-derived splits.

The third important piece was handling the singleton class. `Cover_Type == 5` has only one training sample, so it cannot support reliable 5-fold stratification and provides almost no supervised signal. Drop it, train on the remaining classes, then predict only among those classes.

Pitfalls and lessons:

- Do not train directly on raw `Cover_Type` labels without encoding. LightGBM multiclass expects labels compatible with `0..num_class-1`; use `LabelEncoder` after dropping class 5 and inverse-transform predictions.
- Do not let the singleton class dictate CV design. Keeping class 5 in a 5-fold setup creates unstable or invalid folds and can degrade validation reliability.
- Do not trust row order casually. Always realign predictions to `sample_submission` by `Id` and assert exact order and length.
- The planned XGBoost/CatBoost ensemble was not part of the strong code and has no confirmed score lift in this trajectory. With limited submissions, the proven LightGBM setup is the reliable path.
- The most damaging likely mistake is mishandling class 5 or label encoding, because it can silently make CV invalid, make the multiclass objective inconsistent, or produce invalid class IDs.

# Key Code Snippets

## Feature Engineering Skeleton

```python
def add_features(df):
    df = df.copy()

    soil_cols = [f"Soil_Type{i}" for i in range(1, 41)]
    wild_cols = [f"Wilderness_Area{i}" for i in range(1, 5)]

    soil_values = df[soil_cols].to_numpy(dtype=np.int8, copy=False)
    wild_values = df[wild_cols].to_numpy(dtype=np.int8, copy=False)

    df["Soil_Type"] = (np.argmax(soil_values, axis=1) + 1).astype(np.int8)
    df["Wilderness_Area"] = (np.argmax(wild_values, axis=1) + 1).astype(np.int8)

    h_hydro = df["Horizontal_Distance_To_Hydrology"].astype(np.float32)
    v_hydro = df["Vertical_Distance_To_Hydrology"].astype(np.float32)
    road = df["Horizontal_Distance_To_Roadways"].astype(np.float32)
    fire = df["Horizontal_Distance_To_Fire_Points"].astype(np.float32)
    elev = df["Elevation"].astype(np.float32)

    df["Hydro_Distance"] = np.sqrt(h_hydro * h_hydro + v_hydro * v_hydro).astype(np.float32)
    df["Hydro_Abs_Vertical"] = np.abs(v_hydro).astype(np.float32)

    df["Road_Fire_Sum"] = (road + fire).astype(np.float32)
    df["Road_Fire_Diff"] = (road - fire).astype(np.float32)
    df["Road_Fire_Min"] = np.minimum(road, fire).astype(np.float32)
    df["Road_Fire_Max"] = np.maximum(road, fire).astype(np.float32)

    df["Elevation_Minus_Vertical_Hydro"] = (elev - v_hydro).astype(np.float32)
    df["Elevation_Plus_Vertical_Hydro"] = (elev + v_hydro).astype(np.float32)
    df["Elevation_Minus_Hydro"] = (elev - df["Hydro_Distance"]).astype(np.float32)
    df["Elevation_Minus_Road"] = (elev - road).astype(np.float32)
    df["Elevation_Minus_Fire"] = (elev - fire).astype(np.float32)

    shade9 = df["Hillshade_9am"].astype(np.float32)
    shade12 = df["Hillshade_Noon"].astype(np.float32)
    shade3 = df["Hillshade_3pm"].astype(np.float32)

    df["Hillshade_Sum"] = (shade9 + shade12 + shade3).astype(np.float32)
    df["Hillshade_Mean"] = (df["Hillshade_Sum"] / 3.0).astype(np.float32)
    df["Hillshade_Range"] = (
        np.maximum.reduce([shade9.to_numpy(), shade12.to_numpy(), shade3.to_numpy()])
        - np.minimum.reduce([shade9.to_numpy(), shade12.to_numpy(), shade3.to_numpy()])
    ).astype(np.float32)

    df["Hillshade_9am_Noon_Diff"] = (shade9 - shade12).astype(np.float32)
    df["Hillshade_Noon_3pm_Diff"] = (shade12 - shade3).astype(np.float32)

    aspect_rad = np.deg2rad(df["Aspect"].astype(np.float32))
    slope_rad = np.deg2rad(df["Slope"].astype(np.float32))

    df["Aspect_Sin"] = np.sin(aspect_rad).astype(np.float32)
    df["Aspect_Cos"] = np.cos(aspect_rad).astype(np.float32)
    df["Slope_Sin"] = np.sin(slope_rad).astype(np.float32)
    df["Slope_Cos"] = np.cos(slope_rad).astype(np.float32)

    df["Elev_Wilderness"] = (
        df["Elevation"].astype(np.int32) * df["Wilderness_Area"].astype(np.int32)
    ).astype(np.int32)
    df["Elev_Soil"] = (
        df["Elevation"].astype(np.int32) * df["Soil_Type"].astype(np.int32)
    ).astype(np.int32)

    return df
```

## Type Reduction

```python
def reduce_types(df):
    for col in df.columns:
        if col in {"Id", "Cover_Type"}:
            continue
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
    return df
```

## Target Handling

```python
class_counts = train["Cover_Type"].value_counts().sort_index()

# Class 5 has one training row. Drop it before 5-fold stratified training.
train = train.loc[train["Cover_Type"] != 5].reset_index(drop=True)

y_raw = train.pop("Cover_Type").astype(np.int16)
train_ids = train.pop("Id")
test_ids = test.pop("Id")

train = reduce_types(add_features(train))
test = reduce_types(add_features(test))

features = list(train.columns)
categorical_features = ["Soil_Type", "Wilderness_Area"]

encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)
classes = encoder.classes_.astype(int)
```

## 5-Fold LightGBM CV And Test Averaging

```python
oof = np.zeros((len(train), len(classes)), dtype=np.float32)
test_pred = np.zeros((len(test), len(classes)), dtype=np.float32)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=202112)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
    model = lgb.LGBMClassifier(**params)

    model.fit(
        train.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(train.iloc[va_idx], y[va_idx])],
        eval_metric="multi_error",
        categorical_feature=categorical_features,
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    valid_pred = model.predict_proba(
        train.iloc[va_idx],
        num_iteration=model.best_iteration_,
    )
    oof[va_idx] = valid_pred.astype(np.float32)

    fold_acc = accuracy_score(y[va_idx], valid_pred.argmax(axis=1))
    print(f"fold={fold} accuracy={fold_acc:.6f} best_iter={model.best_iteration_}")

    test_pred += (
        model.predict_proba(test, num_iteration=model.best_iteration_) / skf.n_splits
    ).astype(np.float32)
```

## OOF Check And Submission Alignment

```python
oof_score = accuracy_score(y, oof.argmax(axis=1))
print(f"OOF accuracy={oof_score:.6f}")

test_labels = encoder.inverse_transform(test_pred.argmax(axis=1)).astype(np.int64)

sub = pd.DataFrame({
    "Id": test_ids.to_numpy(),
    "Cover_Type": test_labels,
})

sub = sample[["Id"]].merge(sub, on="Id", how="left")

if sub["Cover_Type"].isna().any():
    raise RuntimeError("Missing predictions after aligning to sample_submission Ids")

sub["Cover_Type"] = sub["Cover_Type"].astype(sample["Cover_Type"].dtype)
sub = sub[sample.columns]

if len(sub) != len(sample) or not sub["Id"].equals(sample["Id"]):
    raise RuntimeError("Submission does not match sample_submission Id/order")

sub.to_csv("submission.csv", index=False)
```

# Score Milestones

Relative ordering (what each step buys):

A naive/simple baseline was not recorded in this trajectory. The strongest configuration — engineered-feature LightGBM multiclass, 5-fold stratified CV, singleton class 5 removed, fold probability averaging, sample-aligned submission — lands comfortably above the useful line with a single model.

Steer by the `criterion.json` pass line, not a fixed threshold. What stays invariant is the ordering: the tuned LightGBM multiclass GBDT with stratified CV and probability averaging is the main lever, the Forest Cover Type feature engineering is second, and correct singleton-class/label handling protects CV validity.
