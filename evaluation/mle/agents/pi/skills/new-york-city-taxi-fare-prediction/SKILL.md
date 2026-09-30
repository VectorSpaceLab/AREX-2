Task: tabular geospatial regression for NYC taxi fare prediction; metric: RMSE on `fare_amount` (lower is better). The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

# NYC Taxi Fare Prediction Skill

## Core Approach

Start from a clean LightGBM geospatial fare model, not from cached submissions or route-prior lookups.

The strongest known path is:

1. Read the full training file in chunks from the MLE-bench public data directory:
   - Prefer `labels.csv`; fall back to `train.csv`.
   - Test file is `test.csv`.
   - Submission template is `sample_submission.csv`.

2. Train on a sampled cleaned subset:
   - Target sample size: `N_SAMPLE = 6_000_000`.
   - Chunk size: `2_000_000`.
   - Keep only fares in `[2.5, 500]`.
   - Keep only `passenger_count` in `[1, 6]`.
   - Main model training rows must have pickup and dropoff coordinates inside a tight NYC bounding box:
     - longitude `[-74.3, -73.7]`
     - latitude `[40.5, 41.0]`
   - Additionally filter obviously bad fare-distance rows:
     - distance in `[0.01, 120.0]` km
     - remove trips with `dist < 0.10` and fare `> 20`
     - keep `fare_per_km` in `[0.35, 80.0]`

3. Preserve a separate dirty-coordinate holdout:
   - Keep up to `N_DIRTY_HOLDOUT = 120_000` rows where fare/passenger are valid but coordinates are invalid.
   - Use this only to choose the best fallback for invalid test rows.
   - Do not delete invalid test rows; they must receive predictions.

4. Feature set:
   - Haversine distance.
   - Manhattan-style distance.
   - Bearing.
   - Absolute latitude/longitude deltas.
   - Pickup and dropoff distance to JFK, LGA, EWR, and Manhattan center.
   - Year, hour, weekday, month.
   - `dist_year = hav * year`.
   - Raw passenger count and raw coordinates.

5. Target transform:
   - Train LightGBM on `log1p(fare_amount)`.
   - Convert predictions with `expm1`.

6. Validation:
   - Use a deterministic random 90/10 holdout split with `SEED = 42`.
   - Report holdout RMSE on the original fare scale, not log scale.
   - The reference script expects roughly low-3s holdout/public behavior when the invalid-test fallback is working.

7. LightGBM hyperparameters from the strongest known code:
   - `objective="regression"`
   - `metric="rmse"`
   - `num_leaves=100`
   - `learning_rate=0.05`
   - `feature_fraction=0.8`
   - `bagging_fraction=0.8`
   - `bagging_freq=1`
   - `min_data_in_leaf=200`
   - `max_bin=255`
   - `num_boost_round=3000`
   - `early_stopping_rounds=100`
   - `num_threads=min(128, os.cpu_count() or 16)`
   - `seed=42`

8. Critical post-processing:
   - Identify invalid test coordinates using the same tight NYC bounding box.
   - Do not let the main model extrapolate on invalid coordinates.
   - Replace invalid-coordinate predictions using a fallback chosen from the dirty-coordinate holdout:
     - global training median,
     - dirty-year median,
     - or coordinate-swap-if-valid else global median.
   - Clip every prediction to `[2.5, 250.0]`.

## What Actually Moved The Metric

The best submitted score in the trajectory was only around the competition median and far from a passing result. That run appears to have benefited from reused old prediction artifacts rather than a clean reproduced solution. Treat it as evidence that the metric is beatable, not as a trustworthy final recipe.

The important lessons from the failed attempts:

1. Invalid test-coordinate handling is the biggest known lever.
   - Repeated poor (4.x RMSE) submissions were caused by allowing the model to extrapolate on rows with coordinates like `(0, 0)`, nulls, or out-of-box locations.
   - A few huge invalid-row errors can dominate RMSE.
   - The fix is to route invalid test rows away from the model and use robust medians or swap-repaired predictions.

2. The feature spine matters.
   - The useful model is not just raw coordinates plus distance.
   - The reference implementation uses six airport distance features, Manhattan center distances, `year`, `bearing`, and `dist_year`.
   - `year` is especially important because fare rules and inflation change over time.

3. Training must be bounded.
   - Several attempts failed because a 6M-row LightGBM script ran until the `12000s` wall-clock limit and never produced a fresh submission.
   - If runtime is tight, first make a smaller clean reproduction with `2M-4M` sampled rows and lower rounds, then scale to `6M`.
   - Do not fall back to stale cached predictions just because training is slow.

Regressions and dead ends:

- Blindly rerunning long reference scripts without timeout control repeatedly produced no useful fresh model.
- Rewriting the pipeline from scratch without preserving the known feature set and invalid-row fallback stayed in the 4.x range.
- Route-prior tables, coordinate-rounded historical lookup caches, and reused submission files are not a clean solution path and should be avoided unless explicitly doing a separate ensembling experiment.
- The single most damaging mistake is predicting invalid test rows with the main geospatial model and only clipping afterward. Clipping alone does not fix the problem.

## Essential Code Skeleton

Use these snippets as the starting structure. They are condensed from the strongest known implementation; keep the logic intact before experimenting.

### Paths And Constants

```python
import os, gc
import numpy as np
import pandas as pd
import lightgbm as lgb

DATA = "./input"
TRAIN = os.path.join(DATA, "labels.csv")
if not os.path.exists(TRAIN):
    TRAIN = os.path.join(DATA, "train.csv")

TEST = os.path.join(DATA, "test.csv")
SAMPLE = os.path.join(DATA, "sample_submission.csv")

N_SAMPLE = 6_000_000
N_DIRTY_HOLDOUT = 120_000
SEED = 42
NUM_THREADS = min(128, os.cpu_count() or 16)

JFK = (40.6413, -73.7781)
LGA = (40.7769, -73.8740)
EWR = (40.6895, -74.1745)
NYC = (40.7589, -73.9851)

LON_MIN, LON_MAX = -74.3, -73.7
LAT_MIN, LAT_MAX = 40.5, 41.0

RENAME = {
    "pickup_longitude": "plon",
    "pickup_latitude": "plat",
    "dropoff_longitude": "dlon",
    "dropoff_latitude": "dlat",
}
```

### Geospatial Features

```python
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p = np.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def manhattan(lat1, lon1, lat2, lon2):
    return haversine(lat1, lon1, lat1, lon2) + haversine(lat1, lon1, lat2, lon1)


def bearing(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    dlon = (lon2 - lon1) * p
    y = np.sin(dlon) * np.cos(lat2 * p)
    x = (
        np.cos(lat1 * p) * np.sin(lat2 * p)
        - np.sin(lat1 * p) * np.cos(lat2 * p) * np.cos(dlon)
    )
    return np.arctan2(y, x)


def valid_coords(df):
    return (
        df.plon.between(LON_MIN, LON_MAX)
        & df.plat.between(LAT_MIN, LAT_MAX)
        & df.dlon.between(LON_MIN, LON_MAX)
        & df.dlat.between(LAT_MIN, LAT_MAX)
    )


def swapped_valid_coords(df):
    return (
        df.plat.between(LON_MIN, LON_MAX)
        & df.plon.between(LAT_MIN, LAT_MAX)
        & df.dlat.between(LON_MIN, LON_MAX)
        & df.dlon.between(LAT_MIN, LAT_MAX)
    )


def swap_lon_lat(df):
    out = df.copy()
    out[["plon", "plat"]] = out[["plat", "plon"]]
    out[["dlon", "dlat"]] = out[["dlat", "dlon"]]
    return out


def add_features(df):
    df["hav"] = haversine(df.plat, df.plon, df.dlat, df.dlon)
    df["man"] = manhattan(df.plat, df.plon, df.dlat, df.dlon)
    df["brg"] = bearing(df.plat, df.plon, df.dlat, df.dlon)
    df["dlat_abs"] = (df.dlat - df.plat).abs()
    df["dlon_abs"] = (df.dlon - df.plon).abs()

    for nm, (la, lo) in [("jfk", JFK), ("lga", LGA), ("ewr", EWR), ("ctr", NYC)]:
        df[f"p_{nm}"] = haversine(df.plat, df.plon, la, lo)
        df[f"d_{nm}"] = haversine(df.dlat, df.dlon, la, lo)

    dt = pd.to_datetime(
        df["pickup_datetime"].str.replace(" UTC", "", regex=False),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    df["year"] = dt.dt.year.fillna(2012).astype("int16")
    df["hour"] = dt.dt.hour.fillna(12).astype("int8")
    df["wday"] = dt.dt.dayofweek.fillna(0).astype("int8")
    df["month"] = dt.dt.month.fillna(1).astype("int8")
    df["dist_year"] = df["hav"] * df["year"]
    return df


FEATS = [
    "hav", "man", "brg", "dlat_abs", "dlon_abs",
    "p_jfk", "d_jfk", "p_lga", "d_lga", "p_ewr", "d_ewr", "p_ctr", "d_ctr",
    "year", "hour", "wday", "month", "dist_year", "passenger_count",
    "plat", "plon", "dlat", "dlon",
]
```

### Chunked Training Read And Cleaning

```python
def read_train():
    usecols = [
        "fare_amount", "pickup_datetime", "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude", "passenger_count",
    ]
    dtype = {
        "fare_amount": "float32",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "float32",
    }

    keep, dirty_keep = [], []
    seen = 0

    for ch in pd.read_csv(TRAIN, usecols=usecols, dtype=dtype, chunksize=2_000_000):
        ch = ch.rename(columns=RENAME)

        fare_ok = ch.fare_amount.between(2.5, 500) & ch.passenger_count.between(1, 6)
        coord_ok = valid_coords(ch)

        dirty = ch[fare_ok & ~coord_ok]
        if len(dirty):
            dirty_keep.append(dirty)
            if sum(len(x) for x in dirty_keep) > N_DIRTY_HOLDOUT * 2:
                big_dirty = pd.concat(dirty_keep, ignore_index=True)
                big_dirty = big_dirty.sample(
                    n=min(N_DIRTY_HOLDOUT, len(big_dirty)), random_state=SEED
                )
                dirty_keep = [big_dirty]

        dist = haversine(ch.plat, ch.plon, ch.dlat, ch.dlon)
        fare_per_km = ch.fare_amount / np.maximum(dist, 0.25)
        sane_trip = (
            dist.between(0.01, 120.0)
            & ~((dist < 0.10) & (ch.fare_amount > 20.0))
            & fare_per_km.between(0.35, 80.0)
        )

        ch = ch[fare_ok & coord_ok & sane_trip]
        seen += len(ch)
        keep.append(ch)

        if sum(len(x) for x in keep) > N_SAMPLE * 1.5:
            big = pd.concat(keep, ignore_index=True)
            big = big.sample(n=min(N_SAMPLE, len(big)), random_state=SEED)
            keep = [big]

        gc.collect()

    df = pd.concat(keep, ignore_index=True)
    if len(df) > N_SAMPLE:
        df = df.sample(n=N_SAMPLE, random_state=SEED).reset_index(drop=True)

    dirty_df = pd.concat(dirty_keep, ignore_index=True) if dirty_keep else pd.DataFrame()
    if len(dirty_df) > N_DIRTY_HOLDOUT:
        dirty_df = dirty_df.sample(n=N_DIRTY_HOLDOUT, random_state=SEED).reset_index(drop=True)

    print(f"clean rows seen={seen:,}; sampled={len(df):,}; dirty holdout={len(dirty_df):,}")
    return df, dirty_df
```

### Holdout Split And LightGBM Training

```python
tr, dirty_df = read_train()
global_median = float(tr["fare_amount"].median())

tr = add_features(tr)
y = np.log1p(tr["fare_amount"].values)
X = tr[FEATS].astype("float32")
del tr
gc.collect()

n = len(X)
idx = np.random.RandomState(SEED).permutation(n)
cut = int(n * 0.9)
tri, vai = idx[:cut], idx[cut:]

dtr = lgb.Dataset(X.iloc[tri], y[tri])
dva = lgb.Dataset(X.iloc[vai], y[vai])

params = dict(
    objective="regression",
    metric="rmse",
    num_leaves=100,
    learning_rate=0.05,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_data_in_leaf=200,
    max_bin=255,
    num_threads=NUM_THREADS,
    seed=SEED,
    verbose=-1,
)

model = lgb.train(
    params,
    dtr,
    num_boost_round=3000,
    valid_sets=[dva],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
)

valid_pred = np.expm1(model.predict(X.iloc[vai]))
valid_true = np.expm1(y[vai])
holdout_rmse = float(np.sqrt(np.mean((valid_pred - valid_true) ** 2)))
print(f"holdout RMSE original scale = {holdout_rmse:.5f}")
```

### Dirty-Coordinate Fallback Selection

```python
def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def dirty_fallback_report(model, dirty_df, global_median):
    if dirty_df is None or len(dirty_df) == 0:
        return "global_median", global_median

    d = dirty_df.copy()
    y = d["fare_amount"].values.astype("float32")
    dt = pd.to_datetime(
        d["pickup_datetime"].str.replace(" UTC", "", regex=False),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    d["_year"] = dt.dt.year.fillna(2012).astype("int16")

    results = {}

    pred_global = np.full(len(d), global_median, dtype="float32")
    results["global_median"] = rmse(pred_global, y)

    year_medians = d.groupby("_year")["fare_amount"].median()
    overall = float(np.median(y))
    pred_year = d["_year"].map(year_medians).fillna(overall).values.astype("float32")
    results["dirty_year_median"] = rmse(pred_year, y)

    pred_swap = np.full(len(d), global_median, dtype="float32")
    sw = swapped_valid_coords(d).values
    if sw.any():
        fixed = swap_lon_lat(d.loc[sw].drop(columns=["_year"]))
        fixed = add_features(fixed)
        pred_swap[sw] = np.expm1(model.predict(fixed[FEATS].astype("float32")))
    results["swap_if_valid_else_global"] = rmse(np.clip(pred_swap, 2.5, 250.0), y)

    best = min(results, key=results.get)
    print("dirty fallback RMSE:", results, "best:", best)

    if best == "dirty_year_median":
        return best, year_medians.to_dict()
    return best, global_median
```

### Test Prediction And Critical Post-Processing

```python
fallback_kind, fallback_value = dirty_fallback_report(model, dirty_df, global_median)

te = pd.read_csv(TEST).rename(columns=RENAME)
te = add_features(te)

pred = np.expm1(model.predict(te[FEATS].astype("float32")))

bad = ~valid_coords(te)
n_bad = int(bad.sum())

if n_bad:
    bad_values = bad.values

    if fallback_kind == "swap_if_valid_else_global":
        sw = (bad & swapped_valid_coords(te)).values
        pred[bad_values] = float(fallback_value)

        if sw.any():
            fixed = swap_lon_lat(te.loc[sw])
            fixed = add_features(fixed)
            pred[sw] = np.expm1(model.predict(fixed[FEATS].astype("float32")))

    elif fallback_kind == "dirty_year_median":
        years = te.loc[bad_values, "year"].astype(int)
        med_map = {int(k): float(v) for k, v in fallback_value.items()}
        pred[bad_values] = np.array(
            [med_map.get(int(y), global_median) for y in years], dtype="float32"
        )

    else:
        pred[bad_values] = float(fallback_value)

pred = np.clip(pred, 2.5, 250.0)

sub = pd.read_csv(SAMPLE)
sub[sub.columns[1]] = pred
sub.to_csv("submission.csv", index=False)
print({"invalid_test_rows": n_bad, "range": [float(pred.min()), float(pred.max())]})
```

## Optional Ensembling / Patch Ideas

Only try these after producing a clean LightGBM submission and confirming invalid-row fallback is active.

A historical best-effort script blended two old prediction files as:

```python
pred = 0.90 * direct_lgbm_test_pred + 0.10 * clean_lgbm_test_pred
```

Then it applied short-trip fare floors and category/year medians for dirty test rows. This reached the trajectory best score, but it depended on old artifacts and was not a clean reproducible solution.

If building a clean version of that idea, generate both components fresh:

- `direct`: train with lighter cleaning that keeps more normal coordinate-valid rows.
- `clean`: train with stricter distance/fare sanity filters.
- Blend around `0.85-0.95` direct and `0.05-0.15` clean.
- Apply short-trip floors only after checking local holdout effects.

Dirty category fallback from the old script may be worth reusing conceptually:

```python
def dirty_categories(df):
    p0 = (df.pickup_longitude.abs() < 1e-9) & (df.pickup_latitude.abs() < 1e-9)
    d0 = (df.dropoff_longitude.abs() < 1e-9) & (df.dropoff_latitude.abs() < 1e-9)

    pvalid = df.pickup_longitude.between(LON_MIN, LON_MAX) & df.pickup_latitude.between(LAT_MIN, LAT_MAX)
    dvalid = df.dropoff_longitude.between(LON_MIN, LON_MAX) & df.dropoff_latitude.between(LAT_MIN, LAT_MAX)

    out = np.full(len(df), "valid", dtype=object)
    out[p0 & d0] = "both_zero"
    out[p0 & ~d0] = "pickup_zero"
    out[~p0 & d0] = "dropoff_zero"
    out[(out == "valid") & (~pvalid) & dvalid] = "pickup_bad"
    out[(out == "valid") & pvalid & (~dvalid)] = "dropoff_bad"
    out[(out == "valid") & (~pvalid) & (~dvalid)] = "both_bad_nonzero"
    return out
```

Derive category/year medians from the current training dirty holdout, not from hard-coded historical values, unless the competition rules and workspace explicitly allow reusing prior artifacts.

## Runtime Strategy

The main failure pattern was never getting a fresh strong model to finish. Use staged runs:

1. Smoke run:
   - `N_SAMPLE = 200_000`
   - `num_boost_round = 300`
   - Verify columns, features, invalid test count, prediction range, and submission shape.

2. Medium run:
   - `N_SAMPLE = 2_000_000`
   - Keep full feature set and fallback.
   - Confirm holdout RMSE is moving toward the low 3s.

3. Full run:
   - `N_SAMPLE = 6_000_000`
   - Full `3000` rounds with early stopping.
   - Keep `NUM_THREADS <= 128`.
   - If wall clock is still a problem, cap rounds lower only after recording best iteration from the medium run.

Do not spend a full attempt waiting on a process that is unlikely to finish. Produce a fresh submission from the largest completed run, then iterate.

## Score Milestones (relative)

Use these as rough self-checks:

- Bad or naive geospatial model with weak cleaning: often in the 4.x RMSE range.
- Best submitted trajectory result: only around the competition median (artifact-dependent, not passing).
- A clean strong LightGBM with full features and invalid-test fallback should aim well below 3.0; the reference implementation comments expect roughly low-3s if reproduced correctly, and better with the full fallback working.

A submission above ~3.5 usually means invalid test fallback failed, feature coverage was reduced, or training silently fell back to stale or undersized artifacts. Self-check against the `criterion.json` pass line, not a fixed threshold; what stays invariant is the ORDERING — invalid-coordinate routing is the largest lever, the airport/`year`/`bearing` feature spine is next, and clean fresh training beats reused artifacts.
