NOMAD2018 transparent conductors is a small-data two-target crystal/tabular regression task; the leaderboard metric is the mean RMSLE over `formation_energy_ev_natom` and `bandgap_energy_ev` (lower is better). The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

## Approach

Use a feature-heavy classical regression ensemble, not a neural model. The strongest configuration used:

1. Read `train.csv`, `test.csv`, `sample_submission.csv`, plus each sample's `geometry.xyz` under `train/<id>/geometry.xyz` or `test/<id>/geometry.xyz`.
2. Engineer tabular crystal/composition features:
   - Cell volume from lattice lengths and angles.
   - Atom density, lattice length aggregates, angle aggregates.
   - Approximate atom counts for Al/Ga/In/O.
   - Composition products and cation entropy.
   - Weighted atomic number and cation radius.
   - Treat `spacegroup` as categorical for LightGBM.
3. Parse `geometry.xyz` and create periodic-distance geometry features:
   - Per-element counts and coordinate summary statistics.
   - Pairwise periodic distances using minimum-image convention.
   - Pairwise Coulomb-like values `Z_i * Z_j / distance`.
   - Pair-specific distance summaries and histogram bins for key pairs: `Al-O`, `Ga-O`, `In-O`, `O-O`, `Al-Ga`, `Al-In`, `Ga-In`.
   - Nearest-neighbor distance summaries by center element.
4. Train on `log1p(target)` because the metric is RMSLE. Predict in log-space and convert with `expm1`.
5. Use 5-fold shuffled `KFold`, one model per target:
   - LightGBM: `learning_rate=0.04`, `n_estimators=900`, `num_leaves=23` for formation energy and `31` for bandgap, `min_child_samples=10`, `subsample=0.85`, `colsample_bytree=0.8`, `reg_alpha=0.02`, `reg_lambda=0.2`, early stopping `80`.
   - ExtraTrees: `n_estimators=260`, `max_features=0.55`, `min_samples_leaf=1`, `bootstrap=False`.
   - RandomForest: `n_estimators=180`, `max_features=0.65`, `min_samples_leaf=1`, `bootstrap=True`.
6. Optimize a simple convex blend on OOF log predictions:
   - Grid LightGBM weight from `0.55` to `0.90`.
   - Grid ExtraTrees weight from `0.0` to `0.35`.
   - RandomForest weight is `1 - wl - we`.
   - Select the OOF blend with lowest mean RMSLE after `expm1`.
7. Post-process final predictions:
   - `formation_energy_ev_natom`: clip to `[0, 1.0]`.
   - `bandgap_energy_ev`: clip to `[0, 7.0]`.
   - Preserve exact sample submission column order and `id` order.

## What Actually Moved The Metric

The biggest lift was using `geometry.xyz`, not just `train.csv`/`test.csv`. The public CSV has only coarse lattice/composition descriptors; extracting periodic pairwise distances, Coulomb-like interactions, element-pair histograms, and nearest-neighbor summaries is the main reason a classical ensemble can be competitive.

The second critical move was training each target in `log1p` space and scoring OOF with the actual RMSLE transformation. Raw-target RMSE optimization is misaligned with the leaderboard and is the easiest way to waste the small dataset.

The third useful move was blending diverse tree models in log space. LightGBM is the anchor, but ExtraTrees and RandomForest add enough variance reduction to matter. The blend was selected by OOF RMSLE, not by fixed intuition.

No detailed failed-run history was preserved beyond the strongest trajectory, so do not overinterpret exact ablations. The lessons from the final code are clear:
- Do not ignore `geometry.xyz`; the tabular CSV alone is unlikely to be competitive.
- Do not optimize raw RMSE or blend after `expm1`; keep CV and blending in log space, then score through RMSLE.
- Do not submit negative predictions; RMSLE requires non-negative values and clipping is part of the winning post-processing.
- Do not treat this as a large-data deep-learning problem. With only about 2400 training rows and about 600 test rows, robust handcrafted descriptors plus tree ensembles are the pragmatic path.
- The single most damaging avoidable step is using raw targets/raw RMSE as the training and model-selection objective, because it directly mismatches the leaderboard metric.

## Key Code Snippets

### Metric And Target Transform

Use the exact leaderboard-style mean RMSLE across the two targets. Train models on `np.log1p(y)`, but score by converting predictions back with `np.expm1`.

```python
import math
import numpy as np
from sklearn.metrics import mean_squared_error

TARGETS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

def rmsle_score(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    vals = []
    for i in range(y_true.shape[1]):
        vals.append(math.sqrt(
            mean_squared_error(np.log1p(y_true[:, i]), np.log1p(y_pred[:, i]))
        ))
    return float(np.mean(vals)), vals

y = train[TARGETS].to_numpy()
y_log = np.log1p(y)
```

### Tabular Crystal And Composition Features

The `percent_atom_o` column is not present in the same form as the cations in this dataset; the winning code used a fixed `1.5` value consistent with the competition representation and built engineered features around the cation percentages.

```python
import math
import numpy as np

def cell_volume(a, b, c, alpha, beta, gamma):
    ar, br, gr = np.deg2rad([alpha, beta, gamma])
    inner = (
        1
        + 2 * np.cos(ar) * np.cos(br) * np.cos(gr)
        - np.cos(ar) ** 2
        - np.cos(br) ** 2
        - np.cos(gr) ** 2
    )
    return a * b * c * math.sqrt(max(inner, 1e-12))

def add_tabular_features(df):
    out = df.copy()

    a = out["lattice_vector_1_ang"].to_numpy()
    b = out["lattice_vector_2_ang"].to_numpy()
    c = out["lattice_vector_3_ang"].to_numpy()
    al = out["lattice_angle_alpha_degree"].to_numpy()
    be = out["lattice_angle_beta_degree"].to_numpy()
    ga = out["lattice_angle_gamma_degree"].to_numpy()

    out["percent_atom_o"] = 1.5
    out["cell_volume"] = [cell_volume(*vals) for vals in zip(a, b, c, al, be, ga)]
    out["density_atoms_per_vol"] = out["number_of_total_atoms"] / out["cell_volume"]

    out["abc_sum"] = a + b + c
    out["abc_mean"] = (a + b + c) / 3.0
    out["abc_std"] = np.std(np.vstack([a, b, c]), axis=0)
    out["abc_min"] = np.min(np.vstack([a, b, c]), axis=0)
    out["abc_max"] = np.max(np.vstack([a, b, c]), axis=0)
    out["abc_range"] = out["abc_max"] - out["abc_min"]
    out["abc_prod"] = a * b * c

    out["angle_sum"] = al + be + ga
    out["angle_mean"] = (al + be + ga) / 3.0
    out["angle_std"] = np.std(np.vstack([al, be, ga]), axis=0)
    out["angle_gamma_120_abs"] = np.abs(ga - 120.0)
    out["angle_orth_abs"] = np.abs(al - 90.0) + np.abs(be - 90.0) + np.abs(ga - 90.0)

    for x in ["al", "ga", "in"]:
        out[f"n_{x}"] = out[f"percent_atom_{x}"] * out["number_of_total_atoms"] * 0.4
    out["n_o"] = out["number_of_total_atoms"] * 0.6

    comps = ["percent_atom_al", "percent_atom_ga", "percent_atom_in"]
    for i, c1 in enumerate(comps):
        for c2 in comps[i:]:
            out[f"{c1}_x_{c2}"] = out[c1] * out[c2]

    out["al_ga_in_entropy"] = -sum(
        out[c].clip(1e-9, 1).mul(np.log(out[c].clip(1e-9, 1))) for c in comps
    )
    out["weighted_atomic_num"] = (
        13 * out["percent_atom_al"]
        + 31 * out["percent_atom_ga"]
        + 49 * out["percent_atom_in"]
        + 8 * 1.5
    )
    out["weighted_cation_radius"] = (
        0.535 * out["percent_atom_al"]
        + 0.62 * out["percent_atom_ga"]
        + 0.80 * out["percent_atom_in"]
    )

    out["spacegroup"] = out["spacegroup"].astype("category")
    out["atoms_x_spacegroup"] = (
        out["number_of_total_atoms"].astype(float) * out["spacegroup"].astype(int)
    )
    return out
```

### Geometry Parsing And Summary Blocks

Every `geometry.xyz` file contains `lattice_vector` rows and `atom x y z element` rows. Parse these files and convert variable-length structures into fixed-length descriptors.

```python
import numpy as np

ELEMENTS = ["Al", "Ga", "In", "O"]
PAIR_KEYS = [
    ("Al", "O"), ("Ga", "O"), ("In", "O"), ("O", "O"),
    ("Al", "Ga"), ("Al", "In"), ("Ga", "In"),
]

def parse_geometry(path):
    lattice = []
    elems = []
    coords = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "lattice_vector":
                lattice.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "atom":
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elems.append(parts[4])

    return (
        np.asarray(lattice, dtype=float),
        np.asarray(coords, dtype=float),
        np.asarray(elems),
    )

def stat_block(prefix, vals, feat):
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        for s in ["count", "mean", "std", "min", "p10", "p25", "p50", "p75", "p90", "max"]:
            feat[f"{prefix}_{s}"] = 0.0
        return

    feat[f"{prefix}_count"] = float(vals.size)
    feat[f"{prefix}_mean"] = float(np.mean(vals))
    feat[f"{prefix}_std"] = float(np.std(vals))
    feat[f"{prefix}_min"] = float(np.min(vals))
    feat[f"{prefix}_p10"] = float(np.percentile(vals, 10))
    feat[f"{prefix}_p25"] = float(np.percentile(vals, 25))
    feat[f"{prefix}_p50"] = float(np.percentile(vals, 50))
    feat[f"{prefix}_p75"] = float(np.percentile(vals, 75))
    feat[f"{prefix}_p90"] = float(np.percentile(vals, 90))
    feat[f"{prefix}_max"] = float(np.max(vals))
```

### Periodic Pairwise Geometry Features

The important detail is the minimum-image convention: convert Cartesian coordinates to fractional coordinates, subtract pairwise fractional positions, round to the nearest periodic image, then convert back to Cartesian distances.

```python
def geometry_features(data_dir, split, sample_id):
    lattice, coords, elems = parse_geometry(
        data_dir / split / str(int(sample_id)) / "geometry.xyz"
    )
    feat = {}

    for e in ELEMENTS:
        mask = elems == e
        feat[f"geom_count_{e}"] = float(mask.sum())
        if mask.any():
            stat_block(f"geom_{e}_x", coords[mask, 0], feat)
            stat_block(f"geom_{e}_y", coords[mask, 1], feat)
            stat_block(f"geom_{e}_z", coords[mask, 2], feat)
        else:
            stat_block(f"geom_{e}_x", [], feat)
            stat_block(f"geom_{e}_y", [], feat)
            stat_block(f"geom_{e}_z", [], feat)

    inv_lattice = np.linalg.inv(lattice)
    frac = coords @ inv_lattice

    delta_frac = frac[:, None, :] - frac[None, :, :]
    delta_frac -= np.round(delta_frac)
    delta = delta_frac @ lattice
    dist_mat = np.sqrt(np.sum(delta * delta, axis=2))

    n = len(coords)
    iu = np.triu_indices(n, k=1)
    all_d = dist_mat[iu]

    zmap = {"Al": 13.0, "Ga": 31.0, "In": 49.0, "O": 8.0}
    z = np.array([zmap[x] for x in elems], dtype=float)
    coulomb = (z[:, None] * z[None, :] / np.clip(dist_mat, 1e-6, None))[iu]

    stat_block("geom_pair_all_dist", all_d, feat)
    stat_block("geom_pair_coulomb", coulomb, feat)

    for pair in PAIR_KEYS:
        key = f"{pair[0]}_{pair[1]}"

        if pair[0] == pair[1]:
            idx = np.where(elems == pair[0])[0]
            if len(idx) > 1:
                sub = dist_mat[np.ix_(idx, idx)]
                vals = sub[np.triu_indices(len(idx), k=1)]
            else:
                vals = np.array([], dtype=float)
        else:
            idx1 = np.where(elems == pair[0])[0]
            idx2 = np.where(elems == pair[1])[0]
            vals = dist_mat[np.ix_(idx1, idx2)].ravel() if len(idx1) and len(idx2) else np.array([], dtype=float)

        stat_block(f"geom_pair_{key}_dist", vals, feat)

        hist, _ = np.histogram(vals, bins=[0, 1.6, 2.0, 2.4, 2.8, 3.2, 4.0, 5.5, 8.0, 20.0])
        denom = max(vals.size, 1)
        for k, h in enumerate(hist):
            feat[f"geom_pair_{key}_hist_{k}"] = float(h) / denom

    for center in ELEMENTS:
        mask = elems == center
        nn = []
        if mask.any() and len(coords) > 1:
            d = dist_mat.copy()
            np.fill_diagonal(d, np.inf)
            nn = np.min(d[mask], axis=1)
        stat_block(f"geom_nn_{center}", nn, feat)

    return feat
```

### Feature Matrix Assembly

Keep train/test columns identical after concatenating tabular and geometry features. Replace infinities and missing values with zero before fitting.

```python
import pandas as pd
import numpy as np

def make_features(data_dir, train, test):
    base_train = add_tabular_features(train.drop(columns=TARGETS))
    base_test = add_tabular_features(test)

    geom_train = pd.DataFrame([
        geometry_features(data_dir, "train", sample_id) for sample_id in train["id"]
    ])
    geom_test = pd.DataFrame([
        geometry_features(data_dir, "test", sample_id) for sample_id in test["id"]
    ])

    x_train = pd.concat([base_train.reset_index(drop=True), geom_train], axis=1)
    x_test = pd.concat([base_test.reset_index(drop=True), geom_test], axis=1)

    x_train = x_train.drop(columns=["id"])
    x_test = x_test.drop(columns=["id"])
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    x_train = x_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    x_test = x_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    return x_train, x_test
```

### LightGBM 5-Fold Model

Use `spacegroup` as a categorical feature for LightGBM. Keep the target-specific `num_leaves`.

```python
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold

def fit_lgbm(x_train, y_log, x_test, target, seed):
    params = dict(
        objective="regression",
        learning_rate=0.04,
        n_estimators=900,
        num_leaves=23 if target == "formation_energy_ev_natom" else 31,
        min_child_samples=10,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.02,
        reg_lambda=0.2,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(x_train))
    pred = np.zeros(len(x_test))

    for tr_idx, va_idx in kf.split(x_train):
        model = lgb.LGBMRegressor(**params)
        model.fit(
            x_train.iloc[tr_idx],
            y_log[tr_idx],
            eval_set=[(x_train.iloc[va_idx], y_log[va_idx])],
            eval_metric="rmse",
            categorical_feature=["spacegroup"],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        oof[va_idx] = model.predict(
            x_train.iloc[va_idx], num_iteration=model.best_iteration_
        )
        pred += model.predict(x_test, num_iteration=model.best_iteration_) / kf.n_splits

    return oof, pred
```

### ExtraTrees And RandomForest 5-Fold Models

One-hot encode `spacegroup` for sklearn tree models.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def fit_tree_model(model_cls, x_train, y_log, x_test, seed, **kwargs):
    x_num = pd.get_dummies(x_train, columns=["spacegroup"], dtype=float)
    xt_num = pd.get_dummies(x_test, columns=["spacegroup"], dtype=float)
    xt_num = xt_num.reindex(columns=x_num.columns, fill_value=0)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(x_num))
    pred = np.zeros(len(xt_num))

    for tr_idx, va_idx in kf.split(x_num):
        model = model_cls(random_state=seed, n_jobs=4, **kwargs)
        model.fit(x_num.iloc[tr_idx], y_log[tr_idx])
        oof[va_idx] = model.predict(x_num.iloc[va_idx])
        pred += model.predict(xt_num) / kf.n_splits

    return oof, pred
```

Use these exact model settings:

```python
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

et_oof, et_pred = fit_tree_model(
    ExtraTreesRegressor,
    x_train,
    y_log_target,
    x_test,
    seed=777 + target_index,
    n_estimators=260,
    max_features=0.55,
    min_samples_leaf=1,
    bootstrap=False,
)

rf_oof, rf_pred = fit_tree_model(
    RandomForestRegressor,
    x_train,
    y_log_target,
    x_test,
    seed=909 + target_index,
    n_estimators=180,
    max_features=0.65,
    min_samples_leaf=1,
    bootstrap=True,
)
```

### Per-Target Training Loop

Store OOF and test predictions in log space for each model family.

```python
all_oof = {
    "lgbm": np.zeros_like(y_log),
    "et": np.zeros_like(y_log),
    "rf": np.zeros_like(y_log),
}
all_test = {
    "lgbm": np.zeros((len(test), 2)),
    "et": np.zeros((len(test), 2)),
    "rf": np.zeros((len(test), 2)),
}

for ti, target in enumerate(TARGETS):
    lgb_oofs, lgb_preds = [], []

    for seed in [2024]:
        oof, pred = fit_lgbm(x_train, y_log[:, ti], x_test, target, seed)
        lgb_oofs.append(oof)
        lgb_preds.append(pred)

    all_oof["lgbm"][:, ti] = np.mean(lgb_oofs, axis=0)
    all_test["lgbm"][:, ti] = np.mean(lgb_preds, axis=0)

    all_oof["et"][:, ti], all_test["et"][:, ti] = fit_tree_model(
        ExtraTreesRegressor,
        x_train,
        y_log[:, ti],
        x_test,
        777 + ti,
        n_estimators=260,
        max_features=0.55,
        min_samples_leaf=1,
        bootstrap=False,
    )

    all_oof["rf"][:, ti], all_test["rf"][:, ti] = fit_tree_model(
        RandomForestRegressor,
        x_train,
        y_log[:, ti],
        x_test,
        909 + ti,
        n_estimators=180,
        max_features=0.65,
        min_samples_leaf=1,
        bootstrap=True,
    )
```

### OOF Blend Search And Submission Post-Processing

Blend in log space, score after `expm1`, then use the same weights for test predictions.

```python
best = None

for wl in np.linspace(0.55, 0.9, 8):
    for we in np.linspace(0.0, 0.35, 8):
        wr = 1.0 - wl - we
        if wr < 0:
            continue

        oof_log = (
            wl * all_oof["lgbm"]
            + we * all_oof["et"]
            + wr * all_oof["rf"]
        )
        score, parts = rmsle_score(y, np.expm1(oof_log))
        cand = (score, wl, we, wr, parts)

        if best is None or cand < best:
            best = cand

score, wl, we, wr, parts = best

test_log = (
    wl * all_test["lgbm"]
    + we * all_test["et"]
    + wr * all_test["rf"]
)
pred = np.expm1(test_log)

pred[:, 0] = np.clip(pred[:, 0], 0, 1.0)
pred[:, 1] = np.clip(pred[:, 1], 0, 7.0)

sub = sample.copy()
sub[TARGETS] = pred

assert list(sub.columns) == list(sample.columns)
assert len(sub) == len(sample)
assert sub["id"].equals(sample["id"])

sub.to_csv("submission.csv", index=False)
```

## Score Milestones

Naive or weak baseline: not recorded in the trajectory. Expect CSV-only or raw-target tree models to sit in the weaker part of the field unless heavily tuned.

Practical self-checks:
- If OOF mean RMSLE is worse than about `0.065`, geometry features, target transform, or validation scoring are likely wrong.
- A strong tabular-plus-geometry LightGBM should approach competitive territory.
- The strongest configuration is the log-space 5-fold LightGBM + ExtraTrees + RandomForest blend with periodic geometry features and clipped predictions.

The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score. What stays invariant across data/task variants is the ORDERING above: geometry features are the biggest lever, log-space training/scoring is essential, and log-space tree blending adds the final variance reduction.
