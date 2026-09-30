Task: binary text + Reddit metadata classification for Kaggle `random-acts-of-pizza`; metric: ROC AUC (higher is better). The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

**Approach**

Build a compact, classical NLP ensemble. The successful solution did not rely on transformers; it used TF-IDF / count n-grams, Naive Bayes SVM-style log-count ratios, handcrafted Reddit/user/request metadata, 5-fold out-of-fold validation, and a small blend search optimized directly for OOF AUC.

1. Load `train.json`, `test.json`, and `sampleSubmission.csv`.
2. Target is `requester_received_pizza`.
3. Construct one text field from:
   - `request_title`
   - `request_text_edit_aware` if present/nonempty, otherwise `request_text`
   - joined `requester_subreddits_at_request`
4. Build handcrafted metadata:
   - raw Reddit/user numeric columns
   - `log1p(abs(x))` and zero indicators for non-timestamp numeric columns
   - request hour/day/month/weekend and cyclic hour/day features
   - text length, word count, punctuation counts, uppercase ratio
   - title `[request]` / `[offer]` indicators
   - keyword flags and counts for money, reciprocity, gratitude, hunger, hardship, student, URL/image terms
   - subreddit list length
   - ratio features such as upvote balance, posts/comments per day, RAOP post/comment fractions
5. Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)`.
6. Train these model families:
   - TF-IDF word + char + scaled metadata into balanced `LogisticRegression`.
   - NB-SVM-style word count model: binary word `CountVectorizer`, log-count ratio multiplication, balanced logistic regression.
   - NB-SVM-style char count model: binary char `CountVectorizer`, log-count ratio multiplication, balanced logistic regression.
   - Dense metadata model: word TF-IDF -> `TruncatedSVD(n_components=80)`, concatenate with metadata, scale, train `HistGradientBoostingClassifier`.
   - Dense metadata model: same dense matrix, train `ExtraTreesClassifier`.
7. For each family, tune `C` or use fixed tree hyperparameters by OOF AUC.
8. Average fold predictions with a full-data refit prediction using `0.5 * cv_average + 0.5 * full_fit`.
9. Search blend weights on OOF predictions with single-model, two-model, and random Dirichlet candidate weights. Use the best OOF-AUC weights for test prediction.
10. Map predictions back to `sampleSubmission.csv` by `request_id`, clip to `[1e-5, 1 - 1e-5]`, and write `submission.csv`.

**What Actually Moved The Metric**

The metric was moved by a robust classical NLP ensemble, not by a single metadata model.

The most important lift was combining word and character text models. Word TF-IDF captured explicit request language, while char n-grams helped with misspellings, informal Reddit phrasing, punctuation, and short expressions. The winning code used both standard TF-IDF logistic regression and NB-SVM-style count features, giving multiple useful views of the same request text.

The second important lift was adding task-specific metadata and request-style features. Reddit account age, activity counts, upvote/downvote aggregates, RAOP-specific history, request timing, text length, title tags, and keyword buckets all contributed signal beyond raw text. The metadata alone is not enough, but it improves the blend.

The third important lift was OOF-driven blending. The solution did not hand-pick a fixed average. It generated OOF predictions for every component, then searched weights by ROC AUC on the same CV folds. This was enough to cross the pass line.

**Pitfalls / Lessons**

Do not spend the first pass on large transformers if the goal is to reliably pass. Later attempts planned RoBERTa/DeBERTa fine-tuning to chase a higher score, but those runs failed for infrastructure reasons and did not produce a better submission. For a reliable pass, the classical ensemble is faster, cheaper, and easier to validate.

The damaging failure mode was relying on unavailable external/compute resources for transformer experiments before locking in a strong local baseline. The successful run was self-contained with `sklearn`, `scipy`, `numpy`, and `pandas`.

Do not submit predictions in test-row order blindly. Use `sampleSubmission.csv`, map by `request_id`, and explicitly fail if any IDs are missing.

Do not optimize only individual model CV scores. Some weaker individual models still help the blend; keep their OOF/test predictions and let the weight search decide.

**Key Code Snippets**

Text construction:

```python
def as_text(s):
    return s.fillna("").astype(str)

def make_text(df):
    title = as_text(df.get("request_title", pd.Series("", index=df.index)))
    body = as_text(df.get("request_text_edit_aware", pd.Series("", index=df.index)))
    if body.str.len().sum() == 0:
        body = as_text(df.get("request_text", pd.Series("", index=df.index)))

    subreddits = df.get(
        "requester_subreddits_at_request",
        pd.Series("", index=df.index),
    ).apply(lambda x: " ".join(x) if isinstance(x, list) else "")

    return (title + " \n " + body + " \n subreddits " + subreddits).fillna("")
```

Core metadata features:

```python
numeric_cols = [
    "requester_account_age_in_days_at_request",
    "requester_days_since_first_post_on_raop_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_upvotes_plus_downvotes_at_request",
    "unix_timestamp_of_request",
    "unix_timestamp_of_request_utc",
]

def numeric_features(df, text):
    feats = pd.DataFrame(index=df.index)

    for col in numeric_cols:
        vals = pd.to_numeric(df.get(col, pd.Series(0, index=df.index)),
                             errors="coerce").fillna(0.0)
        feats[col] = vals
        if col not in ("unix_timestamp_of_request", "unix_timestamp_of_request_utc"):
            feats[f"log1p_abs_{col}"] = np.log1p(np.abs(vals))
            feats[f"is_zero_{col}"] = (vals == 0).astype(float)

    dt = pd.to_datetime(
        pd.to_numeric(
            df.get("unix_timestamp_of_request_utc", pd.Series(0, index=df.index)),
            errors="coerce",
        ).fillna(0),
        unit="s",
    )
    feats["hour"] = dt.dt.hour
    feats["dow"] = dt.dt.dayofweek
    feats["month"] = dt.dt.month
    feats["is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)
    feats["hour_sin"] = np.sin(2 * np.pi * feats["hour"] / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * feats["hour"] / 24)
    feats["dow_sin"] = np.sin(2 * np.pi * feats["dow"] / 7)
    feats["dow_cos"] = np.cos(2 * np.pi * feats["dow"] / 7)

    lower = text.str.lower()
    word_count = lower.str.split().apply(len)
    char_count = lower.str.len()
    title = as_text(df.get("request_title", pd.Series("", index=df.index)))

    feats["text_chars"] = char_count
    feats["text_words"] = word_count
    feats["title_chars"] = title.str.len()
    feats["title_words"] = title.str.split().apply(len)
    feats["avg_word_len"] = char_count / np.maximum(word_count, 1)
    feats["exclaim_count"] = text.str.count("!")
    feats["question_count"] = text.str.count(r"\?")
    feats["uppercase_ratio"] = text.apply(
        lambda s: sum(c.isupper() for c in s) / max(len(s), 1)
    )
    feats["has_request_tag"] = title.str.contains(
        r"\[request\]|\(request\)", flags=re.I, regex=True
    ).astype(float)
    feats["has_offer_tag"] = title.str.contains(
        r"\[offer\]|\(offer\)", flags=re.I, regex=True
    ).astype(float)

    return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
```

Keyword indicators used by the solution:

```python
PATTERNS = {
    "money": re.compile(r"\b(broke|money|cash|paid|payday|rent|bill|bills|dollar|bank|unemployed|jobless|hours cut)\b", re.I),
    "reciprocity": re.compile(r"\b(pay ?it ?forward|return the favor|repay|pay back|trade|exchange|venmo|paypal|draw|poem|song)\b", re.I),
    "gratitude": re.compile(r"\b(thanks|thank you|appreciate|grateful|bless|kindness)\b", re.I),
    "hunger": re.compile(r"\b(hungry|starving|food|eat|meal|dinner|lunch|breakfast|fridge|pantry)\b", re.I),
    "hardship": re.compile(r"\b(sick|ill|hospital|medicine|family|kid|kids|child|children|pregnant|homeless|car broke|stranded|emergency)\b", re.I),
    "student": re.compile(r"\b(student|college|university|school|finals|exam|dorm|campus)\b", re.I),
    "url": re.compile(r"https?://|www\.|imgur|\.jpg|\.png", re.I),
}

for name, pat in PATTERNS.items():
    feats[f"kw_{name}"] = lower.str.contains(pat).astype(float)
    feats[f"kw_count_{name}"] = lower.str.count(pat)
```

Ratio features:

```python
def add_ratio_features(meta):
    m = meta.copy()

    plus = m["requester_upvotes_plus_downvotes_at_request"].replace(0, np.nan)
    m["upvote_balance_ratio"] = (
        m["requester_upvotes_minus_downvotes_at_request"] / plus
    ).fillna(0.0)

    age = m["requester_account_age_in_days_at_request"] + 1.0
    m["comments_per_day"] = m["requester_number_of_comments_at_request"] / age
    m["posts_per_day"] = m["requester_number_of_posts_at_request"] / age

    m["raop_comment_frac"] = (
        m["requester_number_of_comments_in_raop_at_request"]
        / (m["requester_number_of_comments_at_request"] + 1.0)
    )
    m["raop_post_frac"] = (
        m["requester_number_of_posts_on_raop_at_request"]
        / (m["requester_number_of_posts_at_request"] + 1.0)
    )

    return m.replace([np.inf, -np.inf], np.nan).fillna(0.0)
```

CV setup:

```python
SEED = 477844

y = train["requester_received_pizza"].astype(int).to_numpy()

folds = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)
```

TF-IDF logistic model:

```python
word = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=65000,
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w\w+\b",
)

char = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=90000,
    sublinear_tf=True,
)

xw = word.fit_transform(train_text)
xtw = word.transform(test_text)
xc = char.fit_transform(train_text)
xtc = char.transform(test_text)

scaler = StandardScaler()
xm = sparse.csr_matrix(scaler.fit_transform(meta_train))
xtm = sparse.csr_matrix(scaler.transform(meta_test))

x = sparse.hstack([xw, xc, xm], format="csr")
xt = sparse.hstack([xtw, xtc, xtm], format="csr")

best = None
for c in [0.03, 0.06, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0]:
    oof = np.zeros(len(y))

    for tr, va in folds.split(x, y):
        clf = LogisticRegression(
            C=c,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        )
        clf.fit(x[tr], y[tr])
        oof[va] = clf.predict_proba(x[va])[:, 1]

    auc = roc_auc_score(y, oof)
    if best is None or auc > best[0]:
        best = (auc, c, oof)
```

NB-SVM-style text model:

```python
def nb_features(x, y):
    p = np.asarray(x[y == 1].sum(axis=0)).ravel() + 1.0
    q = np.asarray(x[y == 0].sum(axis=0)).ravel() + 1.0
    return np.log((p / p.sum()) / (q / q.sum()))

# Word version
vec = CountVectorizer(
    lowercase=True,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    max_features=120000,
    binary=True,
    token_pattern=r"(?u)\b\w\w+\b",
)

# Char version uses:
# analyzer="char_wb", ngram_range=(3, 6), min_df=2,
# max_features=150000, binary=True

x = vec.fit_transform(train_text)
xt = vec.transform(test_text)

for c in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
    oof = np.zeros(len(y))

    for tr, va in folds.split(x, y):
        r = nb_features(x[tr], y[tr])

        clf = LogisticRegression(
            C=c,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        )
        clf.fit(x[tr].multiply(r), y[tr])
        oof[va] = clf.predict_proba(x[va].multiply(r))[:, 1]

    auc = roc_auc_score(y, oof)
```

Dense metadata/SVD models:

```python
tfidf = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    max_features=25000,
    sublinear_tf=True,
)

x_text = tfidf.fit_transform(train_text)
xt_text = tfidf.transform(test_text)

svd = TruncatedSVD(n_components=80, random_state=SEED)

dense_train = np.hstack([meta_train.to_numpy(), svd.fit_transform(x_text)])
dense_test = np.hstack([meta_test.to_numpy(), svd.transform(xt_text)])

scaler = StandardScaler()
dense_train = scaler.fit_transform(dense_train)
dense_test = scaler.transform(dense_test)

models = [
    (
        "hgb",
        HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.035,
            l2_regularization=0.05,
            max_leaf_nodes=15,
            random_state=SEED,
        ),
    ),
    (
        "et",
        ExtraTreesClassifier(
            n_estimators=450,
            max_depth=6,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
    ),
]
```

Fold prediction plus full-data refit:

```python
pred = np.zeros(xt.shape[0])

for tr, va in folds.split(x, y):
    clf = LogisticRegression(
        C=best_c,
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
    )
    clf.fit(x[tr], y[tr])
    pred += clf.predict_proba(xt)[:, 1] / folds.n_splits

clf = LogisticRegression(
    C=best_c,
    solver="liblinear",
    class_weight="balanced",
    max_iter=2000,
    random_state=SEED,
)
clf.fit(x, y)
pred_full = clf.predict_proba(xt)[:, 1]

final_model_pred = 0.5 * pred + 0.5 * pred_full
```

OOF blend search:

```python
def search_blend(model_outputs, y):
    names = list(model_outputs)
    oofs = np.vstack([model_outputs[n][0] for n in names])
    preds = np.vstack([model_outputs[n][1] for n in names])

    best = (-1.0, None)
    candidates = []

    for i in range(len(names)):
        w = np.zeros(len(names))
        w[i] = 1.0
        candidates.append(w)

    for main in range(len(names)):
        for aux in range(len(names)):
            if main == aux:
                continue
            for a in [0.65, 0.75, 0.85, 0.9, 0.95]:
                w = np.zeros(len(names))
                w[main] = a
                w[aux] = 1.0 - a
                candidates.append(w)

    rng = np.random.default_rng(SEED)
    for alpha in [0.15, 0.4, 0.8, 1.5, 4.0]:
        candidates.extend(rng.dirichlet(np.full(len(names), alpha), size=4000))

    for w in candidates:
        auc = roc_auc_score(y, np.dot(w, oofs))
        if auc > best[0]:
            best = (auc, w)

    auc, w = best
    return auc, np.dot(w, oofs), np.dot(w, preds), dict(zip(names, w))
```

Submission mapping:

```python
out = sample.copy()
id_col, target_col = sample.columns.tolist()

pred_by_id = pd.Series(blend_pred, index=test["request_id"].astype(str))
out[target_col] = out[id_col].astype(str).map(pred_by_id).astype(float)

if out[target_col].isna().any():
    missing = out.loc[out[target_col].isna(), id_col].tolist()[:10]
    raise RuntimeError(f"missing predictions for ids: {missing}")

out[target_col] = out[target_col].clip(1e-5, 1 - 1e-5)
out.to_csv("submission.csv", index=False)
```

**Score Milestones (relative)**

Naive / weak baseline: around the competition median.

Metadata plus basic classical text models should be expected to move above median.

The pass-reaching configuration was the 5-fold OOF blend of:
`tfidf_logit`, `nbsvm_word`, `nbsvm_char`, dense SVD + histogram gradient boosting, and dense SVD + extra trees, with weights selected by OOF AUC.

Self-check against the `criterion.json` pass line, not a fixed threshold. What stays invariant is the ORDERING: word+char text views first, task-specific Reddit/request metadata next, and OOF-driven blend-weight search last — each adds signal, while a single metadata model or a fixed hand-picked average does not.
