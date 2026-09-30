# Text Normalization (English written→spoken); metric: token-level exact-match accuracy on `after` strings (higher is better). The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

## Approach

Use a deterministic token-level normalizer. The strong solution did **not** use seq2seq, neural modeling, or ensembling. It used a very strong memorization baseline plus a few high-precision numeric rules.

1. Read the prepared public files:

   - `en_train.csv.zip`
   - `en_test_2.csv.zip`
   - `en_sample_submission_2.csv.zip`

   The final code used:

   ```python
   DATA_DIR = "./input"
   TRAIN_PATH = os.path.join(DATA_DIR, "en_train.csv.zip")
   TEST_PATH = os.path.join(DATA_DIR, "en_test_2.csv.zip")
   SAMPLE_PATH = os.path.join(DATA_DIR, "en_sample_submission_2.csv.zip")
   ```

2. Build a token lookup table from full train:

   - Group by `(before, after)`.
   - For each `before`, choose the most frequent `after`.
   - At inference, predict `mapping[before]` when present.
   - If unseen, default to `before`.

3. Add only conservative numeric rules:

   - Integer/cardinal conversion for plain integers and comma-formatted integers.
   - ISO date conversion for tokens like `YYYY-MM-DD`.
   - Numeric ordinal conversion for tokens like `1st`, `22nd`, `103rd`.

   Apply rules only when the token contains a digit and the current prediction is still identity-like:

   - lookup missing, or
   - lookup prediction equals `before`.

   Do **not** apply rules to pure alphabetic tokens.

4. Use a simple validation split:

   - Sort by existing `sentence_id`.
   - Train on the first 90% of sentence ids.
   - Validate on the last 10% of sentence ids.

   ```python
   cut = int(int(train["sentence_id"].max()) * 0.9)
   tr = train[train["sentence_id"] <= cut]
   va = train[train["sentence_id"] > cut]
   ```

5. Rule acceptance criterion:

   - For non-integer extra rules, audit on holdout before trusting them.
   - Keep only rules with precision about `>= 0.99` and positive `delta_correct`.
   - The strong run kept ISO-date and numeric-ordinal rules because they were high precision on holdout.
   - Broader money/measure/decimal/percentage/email/letters rules looked tempting but were not used because holdout precision was only around `0.96-0.98` or risked damaging `PLAIN`.

6. Submission construction is critical:

   - Use exactly the sample submission columns.
   - Use exactly the sample submission row order.
   - Construct ids as `sentence_id + "_" + token_id`.
   - Validate that ids match `en_sample_submission_2.csv.zip` exactly before writing.

## What Actually Moved The Metric

The biggest issue was **not modeling**; it was submission validity. Several early attempts produced `score=None` because the submission was invalid. The most damaging failure mode was writing a file that looked plausible but did not exactly match the sample submission schema/order. Fixing this and validating locally was the breakthrough.

Metric-moving changes, in order of leverage:

1. **Submission validation unlocked scoring**

   Earlier attempts scored `None`. The first reliable scored baseline came only after enforcing:

   - columns exactly `["id", "after"]`,
   - row count equals sample,
   - id order equals sample,
   - no missing `after`.

   This produced the first valid, passing-level score.

2. **Full-train `before -> most frequent after` lookup was the core model**

   Most tokens are `PLAIN` or repeat seen normalizations. A highest-frequency dictionary over the entire training set is extremely strong for this competition. Unknown tokens defaulting to identity is also important; aggressive transformation hurts.

   This lookup baseline is the bulk of the score.

3. **Narrow integer rule added a small increment**

   Adding a conservative integer/cardinal rule for unseen or identity numeric tokens improved the score by a small margin over the lookup baseline.

4. **Strict ISO-date and numeric-ordinal rules added a further increment**

   Adding only the high-precision ISO date and numeric ordinal rules lifted the final score a little more. Each such rule was accepted only after passing the holdout precision gate.

Pitfalls and lessons:

- Do not train a seq2seq model first. It is unnecessary and creates many ways to underperform.
- Do not use class-aware lookup directly on test, because test has no `class` labels.
- Do not apply broad handwritten rules globally. Money, measure, decimal, percent, electronic, and letters rules can improve some examples but are not precise enough without class prediction.
- Do not alter sample submission order. This caused invalid submissions and wasted most failed attempts.
- Do not read or submit against the wrong test/sample pair. For this competition use `en_test_2.csv.zip` with `en_sample_submission_2.csv.zip`.

## Key Code Snippets

### Highest-frequency lookup

```python
def build_best_mapping(train: pd.DataFrame):
    counts = (
        train.groupby(["before", "after"], sort=False)
        .size()
        .rename("cnt")
        .reset_index()
    )
    idx = counts.groupby("before", sort=False)["cnt"].idxmax()
    best = counts.loc[idx].copy()
    return dict(zip(best["before"].astype(str), best["after"].astype(str)))
```

### Holdout split and local token accuracy

```python
def token_acc(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(
        (y_true.astype(str).to_numpy() == y_pred.astype(str).to_numpy()).mean()
    )


def evaluate_holdout(train: pd.DataFrame):
    cut = int(int(train["sentence_id"].max()) * 0.9)
    tr = train[train["sentence_id"] <= cut]
    va = train[train["sentence_id"] > cut]

    mapping = build_best_mapping(tr)

    base, _ = predict_series(va["before"], mapping, [])
    integer, _ = predict_series(
        va["before"],
        mapping,
        [("integer", rule_integer)],
    )
    conservative, _ = predict_series(
        va["before"],
        mapping,
        [
            ("integer", rule_integer),
            ("iso_date", rule_iso_date),
            ("ordinal", rule_ordinal),
        ],
    )

    return {
        "split": "last_10pct_sentence_id_holdout",
        "base_lookup_acc": token_acc(va["after"], base),
        "lookup_plus_integer_acc": token_acc(va["after"], integer),
        "lookup_plus_integer_iso_ordinal_acc": token_acc(
            va["after"], conservative
        ),
    }
```

### Number words

```python
SMALL = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]

TENS = [
    "", "", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety",
]

def int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + int_to_words(-n)
    if n < 20:
        return SMALL[n]
    if n < 100:
        q, r = divmod(n, 10)
        return TENS[q] if r == 0 else TENS[q] + " " + SMALL[r]
    if n < 1000:
        q, r = divmod(n, 100)
        return SMALL[q] + " hundred" if r == 0 else SMALL[q] + " hundred " + int_to_words(r)

    for scale, name in [
        (10**12, "trillion"),
        (10**9, "billion"),
        (10**6, "million"),
        (1000, "thousand"),
    ]:
        if n >= scale:
            q, r = divmod(n, scale)
            return (
                int_to_words(q) + " " + name
                if r == 0
                else int_to_words(q) + " " + name + " " + int_to_words(r)
            )

    raise ValueError(n)
```

### Integer rule

```python
INT_RE = re.compile(r"^[+]?\d{1,18}$")
COMMA_INT_RE = re.compile(r"^[+]?\d{1,3}(,\d{3})+$")

def rule_integer(token: str):
    if token.isalpha():
        return None

    raw = token.replace(",", "")

    if COMMA_INT_RE.match(token) or INT_RE.match(token):
        try:
            return int_to_words(int(raw.lstrip("+")))
        except Exception:
            return None

    return None
```

### Ordinal and year/date rules

```python
ORDINAL_SMALL = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth",
    10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth",
    14: "fourteenth", 15: "fifteenth", 16: "sixteenth",
    17: "seventeenth", 18: "eighteenth", 19: "nineteenth",
}

ORDINAL_TENS = {
    20: "twentieth", 30: "thirtieth", 40: "fortieth",
    50: "fiftieth", 60: "sixtieth", 70: "seventieth",
    80: "eightieth", 90: "ninetieth",
}

MONTHS = {
    "01": "january", "02": "february", "03": "march", "04": "april",
    "05": "may", "06": "june", "07": "july", "08": "august",
    "09": "september", "10": "october", "11": "november", "12": "december",
}

ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
ORDINAL_RE = re.compile(r"^(\d+)(st|nd|rd|th)$", re.I)


def ordinal_int(n: int) -> str:
    if n in ORDINAL_SMALL:
        return ORDINAL_SMALL[n]
    if n in ORDINAL_TENS:
        return ORDINAL_TENS[n]
    if n < 100:
        return int_to_words(n // 10 * 10) + " " + ordinal_int(n % 10)
    if n < 1000:
        q, r = divmod(n, 100)
        return (
            int_to_words(q) + " hundredth"
            if r == 0
            else int_to_words(q) + " hundred " + ordinal_int(r)
        )

    q, r = divmod(n, 1000)
    return (
        int_to_words(q) + " thousandth"
        if r == 0
        else int_to_words(q) + " thousand " + ordinal_int(r)
    )


def year_words(n: int) -> str:
    if n == 2000:
        return "two thousand"
    if 2001 <= n <= 2009:
        return "two thousand " + SMALL[n - 2000]
    if 2010 <= n <= 2099:
        return "twenty " + int_to_words(n - 2000)
    if 1000 <= n <= 1999:
        first, last = divmod(n, 100)
        if last == 0:
            return int_to_words(first) + " hundred"
        if last < 10:
            return int_to_words(first) + " o " + SMALL[last]
        return int_to_words(first) + " " + int_to_words(last)
    return int_to_words(n)


def rule_iso_date(token: str):
    m = ISO_DATE_RE.match(token)
    if not m:
        return None

    year, month, day = m.groups()
    d = int(day)

    if month not in MONTHS or not (1 <= d <= 31):
        return None

    return f"the {ordinal_int(d)} of {MONTHS[month]} {year_words(int(year))}"


def rule_ordinal(token: str):
    m = ORDINAL_RE.match(token)
    if not m:
        return None

    try:
        return ordinal_int(int(m.group(1)))
    except Exception:
        return None
```

### Prediction with conservative rule gate

```python
def predict_series(before: pd.Series, mapping: dict, enabled_rules):
    before_s = before.astype(str)

    pred = before_s.map(mapping)
    missing = pred.isna()
    pred = pred.fillna(before_s).astype(str)

    counts = {}

    for name, func in enabled_rules:
        gate = (
            (missing | (pred == before_s))
            & before_s.str.contains(r"[0-9]", regex=True)
        )

        if not gate.any():
            counts[name] = 0
            continue

        ruled = before_s[gate].map(func)
        hit = ruled.notna() & (ruled.astype(str) != pred[gate].astype(str))

        if hit.any():
            pred.loc[ruled.index[hit]] = ruled.loc[hit].astype(str)

        counts[name] = int(hit.sum())

    return pred, counts
```

Important details:

- The rule gate uses `(missing | (pred == before_s))`, not all rows.
- The digit check prevents accidental normalization of plain words.
- `pred` is always filled with strings before submission.
- The final configuration used:

```python
rules = [
    ("integer", rule_integer),
    ("iso_date", rule_iso_date),
    ("ordinal", rule_ordinal),
]
```

### Submission validation

```python
def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame):
    assert list(sub.columns) == list(sample.columns), (sub.columns, sample.columns)
    assert len(sub) == len(sample), (len(sub), len(sample))
    assert sub["id"].astype(str).equals(sample["id"].astype(str))
    assert int(sub["after"].isna().sum()) == 0
```

### Final submission skeleton

```python
train = pd.read_csv(
    TRAIN_PATH,
    dtype={"before": "string", "after": "string"},
    keep_default_na=False,
)
test = pd.read_csv(
    TEST_PATH,
    dtype={"before": "string"},
    keep_default_na=False,
)
sample = pd.read_csv(
    SAMPLE_PATH,
    dtype={"id": "string", "after": "string"},
    keep_default_na=False,
)

mapping = build_best_mapping(train)

rules = [
    ("integer", rule_integer),
    ("iso_date", rule_iso_date),
    ("ordinal", rule_ordinal),
]

pred, test_rule_counts = predict_series(test["before"], mapping, rules)

sub = pd.DataFrame(
    {
        "id": test["sentence_id"].astype(str) + "_" + test["token_id"].astype(str),
        "after": pred.astype(str),
    }
)

validate_submission(sub, sample)
sub.to_csv("submission.csv", index=False)
```

## Diagnostics To Write Out

Always write a small report so the next iteration can tell whether it is improving locally:

```python
report = {
    "method": (
        "full-train before->most-frequent-after lookup; "
        "unknown/identity numeric tokens get integer, ISO-date, and ordinal rules"
    ),
    "rows": int(len(sub)),
    "unique_ids": int(sub["id"].nunique()),
    "test_rule_counts": test_rule_counts,
    "test_changed_rate": float(
        (
            sub["after"].reset_index(drop=True)
            != test["before"].astype(str).reset_index(drop=True)
        ).mean()
    ),
    "holdout": holdout,
}
```

Useful sanity checks:

- `rows == len(sample)`.
- `unique_ids == rows`.
- `test_changed_rate` should be plausible, not near zero and not wildly high.
- `test_rule_counts` should be small and explainable.
- Holdout should show lookup is already very high and rules add only a small positive gain.

## Next Step For Further Gains

The likely path to higher accuracy is not more broad rules by hand. The next useful upgrade is a lightweight token class predictor trained only on train:

- Features: `before`, previous/next token, regex shape, capitalization, punctuation, digit pattern.
- Model: `LogisticRegression` or `LinearSVM`.
- Use predicted class to build/use `(before, predicted_class) -> after` mappings.
- Only then selectively enable riskier rules for classes like `MONEY`, `MEASURE`, `DECIMAL`, `DATE`, and `ELECTRONIC`.

Do not use predicted classes unless out-of-fold validation proves they improve over the plain `before -> after` lookup.

## Score Milestones (relative)

- Invalid submission: `score=None`; this was the most common early failure.
- Identity/default-heavy baseline: barely at the working boundary.
- Valid full-train `before -> most frequent after` lookup with identity fallback: the bulk of the score, enough to reach a passing level.
- Lookup plus narrow integer rule: a small increment above the lookup baseline.
- Lookup plus integer + strict ISO-date + numeric-ordinal rules: the top of the progression, each rule gated by holdout precision.

Self-check against the `criterion.json` pass line, not a fixed threshold. What stays invariant is the ORDERING: valid submission first, then the full-train most-frequent-`after` lookup as the dominant model, then only high-precision (holdout-audited) numeric rules layered on top — broad ungated rules regress.
