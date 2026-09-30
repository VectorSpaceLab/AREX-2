Russian written-to-spoken text normalization; evaluation is per-token exact-match accuracy on `after` (higher is better). The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

## Approach

Use a deterministic token-level normalizer, not a neural model. Most tokens are already normalized by an exact `before -> after` memorization table, and the score is dominated by per-token exact matches.

1. Read `ru_train.csv.zip`, `ru_test_2.csv.zip`, and `ru_sample_submission_2.csv.zip`.
2. Build a majority-vote lookup table from training data:
   - group by `(before, after)`
   - count occurrences
   - for each `before`, keep the most frequent `after`
3. Predict test tokens by lookup first.
4. For lookup misses only, apply small handwritten Russian normalization rules:
   - cardinals, including spaced thousands
   - dates and years
   - times
   - uppercase abbreviations / initials
   - simple Latin letter transliteration fallback
5. Align submission rows to the sample submission by `id = sentence_id + "_" + token_id`.
6. Do not train a neural model. There are no epochs, learning rate, batch size, or folds in the strongest solution.
7. Use a simple validation split for local checks:
   - validation: `sentence_id % 10 == 0`
   - train fold: all other rows
   - metric: `(pred == after).mean()`

## What Actually Moved The Metric

The biggest gain was the majority lookup table. This task has very high token repetition and a very large `self` class, so memorizing the most frequent `after` for every observed `before` is already a strong, passing-level solution on its own.

The useful extension was conservative fallback rules for lookup misses. They target exactly the high-error non-`self` classes that the lookup table cannot cover when a written token appears only in test or appears in a new format: numbers, dates, years, times, abbreviations, and Latin strings.

The key lesson is to apply rules only after lookup misses. The lookup table captures many Russian-specific, dataset-specific, and contextually common normalizations better than generic rules. Overriding known tokens with handcrafted logic risks damaging the dominant memorization signal.

Pitfalls:

- Do not spend early effort on a neural seq2seq model. The strong solution is deterministic and data-driven.
- Do not normalize every numeric-looking token before lookup. A token that appears in train should use the learned majority `after`.
- Do not ignore sample submission alignment. The test rows must be merged back to sample `id` order.
- The failed later attempts in the trajectory did not establish a better approach; they were interrupted before running. The only proven passing path is lookup plus conservative rule fallback.
- The most damaging likely mistake is broad, aggressive rule application to all tokens, because it can corrupt many tokens the lookup table would have predicted exactly.

## Essential Code Skeleton

### Majority Lookup

```python
def build_map(train):
    counts = train.groupby(["before", "after"], sort=False).size().rename("cnt").reset_index()
    idx = counts.groupby("before", sort=False)["cnt"].idxmax()
    return dict(zip(counts.loc[idx, "before"], counts.loc[idx, "after"]))
```

### Validation Split

```python
train = pd.read_csv(os.path.join(DATA_DIR, "ru_train.csv.zip"))

tr = train[train.sentence_id % 10 != 0]
va = train[train.sentence_id % 10 == 0].copy()

mapping = build_map(tr)
pred = predict_series(va.before, mapping, use_rules=True)

print((pred.values == va.after.values).mean())
```

### Russian Cardinal Numbers

```python
ONES = {
    0: "ноль", 1: "один", 2: "два", 3: "три", 4: "четыре",
    5: "пять", 6: "шесть", 7: "семь", 8: "восемь", 9: "девять",
}
ONES_F = {1: "одна", 2: "две"}

TEENS = {
    10: "десять", 11: "одиннадцать", 12: "двенадцать",
    13: "тринадцать", 14: "четырнадцать", 15: "пятнадцать",
    16: "шестнадцать", 17: "семнадцать", 18: "восемнадцать",
    19: "девятнадцать",
}

TENS = {
    20: "двадцать", 30: "тридцать", 40: "сорок", 50: "пятьдесят",
    60: "шестьдесят", 70: "семьдесят", 80: "восемьдесят", 90: "девяносто",
}

HUNDREDS = {
    100: "сто", 200: "двести", 300: "триста", 400: "четыреста",
    500: "пятьсот", 600: "шестьсот", 700: "семьсот",
    800: "восемьсот", 900: "девятьсот",
}

def plural_form(n, one, few, many):
    n_abs = abs(n) % 100
    if 11 <= n_abs <= 14:
        return many
    last = n_abs % 10
    if last == 1:
        return one
    if 2 <= last <= 4:
        return few
    return many

def triplet_to_words(n, feminine=False):
    parts = []
    h = n // 100 * 100
    if h:
        parts.append(HUNDREDS[h])

    r = n % 100
    if 10 <= r <= 19:
        parts.append(TEENS[r])
    else:
        t = r // 10 * 10
        u = r % 10
        if t:
            parts.append(TENS[t])
        if u:
            parts.append(ONES_F[u] if feminine and u in ONES_F else ONES[u])
    return parts

def int_to_cardinal(n):
    n = int(n)
    if n == 0:
        return ONES[0]
    if n < 0:
        return "минус " + int_to_cardinal(-n)

    groups = []
    while n:
        groups.append(n % 1000)
        n //= 1000

    names = [
        None,
        ("тысяча", "тысячи", "тысяч", True),
        ("миллион", "миллиона", "миллионов", False),
        ("миллиард", "миллиарда", "миллиардов", False),
    ]

    words = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            continue
        words.extend(triplet_to_words(g, feminine=(i == 1)))
        if i:
            one, few, many, _ = names[i]
            words.append(plural_form(g, one, few, many))
    return " ".join(words)

def maybe_cardinal(token):
    s = str(token).strip()

    if re.fullmatch(r"-?\d{1,9}", s):
        if len(s) > 1 and s[0] == "0":
            return " ".join(ONES[int(ch)] for ch in s)
        return int_to_cardinal(int(s))

    if re.fullmatch(r"\d{1,3}( \d{3}){1,3}", s):
        return int_to_cardinal(int(s.replace(" ", "")))

    return None
```

### Ordinals, Dates, And Years

```python
MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
MONTH_BY_NUM = {v: k for k, v in MONTHS.items()}

DAY_ORD_GEN = {
    1: "первого", 2: "второго", 3: "третьего", 4: "четвертого",
    5: "пятого", 6: "шестого", 7: "седьмого", 8: "восьмого",
    9: "девятого", 10: "десятого", 11: "одиннадцатого",
    12: "двенадцатого", 13: "тринадцатого", 14: "четырнадцатого",
    15: "пятнадцатого", 16: "шестнадцатого", 17: "семнадцатого",
    18: "восемнадцатого", 19: "девятнадцатого", 20: "двадцатого",
    21: "двадцать первого", 22: "двадцать второго",
    23: "двадцать третьего", 24: "двадцать четвертого",
    25: "двадцать пятого", 26: "двадцать шестого",
    27: "двадцать седьмого", 28: "двадцать восьмого",
    29: "двадцать девятого", 30: "тридцатого",
    31: "тридцать первого",
}

DAY_ORD_NOM = {
    1: "первое", 2: "второе", 3: "третье", 4: "четвертое",
    5: "пятое", 6: "шестое", 7: "седьмое", 8: "восьмое",
    9: "девятое", 10: "десятое", 11: "одиннадцатое",
    12: "двенадцатое", 13: "тринадцатое", 14: "четырнадцатое",
    15: "пятнадцатое", 16: "шестнадцатое", 17: "семнадцатое",
    18: "восемнадцатое", 19: "девятнадцатое", 20: "двадцатое",
    21: "двадцать первое", 22: "двадцать второе",
    23: "двадцать третье", 24: "двадцать четвертое",
    25: "двадцать пятое", 26: "двадцать шестое",
    27: "двадцать седьмое", 28: "двадцать восьмое",
    29: "двадцать девятое", 30: "тридцатое",
    31: "тридцать первое",
}
```

Keep the ordinal implementation small but cover at least masculine ordinals through years around `1000-2999`. The strong code used:

```python
def ordinal_masculine(n):
    n = int(n)
    if n <= 0:
        return int_to_cardinal(n)
    if n < 20:
        return ORD_UNITS_M[n]
    if n < 100:
        if n % 10 == 0:
            return ORD_TENS_M[n]
        return int_to_cardinal(n - n % 10) + " " + ORD_UNITS_M[n % 10]
    if n < 1000:
        if n % 100 == 0:
            return ORD_HUNDREDS_M[n]
        return int_to_cardinal(n - n % 100) + " " + ordinal_masculine(n % 100)
    if n < 2000:
        rest = n - 1000
        if rest == 0:
            return "тысячный"
        return "тысяча " + ordinal_masculine(rest)
    if n < 3000:
        rest = n - 2000
        if rest == 0:
            return "двухтысячный"
        return "две тысячи " + ordinal_masculine(rest)
    return int_to_cardinal(n)
```

For year genitive, convert the final ordinal ending:

```python
def ordinal_genitive_year(n):
    s = ordinal_masculine(n)
    repl = [
        ("первый", "первого"), ("второй", "второго"),
        ("третий", "третьего"), ("четвертый", "четвертого"),
        ("пятый", "пятого"), ("шестой", "шестого"),
        ("седьмой", "седьмого"), ("восьмой", "восьмого"),
        ("девятый", "девятого"), ("десятый", "десятого"),
        ("одиннадцатый", "одиннадцатого"),
        ("двенадцатый", "двенадцатого"),
        ("тринадцатый", "тринадцатого"),
        ("четырнадцатый", "четырнадцатого"),
        ("пятнадцатый", "пятнадцатого"),
        ("шестнадцатый", "шестнадцатого"),
        ("семнадцатый", "семнадцатого"),
        ("восемнадцатый", "восемнадцатого"),
        ("девятнадцатый", "девятнадцатого"),
        ("двадцатый", "двадцатого"),
        ("тридцатый", "тридцатого"),
        ("сороковой", "сорокового"),
        ("пятидесятый", "пятидесятого"),
        ("шестидесятый", "шестидесятого"),
        ("семидесятый", "семидесятого"),
        ("восьмидесятый", "восьмидесятого"),
        ("девяностый", "девяностого"),
        ("сотый", "сотого"),
        ("двухсотый", "двухсотого"),
        ("трехсотый", "трехсотого"),
        ("четырехсотый", "четырехсотого"),
        ("пятисотый", "пятисотого"),
        ("шестисотый", "шестисотого"),
        ("семисотый", "семисотого"),
        ("восьмисотый", "восьмисотого"),
        ("девятисотый", "девятисотого"),
        ("двухтысячный", "двухтысячного"),
        ("тысячный", "тысячного"),
    ]
    for a, b in repl:
        if s.endswith(a):
            return s[:-len(a)] + b
    return s
```

Date fallback patterns:

```python
def maybe_date(token):
    s = str(token).strip()

    m = re.fullmatch(r"(\d{1,2}) ([а-яё]+) (\d{3,4})( года)?", s, flags=re.I)
    if m:
        day = int(m.group(1))
        month = m.group(2).lower()
        year = int(m.group(3))
        if 1 <= day <= 31 and month in MONTHS:
            return f"{DAY_ORD_GEN[day]} {month} {ordinal_genitive_year(year)} года"

    m = re.fullmatch(r"(\d{1,2})\.([0-1]?\d)\.(\d{2}|\d{4})", s)
    if m:
        day = int(m.group(1))
        month_num = int(m.group(2))
        year = int(m.group(3))
        if year < 100:
            year += 2000 if year < 30 else 1900
        if 1 <= day <= 31 and month_num in MONTH_BY_NUM:
            return f"{DAY_ORD_NOM[day]} {MONTH_BY_NUM[month_num]} {ordinal_genitive_year(year)} года"

    m = re.fullmatch(r"(\d{4}) (год|года|году|годах)", s)
    if m:
        year = int(m.group(1))
        tail = m.group(2)
        if tail == "год":
            return f"{ordinal_masculine(year)} год"
        if tail == "года":
            return f"{ordinal_genitive_year(year)} года"
        return f"{ordinal_genitive_year(year)} {tail}"

    return None
```

### Time Fallback

```python
def maybe_time(token):
    s = str(token).strip()
    m = re.fullmatch(r"(\d{1,2}):\s?(\d{2})", s)
    if not m:
        return None

    h, minute = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 24 and 0 <= minute < 60):
        return None

    if minute == 0:
        if h == 1:
            return "один час"
        return int_to_cardinal(h) + " " + plural_form(h, "час", "часа", "часов")

    hp = "час" if h == 1 else int_to_cardinal(h) + " " + plural_form(h, "час", "часа", "часов")
    return hp + " " + int_to_cardinal(minute) + " " + plural_form(minute, "минута", "минуты", "минут")
```

### Abbreviation And Latin Fallbacks

```python
LATIN_TRANS = {
    "a": "а", "b": "б", "c": "с", "d": "д", "e": "е", "f": "ф",
    "g": "г", "h": "х", "i": "и", "j": "д ж", "k": "к", "l": "л",
    "m": "м", "n": "н", "o": "о", "p": "п", "q": "к", "r": "р",
    "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "к с",
    "y": "й", "z": "з",
}

def latin_trans(token):
    s = str(token)
    if not re.search(r"[A-Za-z]", s):
        return None
    if re.search(r"\d|@|/", s):
        return None

    out = []
    for ch in s:
        low = ch.lower()
        if low in LATIN_TRANS:
            for part in LATIN_TRANS[low].split():
                out.append(part + "_trans")
        elif ch in ".-_":
            out.append("sil" if ch in "-_" else "точка")
        else:
            return None

    return " ".join(out) if out else None

def letters_fallback(token):
    s = str(token).strip()

    if re.fullmatch(r"[A-ZА-ЯЁ]{2,6}", s):
        return " ".join(ch.lower() for ch in s)

    if re.fullmatch(r"([A-ZА-ЯЁ]\. ?){1,5}", s):
        chars = re.findall(r"[A-ZА-ЯЁ]", s)
        return " ".join(ch.lower() for ch in chars)

    return None
```

### Prediction Order

This order matters. Dates should run before raw cardinals, because date strings contain digits but need ordinal/month/year handling.

```python
def rule_predict(token):
    for fn in (maybe_date, maybe_time, maybe_cardinal, letters_fallback, latin_trans):
        pred = fn(token)
        if pred is not None:
            return pred
    return str(token)

def predict_series(before, mapping, use_rules=True):
    pred = before.map(mapping)
    miss = pred.isna()
    pred = pred.astype("object")

    if use_rules and miss.any():
        pred.loc[miss] = before.loc[miss].map(rule_predict)
    else:
        pred.loc[miss] = before.loc[miss].astype(str)

    return pred
```

### Submission Alignment

```python
train = pd.read_csv(os.path.join(DATA_DIR, "ru_train.csv.zip"), usecols=["before", "after"])
test = pd.read_csv(os.path.join(DATA_DIR, "ru_test_2.csv.zip"))
sample = pd.read_csv(os.path.join(DATA_DIR, "ru_sample_submission_2.csv.zip"))

mapping = build_map(train)
pred = predict_series(test.before, mapping, use_rules=True)

sub = pd.DataFrame({
    "id": test.sentence_id.astype(str) + "_" + test.token_id.astype(str),
    "after": pred.astype(str),
})

sub = sample[["id"]].merge(sub, on="id", how="left")

assert not sub["after"].isna().any()
assert list(sub.columns) == list(sample.columns)
assert len(sub) == len(sample)
assert set(sub["id"]) == set(sample["id"])

sub.to_csv("submission.csv", index=False)
```

## Local Diagnostics

After validation, inspect errors by class. This is the fastest way to decide whether an added rule is worth keeping.

```python
err = va[pred.values != va.after.values].copy()
err["pred"] = pred[pred.values != va.after.values].values

print(err["class"].value_counts().head(20))
print(err[["class", "before", "after", "pred"]].head(50).to_string(index=False))
```

Keep only rules that improve miss cases without touching memorized tokens.

## Score Milestones (relative — what each step buys)

Naive baseline: exact `before` passthrough scores high because of the dominant `self` class, but it is not the proven strong solution.

Proven strong configuration: a majority `before -> after` lookup table is the dominant lever and already clears a passing level on its own; conservative fallback rules for lookup misses add a small further gain by covering numbers, dates, years, times, abbreviations, and Latin strings.

Self-check against the `criterion.json` pass line, not a fixed threshold. What stays invariant is the ORDERING: the majority lookup table is the decisive lever, conservative miss-only rules add a small increment, and aggressive rules applied before lookup are the main way to regress by corrupting memorized tokens.
