# Task skills

`agents/pi/skills/<competition-id>/SKILL.md` — one per competition, 8–22 KB of English markdown.

A skill is not documentation and not a prompt-engineering trick. It is a **distilled recipe**: the
approach, the specific model family and hyperparameters, the preprocessing that mattered, the
validation scheme, the tricks that actually moved the metric, and the traps. Concretely most skills
have this shape:

1. One line naming the task and the metric, and a pointer to `criterion.json` for the pass line.
2. **Approach** — a paragraph on the overall strategy, including what *not* to do (very often
   "do not reach for a CNN here").
3. Numbered steps, each with runnable code fragments: load, features, model, CV, post-processing,
   submission.
4. **Metric-moving changes, in priority order** — what to do first if the score is short.
5. Failure modes and sanity checks.

The code fragments are the important part. They are concrete enough that the agent can lift and
adapt them, which is what lets a 4-hour run reach a score that took the original author days.

## Conventions

* Refer to the data as `./input` — that is the relative name the harness symlinks in.
* Never hardcode a score line. Say "the pass line is in `criterion.json`". This is what
  "de-bound" means: the same skill then works whether the run discloses medal thresholds, the
  median line, or nothing (`PI_CRITERION`).
* Never name the source solution's standing ("the gold solution", "this reached silver"). It tells
  the agent how hard to try, which contaminates the measurement.
* Write in English regardless of the language you drafted it in; mixed-language skills measurably
  degrade the agent's instruction following.

## Adding one

1. Read a strong public solution to the competition and write the recipe, following the shape
   above. Keep the code fragments minimal and correct.
2. Drop it at `agents/pi/skills/<competition-id>/SKILL.md`.
3. Add the competition's leaderboard thresholds to `agents/pi/tasks.json` (mle-bench ships them
   under `mlebench/competitions/<id>/leaderboard.csv`).
4. Add the id to `splits/lite.txt` if it should be part of the default set.
5. Rebuild the image (`scripts/build_image.sh`) — skills are baked in, not mounted.

## Checking one is de-bound

```bash
grep -rniE 'medal|gold|silver|bronze|\b(gold|silver|bronze)_threshold' agents/pi/skills/
```

should print nothing.
