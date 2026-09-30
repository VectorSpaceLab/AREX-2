#!/usr/bin/env python3
"""In-container runner: solve ONE MLE-bench competition with the `pi` coding agent.

This is the whole harness. It runs *inside* the benchmark container, where the
environment and the input are fixed by the image and the mounts:

    /home/data                  competition data, READ-ONLY  (mle-bench `prepared/public`)
    /home/submission/submission.csv   the graded deliverable
    /home/logs                  extracted logs (pi trajectory, result.json, pi.log)
    /home/code                  extracted code (the agent's workspace)
    /home/agent                 this harness + the skills library

For the competition it:
  1. assembles CONTEXT.md  = objective + pass criterion + task skill + data card,
  2. runs `pi` with a fixed prompt contract, capturing the full trajectory
     (thinking + tool calls + tool results) via `--session-dir`,
  3. optionally runs N refine rounds, each seeded with the previous round's
     workspace, run log and closing self-diagnosis,
  4. syncs ./submission.csv out to /home/submission/ and writes result.json.

It does NOT grade. Grading is `mlebench grade-sample` on the host, so this
container never sees the private test labels.

Configuration is entirely by environment variable -- see docs/configuration.md.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

AGENT_DIR = Path(os.environ.get("AGENT_DIR", "/home/agent"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/home/data"))
SUBMISSION_DIR = Path(os.environ.get("SUBMISSION_DIR", "/home/submission"))
LOGS_DIR = Path(os.environ.get("LOGS_DIR", "/home/logs"))
CODE_DIR = Path(os.environ.get("CODE_DIR", "/home/code"))

COMP = os.environ.get("COMPETITION_ID", "")
WALL = int(os.environ.get("TIME_LIMIT_SECS", "14400"))
REFINE_ROUNDS = int(os.environ.get("REFINE_ROUNDS", "0"))
PROVIDER = os.environ.get("PI_PROVIDER", "deepseek")
MODEL = os.environ.get("PI_MODEL", "deepseek-v4-flash")
THINKING = os.environ.get("PI_THINKING", "high")
PI_BIN = os.environ.get("PI_BIN", "pi")
NO_SKILL = os.environ.get("PI_NO_SKILL", "").lower() in ("1", "true", "yes")
# medal | median | none -- how much of the grading scale CONTEXT.md/criterion.json reveal.
# `none` is the apples-to-apples setting against published MLE-bench numbers; see
# docs/differences.md.
CRITERION = os.environ.get("PI_CRITERION", "medal").lower()
PROVIDER_RETRIES = int(os.environ.get("PI_PROVIDER_RETRIES", "3"))
VALIDATE_URL = os.environ.get("VALIDATE_URL", "http://localhost:5000/validate")
SKILLS = AGENT_DIR / os.environ.get("SKILLS_DIR", "skills")
TASKS = json.loads((AGENT_DIR / "tasks.json").read_text()) if (AGENT_DIR / "tasks.json").exists() else {}
HARDWARE = os.environ.get("HARDWARE", "an unknown accelerator")

# The read-only data is exposed inside the workspace under this relative name, so every path
# the agent writes is relative to its own working directory.
DATA_NAME = os.environ.get("DATA_NAME", "input")
GPU_NOTE = os.environ.get("PI_GPU_NOTE",
    "Before training, check free GPU memory with torch.cuda.mem_get_info() and size the "
    "model/batch to fit. Prefer fp16/bf16 and gradient accumulation over huge batches. "
    "Right-size the number of CV folds to the wall clock (3 folds is usually enough; do not "
    "blindly run 5+ if time is tight).")


def log(msg):
    print(f"[harness] {msg}", flush=True)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- data card

def _head_lines(p, n=3):
    """First n lines only -- some sample submissions are hundreds of MB."""
    out = []
    try:
        with open(p, errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                out.append(line.rstrip("\n"))
    except OSError:
        pass
    return out


def find_sample_sub(pub):
    """Locate the sample submission whatever Kaggle called it."""
    for name in ("sample_submission.csv", "sampleSubmission.csv", "SampleSubmission.csv"):
        if (pub / name).exists():
            return pub / name
    for p in sorted(pub.glob("*.csv")):
        if "ubmission" in p.name.lower():
            return p
    return None


def read_card(pub):
    desc = pub / "description.md"
    out = {"description": desc.read_text(errors="ignore")[:6000] if desc.exists() else ""}
    ss = find_sample_sub(pub)
    if ss:
        out["submission_file"] = ss.name
        head = _head_lines(ss, 3)
        out[ss.name] = {"header": head[0] if head else "", "sample": head[1] if len(head) > 1 else ""}
    for fn in ("train.csv", "test.csv"):
        p = pub / fn
        if p.exists():
            head = _head_lines(p, 3)
            out[fn] = {"header": head[0] if head else "", "sample": head[1] if len(head) > 1 else ""}
    return out


# ------------------------------------------------------------------- workspace + criterion

def setup_ws(ws):
    ws.mkdir(parents=True, exist_ok=True)
    link = ws / DATA_NAME
    if link.is_symlink():
        link.unlink()
    if not link.exists():
        try:
            link.symlink_to(DATA_DIR)
        except OSError as e:
            log(f"could not symlink {link} -> {DATA_DIR}: {e}")
    return ws


def write_criterion(ws):
    """The pass line the skills refer to. `higher_is_better` is inferred from the thresholds
    themselves (gold above median => higher is better), so it is right for both directions."""
    thr = (TASKS.get(COMP) or {}).get("thresholds") or {}
    g, md, br = thr.get("gold_threshold"), thr.get("median_threshold"), thr.get("bronze_threshold")
    higher = (g is not None and md is not None and g >= md)
    if CRITERION == "none" or not thr:
        crit = {"comp": COMP, "higher_is_better": higher if thr else None,
                "pass_rule": "best_effort", "pass_value": None, "thresholds": {},
                "note": "No pass line is disclosed for this run. Push the competition metric as "
                        "high (or as low, if lower is better) as the wall clock allows."}
    elif CRITERION == "median":
        crit = {"comp": COMP, "higher_is_better": higher, "pass_rule": "above_median",
                "pass_value": md, "thresholds": {"median_threshold": md},
                "note": "Pass = beat the competition's median leaderboard score."}
    else:
        crit = {"comp": COMP, "higher_is_better": higher, "pass_rule": "any_medal",
                "pass_value": br, "thresholds": thr,
                "note": "Pass = reach the bronze line or better under the competition metric."}
    (ws / "criterion.json").write_text(json.dumps(crit, indent=2))
    return crit


def build_context(ws, card, refine=None):
    skill_md = "" if NO_SKILL else (SKILLS / COMP / "SKILL.md").read_text()
    if skill_md:
        sk = ws / "skills" / COMP
        sk.mkdir(parents=True, exist_ok=True)
        (sk / "SKILL.md").write_text(skill_md)
    din = f"./{DATA_NAME}"
    crit = write_criterion(ws)
    subf = card.get("submission_file", "sample_submission.csv")
    L = [f"# Task: solve the MLE-bench competition `{COMP}`\n", "## [1] Objective & constraints"]
    L.append(f"- Write your submission to `./submission.csv`, with columns EXACTLY matching the "
             f"sample submission file **`{subf}`** (found under `{din}/`), every id covered, no "
             "missing rows.")
    L.append(f"- Dataset (READ-ONLY): `{din}/` — all competition files are here; never read "
             "anything outside it.")
    L.append("- Training interpreter: `python` / `pip` on PATH (CUDA torch, transformers, sklearn, "
             "pandas preinstalled). First run one line to confirm `torch.cuda.is_available()`.")
    L.append("- Need an extra package? Install it — this container is yours alone.")
    L.append(f"- Hardware: {HARDWARE}.")
    L.append(f"- Pass criterion is in `./criterion.json`: "
             f"{'higher' if crit['higher_is_better'] else 'lower'} score is better; "
             f"pass = **{crit['pass_rule']}**"
             + (f" (target ≈ {crit['pass_value']})" if crit.get("pass_value") is not None else "")
             + f". {crit['note']}\n")
    if NO_SKILL:
        L.append("## [2] How to solve")
        L.append("> No recipe is provided — design the approach yourself from the data card below "
                 "and from what you find in the dataset.\n")
    else:
        L.append("## [2] How to solve (task-specific skill — also materialized under ./skills/)")
        L.append("> The skill below is distilled from a strong solution to this task. Follow its "
                 "recipe and key tricks — do NOT reinvent a different approach.\n")
        L.append(skill_md)
    L.append("\n## [3] What this competition is (data card)")
    for k, v in card.items():
        if k.endswith(".csv"):
            L.append(f"- {k}: columns=`{v['header']}`  example=`{v['sample']}`")
    if card.get("description"):
        L.append("\n### description.md (excerpt)\n" + card["description"])
    if refine:
        L.append("\n## [4] Previous attempt to improve on (this is a REFINE round)")
        L.append(refine.strip())
    L.append("\n## Wrap-up")
    L.append("Produce `./submission.csv`, then end with a short **[Diagnosis]**: (1) one line on "
             "the method used (2) your OOF/CV self-estimate and how far it is from the pass line "
             "(3) the main bottleneck (4) if short of target, what you would change next.")
    (ws / "CONTEXT.md").write_text("\n".join(L))


def build_prompt(refine=False):
    din = f"./{DATA_NAME}"
    if refine:
        head = (
            "You are a machine-learning engineer improving on a previous attempt at the Kaggle "
            "competition specified in ./CONTEXT.md. Your tools are read / bash / edit / write; your "
            "working directory is your scratch space and every path below is relative to it.\n\n"
            "Read ./CONTEXT.md in full — its section [4] records the previous attempt: the code "
            "already present in this directory, the previous run log at ./prev_run.log, and where it "
            "left off. Study the previous code and log against the skill in section [2], diagnose "
            "what is limiting the score, then IMPROVE THE IMPLEMENTATION to raise it — change the "
            "method, features, model, or hyperparameters and retrain. You MUST improve by "
            "retraining; do NOT produce a submission by averaging, copying, or lightly perturbing "
            "the previous submission.\n"
            "The previous round's reusable preprocessing under ./cache/ and any model checkpoints "
            "(*.pt/*.pkl) are already here — REUSE them (skip rebuilding features, warm-start from "
            "the checkpoint) and spend your time on the improvement, not on redoing prep.\n\n")
    elif NO_SKILL:
        head = (
            "You are a machine-learning engineer competing in the Kaggle competition specified in "
            "./CONTEXT.md. Your tools are read / bash / edit / write; your working directory is your "
            "scratch space and every path below is relative to it.\n\n"
            "Begin by reading ./CONTEXT.md in full: it defines the objective, the read-only dataset "
            f"at {din}/ and the pass criterion. No solution recipe is given — explore the data, "
            "choose your own approach, and push the score as high as the wall clock allows.\n\n")
    else:
        head = (
            "You are a machine-learning engineer competing in the Kaggle competition specified in "
            "./CONTEXT.md. Your tools are read / bash / edit / write; your working directory is your "
            "scratch space and every path below is relative to it.\n\n"
            "Begin by reading ./CONTEXT.md in full: it defines the objective, the read-only dataset "
            f"at {din}/, the pass criterion, and (section [2]) a skill distilled from a strong "
            "solution to this competition. Reproduce that solution — follow its recipe and key "
            "tricks faithfully; do not devise an unrelated approach.\n\n")
    return head + (
        "Environment:\n"
        f"- Dataset (read-only): {din}/ — never access anything outside it.\n"
        "- Training interpreter: `python` on PATH (CUDA torch / transformers / sklearn / pandas "
        "preinstalled). Confirm torch.cuda.is_available() once before training.\n"
        f"- Hardware: {HARDWARE}.\n"
        f"- {GPU_NOTE}\n"
        "- Deliverable: ./submission.csv — columns exactly matching the sample-submission file named "
        "in CONTEXT.md (find it under the data dir; the name may be sampleSubmission.csv or "
        "similar), every id present, no missing values.\n\n"
        "Operating requirements:\n"
        "1. BASELINE FIRST — within your first ~15 minutes, write a COMPLETE, valid "
        "./submission.csv from the cheapest possible model (a constant, a quick linear/GBM, or one "
        "fast fold) BEFORE any heavy training or preprocessing. This guarantees a gradeable "
        "submission even if a long step later runs out of wall clock. Only after that submission "
        "exists do you build the strong model, overwriting ./submission.csv as it improves. Never "
        "exit — or hit the wall — with no valid submission.\n"
        "2. CACHE reusable preprocessing under ./cache/ (built features, graphs, tokenized data) "
        "and save model checkpoints as *.pt/*.pkl — these SURVIVE into the next refine round, so "
        "later rounds skip rebuilding. NEVER put predictions, OOF arrays, or anything derived from "
        "test labels in ./cache — those are outputs, not inputs.\n"
        "3. NO LABEL LOOKUP — every test prediction must come from a model you trained on the "
        f"training data under {din}. Do NOT construct predictions by joining test ids to any label "
        "table, embedded label field, or reconstructed train/test split; submissions that match a "
        "known label source row-for-row are void. Train a real model.\n"
        "4. Size training to your wall-clock budget; if folds/epochs won't all fit, reduce them "
        "rather than forfeit the submission. Run long training in the foreground and stream logs to "
        "stdout.\n"
        "5. Close with a [Diagnosis]: (1) one-line method (2) OOF/CV self-estimate and gap to the "
        "pass line (3) the dominant bottleneck (4) the single change you would try next."
    )


# ------------------------------------------------------------------------- trajectory reading

def _newest_traj(traj_dir):
    js = sorted(Path(traj_dir).glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return js[-1] if js else None


def _traj_messages(traj_dir):
    f = _newest_traj(traj_dir)
    if not f:
        return []
    out = []
    for ln in f.read_text(errors="ignore").splitlines():
        try:
            o = json.loads(ln)
        except Exception:
            continue
        if o.get("type") == "message" and isinstance(o.get("message"), dict):
            out.append(o["message"])
    return out


def traj_tokens(traj_dir):
    """Tokens the assistant actually produced. 0 means the run never got off the ground
    (provider 5xx / timeout), which is what the retry below keys on."""
    n = 0
    for f in Path(traj_dir).glob("*.jsonl"):
        for line in f.read_text(errors="ignore").splitlines():
            if '"usage"' not in line:
                continue
            try:
                u = (json.loads(line).get("message") or {}).get("usage") or {}
            except Exception:
                continue
            n += int(u.get("totalTokens") or 0)
    return n


def transcript_from_traj(traj_dir, tool_cap=1400):
    """Render the trajectory into a plain-text run log for the next refine round to read
    (pi's own stdout is near-empty in -p mode)."""
    L = []
    for m in _traj_messages(traj_dir):
        role = m.get("role")
        for b in (m.get("content") or []):
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "thinking" and b.get("thinking"):
                L.append("[think] " + b["thinking"].strip())
            elif bt == "text" and b.get("text") and role == "assistant":
                L.append("[assistant] " + b["text"].strip())
            elif bt == "toolCall":
                args = b.get("arguments") or {}
                cmd = args.get("command") or args.get("path") or json.dumps(args, ensure_ascii=False)
                L.append(f"$ [{b.get('name', '?')}] {str(cmd)[:600]}")
            elif bt == "text" and role == "toolResult":
                t = (b.get("text") or "").rstrip()
                if len(t) > tool_cap:
                    t = t[:tool_cap] + f"\n…(+{len(t) - tool_cap} chars truncated)"
                L.append("  ↳ " + t.replace("\n", "\n    "))
    return "\n".join(L)


def last_diagnosis(traj_dir):
    """The agent's closing [Diagnosis]. If the run was cut off at the wall before wrapping up,
    fall back to its most recent reasoning so a refine round still knows where it stopped."""
    say, think = [], []
    for m in _traj_messages(traj_dir):
        if m.get("role") != "assistant":
            continue
        for b in (m.get("content") or []):
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text" and b.get("text"):
                say.append(b["text"])
            elif b.get("type") == "thinking" and b.get("thinking"):
                think.append(b["thinking"])
    joined = "\n".join(say)
    i = joined.rfind("[Diagnosis]")
    if i >= 0:
        return joined[i:i + 1600].strip()
    tail = (say[-1] if say else "") or (think[-1] if think else "")
    if not tail:
        return ""
    return ("(no [Diagnosis] — run cut off at the wall; latest reasoning:)\n" + tail[-1200:]).strip()


# ------------------------------------------------------------------------------ submission

def sync_submission(ws):
    """Copy the agent's ./submission.csv to the graded location. Called periodically during the
    run and once at the end, so a run killed at the wall still submits its latest file."""
    src, dst = ws / "submission.csv", SUBMISSION_DIR / "submission.csv"
    try:
        if not src.is_file():
            return False
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime and dst.stat().st_size == src.stat().st_size:
            return True
        SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except OSError as e:
        log(f"submission sync failed: {e}")
        return False


def validate_submission():
    """Ask the benchmark's validation server whether the submission is well-formed. Optional:
    in standalone mode there is no server and this just reports unavailable."""
    sub = SUBMISSION_DIR / "submission.csv"
    if not sub.is_file():
        return {"valid": False, "detail": "no submission.csv"}
    try:
        r = subprocess.run(["curl", "-s", "--max-time", "120", "-X", "POST",
                            "-F", f"file=@{sub}", VALIDATE_URL],
                           capture_output=True, text=True, timeout=180)
        body = (r.stdout or "").strip()
        if not body:
            return {"valid": None, "detail": "validation server unavailable"}
        return {"valid": "not valid" not in body.lower() and "invalid" not in body.lower(),
                "detail": body[:500]}
    except Exception as e:
        return {"valid": None, "detail": f"validation skipped: {e!r}"}


# ------------------------------------------------------------------------------- run pi

def run_pi(prompt, cwd, traj_dir, timeout, logf, env):
    """Run pi to completion or to `timeout` seconds, streaming its stdout to logf.
    Returns the exit code, or 124 if we killed it at the wall (same convention as timeout(1))."""
    cmd = [PI_BIN, "--provider", PROVIDER, "--model", MODEL, "--thinking", THINKING,
           "--session-dir", str(traj_dir), "-nc", "-p", prompt]
    p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)

    def _reader():
        try:
            with open(logf, "a") as lf:
                for line in p.stdout:
                    lf.write(line)
                    lf.flush()
        except Exception:
            pass

    threading.Thread(target=_reader, daemon=True).start()
    t0 = time.time()
    while True:
        rc = p.poll()
        if rc is not None:
            return rc
        if time.time() - t0 > timeout:
            log(f"wall clock reached ({timeout}s) — killing pi")
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
            return 124
        time.sleep(3)


def pi_env(ws):
    env = dict(os.environ)
    if os.environ.get("NODE_BIN"):
        env["PATH"] = f"{os.environ['NODE_BIN']}:" + env.get("PATH", "")
    return env


def one_round(ws, traj, logf, prompt, wall, env):
    """One pi invocation, with a retry for the case where the model endpoint is down and the
    run dies in seconds having produced nothing."""
    t0 = time.time()
    rc = run_pi(prompt, ws, traj, wall, logf, env)
    for attempt in range(1, PROVIDER_RETRIES + 1):
        spent = time.time() - t0
        if traj_tokens(traj) > 0 or spent > 600 or spent >= wall:
            break
        back = 60 * 2 ** (attempt - 1)
        log(f"provider retry {attempt}/{PROVIDER_RETRIES}: 0 tokens after {int(spent)}s, "
            f"backing off {back}s")
        with open(logf, "a") as lf:
            lf.write(f"\n# provider-retry {attempt}/{PROVIDER_RETRIES}: 0 tokens after "
                     f"{int(spent)}s, sleeping {back}s\n")
        time.sleep(back)
        rc = run_pi(prompt, ws, traj, max(600, int(wall - (time.time() - t0))), logf, env)
    return rc


# Artifacts worth extracting from the workspace. Everything else the agent leaves behind --
# ./cache, checkpoints, its own venv, the ./input symlink -- can be gigabytes and is not a result.
CODE_GLOBS = ("*.py", "*.sh", "*.ipynb", "*.md", "*.json", "*.txt", "*.log", "*.yaml", "*.yml")
CODE_SKIP_DIRS = {"cache", ".venv", "venv", "input", "skills", "__pycache__", ".git", "wandb"}
CODE_MAX_BYTES = 4 * 1024 * 1024


def export_code(ws):
    """Copy the agent's source + run artifacts into CODE_DIR so they are extracted with the run."""
    for src in sorted(ws.rglob("*")):
        rel = src.relative_to(ws)
        if not src.is_file() or src.is_symlink():
            continue
        if set(rel.parts[:-1]) & CODE_SKIP_DIRS:
            continue
        if not any(src.match(g) for g in CODE_GLOBS) or src.stat().st_size > CODE_MAX_BYTES:
            continue
        dst = CODE_DIR / rel
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        except OSError as e:
            log(f"could not export {rel}: {e}")


def main():
    if not COMP:
        sys.exit("COMPETITION_ID is not set")
    if not DATA_DIR.is_dir() or not any(DATA_DIR.iterdir()):
        sys.exit(f"no competition data at {DATA_DIR}")
    if not NO_SKILL and not (SKILLS / COMP / "SKILL.md").is_file():
        sys.exit(f"no skill for {COMP} at {SKILLS / COMP / 'SKILL.md'} "
                 f"(set PI_NO_SKILL=1 to run without one)")

    for d in (LOGS_DIR, CODE_DIR, SUBMISSION_DIR):
        d.mkdir(parents=True, exist_ok=True)
    ws = setup_ws(AGENT_DIR / "workspace")
    traj = LOGS_DIR / "traj"
    traj.mkdir(parents=True, exist_ok=True)
    logf = LOGS_DIR / "pi.log"
    card = read_card(DATA_DIR)
    env = pi_env(ws)

    stop = threading.Event()

    def _syncer():                   # keep /home/submission current even if we die at the wall
        while not stop.wait(60):
            sync_submission(ws)

    threading.Thread(target=_syncer, daemon=True).start()

    rounds, t_start = [], time.time()
    total_rounds = 1 + max(0, REFINE_ROUNDS)
    per_round = max(600, WALL // total_rounds)
    # A refine round needs enough clock to be worth starting; scale the floor to the budget so
    # short smoke runs still exercise the refine path.
    min_round = min(600, max(30, WALL // (total_rounds * 2)))
    refine_note = None
    for rnd in range(1, total_rounds + 1):
        left = WALL - (time.time() - t_start)
        if rnd > 1 and left < min_round:
            log(f"only {int(left)}s left — skipping refine round {rnd}")
            break
        budget = max(min_round, int(min(per_round, left)))
        build_context(ws, card, refine=refine_note)
        prompt = build_prompt(refine=bool(refine_note))
        round_traj = traj / f"round{rnd}"
        round_traj.mkdir(parents=True, exist_ok=True)
        with open(logf, "a") as lf:
            lf.write(f"\n# ===== round {rnd}/{total_rounds}  {COMP}  pi/{MODEL}  "
                     f"budget={budget}s  {now_iso()} =====\n")
        log(f"round {rnd}/{total_rounds}: {COMP} via pi/{MODEL} (budget {budget}s)")
        rc = one_round(ws, round_traj, logf, prompt, budget, env)
        sync_submission(ws)
        rounds.append({"round": rnd, "rc": rc, "tokens": traj_tokens(round_traj),
                       "wall_s": round(time.time() - t_start), "at": now_iso()})
        log(f"round {rnd} done: rc={rc} tokens={rounds[-1]['tokens']}")
        if rnd < total_rounds:
            (ws / "prev_run.log").write_text(transcript_from_traj(round_traj))
            diag = last_diagnosis(round_traj)
            refine_note = ("The previous round's code is already in this directory and its full run "
                           "log is at `./prev_run.log`. Its closing self-assessment was:\n\n"
                           + (diag or "(none — it produced no wrap-up)"))

    stop.set()
    sync_submission(ws)
    export_code(ws)
    result = {
        "comp": COMP, "solver": f"pi/{MODEL}", "provider": PROVIDER, "thinking": THINKING,
        "skill": None if NO_SKILL else str((SKILLS / COMP / "SKILL.md").relative_to(AGENT_DIR)),
        "criterion_mode": CRITERION, "wall_budget_s": WALL,
        "wall_s": round(time.time() - t_start), "rounds": rounds,
        "tokens": sum(r["tokens"] for r in rounds),
        "submission": (SUBMISSION_DIR / "submission.csv").is_file(),
        "validation": validate_submission(),
        "traj": [str(p.relative_to(LOGS_DIR)) for p in sorted(traj.rglob("*.jsonl"))],
        "at": now_iso(),
    }
    (LOGS_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    log(f"finished: submission={result['submission']} validation={result['validation']} "
        f"wall={result['wall_s']}s tokens={result['tokens']}")
    # Always exit 0: a missing submission is a benchmark result, not a harness failure.
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    sys.exit(main())
