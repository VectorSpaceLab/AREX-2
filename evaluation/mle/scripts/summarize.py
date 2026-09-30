#!/usr/bin/env python3
"""Collect finished runs into one table.

Reads both layouts:
  - standalone runs made by scripts/run_one.sh:  <run>/logs/result.json + <run>/grade.log
  - mle-bench run groups made by run_agent.py:   <group>/<comp>/<seed>/logs/result.json
                                                 (grade with `mlebench grade` on the group)

  scripts/summarize.py runs/
  scripts/summarize.py runs/ --csv results.csv
"""
import argparse
import json
import re
from pathlib import Path

THRESHOLD_KEYS = ("gold_threshold", "silver_threshold", "bronze_threshold", "median_threshold")


def parse_grade(path: Path):
    """mlebench grade-sample prints log lines around one JSON object; pull the object out."""
    if not path.is_file():
        return {}
    txt = path.read_text(errors="ignore")
    for m in re.finditer(r"\{", txt):
        for end in re.finditer(r"\}", txt[m.start():]):
            blob = txt[m.start(): m.start() + end.end()]
            try:
                o = json.loads(blob)
            except Exception:
                continue
            if "score" in o or "valid_submission" in o:
                return o
    return {}


def medal(g):
    for k, name in (("gold_medal", "gold"), ("silver_medal", "silver"), ("bronze_medal", "bronze")):
        if g.get(k):
            return name
    if g.get("above_median"):
        return "above-median"
    return "none"


def collect(root: Path):
    rows = []
    for res in sorted(root.rglob("logs/result.json")):
        run = res.parent.parent
        r = json.loads(res.read_text())
        g = parse_grade(run / "grade.log")
        if not g:                                     # mle-bench writes grades next to the group
            for cand in (run / "grades.json", run.parent / "grades.json"):
                if cand.is_file():
                    try:
                        data = json.loads(cand.read_text())
                    except Exception:
                        continue
                    for item in (data.get("competition_reports") or []):
                        if item.get("competition_id") == r.get("comp"):
                            g = item
                            break
        rows.append({
            "comp": r.get("comp"), "model": r.get("solver"),
            "criterion": r.get("criterion_mode"), "rounds": len(r.get("rounds") or []),
            "wall_s": r.get("wall_s"), "tokens": r.get("tokens"),
            "submitted": bool(r.get("submission")),
            "score": g.get("score"), "valid": g.get("valid_submission"),
            "medal": medal(g) if g else None, "run": str(run),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, nargs="?", default=Path("runs"))
    ap.add_argument("--csv", type=Path, default=None)
    a = ap.parse_args()

    rows = collect(a.root)
    if not rows:
        print(f"no runs found under {a.root}")
        return
    w = max(len(str(r["comp"])) for r in rows) + 2
    print(f"{'competition':{w}} {'score':>12} {'medal':>13} {'valid':>6} {'sub':>4} "
          f"{'wall':>7} {'tokens':>10} {'rnds':>4}")
    for r in rows:
        print(f"{str(r['comp']):{w}} {str(r['score'])[:12]:>12} {str(r['medal']):>13} "
              f"{str(r['valid']):>6} {('Y' if r['submitted'] else '-'):>4} "
              f"{str(r['wall_s']) + 's':>7} {str(r['tokens']):>10} {str(r['rounds']):>4}")

    graded = [r for r in rows if r["score"] is not None]
    if graded:
        medals = sum(1 for r in graded if r["medal"] in ("gold", "silver", "bronze"))
        above = sum(1 for r in graded if r["medal"] in ("gold", "silver", "bronze", "above-median"))
        print(f"\n{len(rows)} runs, {len(graded)} graded, "
              f"{sum(1 for r in rows if r['submitted'])} with a submission, "
              f"{medals} any-medal, {above} above-median")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
