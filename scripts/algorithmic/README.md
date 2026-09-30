# Frontier-CS runner

`run.sh` evaluates one downloaded Frontier-CS problem with a C++17 solution.
The problem statement, checker, limits, and test data stay under the
algorithmic data directory; the wrapper only selects the problem and backend.

| Input | Default | Meaning |
| --- | --- | --- |
| `PROBLEM` | required | Numeric problem ID |
| `SOLUTION` | required | Path to a C++17 source file |
| `FRONTIER_BACKEND` | `docker` | `docker` or `skypilot` |
| `FRONTIER_JUDGE_URL` | empty | Optional remote judge endpoint |

Examples:

```bash
python3 evaluate.py download algorithmic
scripts/algorithmic/run.sh 1 path/to/solution.cpp --dry-run
scripts/algorithmic/run.sh 1 path/to/solution.cpp
FRONTIER_BACKEND=skypilot scripts/algorithmic/run.sh 1 path/to/solution.cpp
```

The final result keeps the checker fields `scoreRatio` and
`scoreRatioUnbounded`; a compile failure, timeout, or failed required case is
not converted into a research-style pass rate.
