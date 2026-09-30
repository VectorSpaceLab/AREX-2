# Frontier-CS evaluation

This directory contains the two parts of the Frontier-CS track:

- `source/` is the Python evaluator and solution runner.
- `judge/` is the local Docker/SkyPilot checker service.
- `profiles/` contains versioned Harbor/Pi runtime configurations used by
  the long-running experiments.

The public single-problem command is:

```bash
python3 evaluate.py algorithmic 1 path/to/solution.cpp --backend docker
```

The optional multi-problem Pi runner is exposed from
`scripts/algorithmic/pi_concurrent.sh`. It is kept separate from the public
single-problem command because it requires Harbor, Pi credentials, and a
machine-specific runtime setup.
