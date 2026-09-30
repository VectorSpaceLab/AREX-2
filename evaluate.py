#!/usr/bin/env python3
"""Unified checkout entry point for all AREX-2 evaluation tracks.

Research datasets can be selected directly::

    python3 evaluate.py BrowseComp --n 10

Other operations use the explicit subcommand form::

    python3 evaluate.py list
    python3 evaluate.py doctor
    python3 evaluate.py download BrowseComp
    python3 evaluate.py algorithmic 1 path/to/solution.cpp --backend docker
    python3 evaluate.py mle leaf-classification

The implementation lives in ``evaluation/arex_eval``; this file keeps the
checkout-level command independent of Python package installation details.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "evaluation"))
from arex_eval.cli import main


_COMMANDS = frozenset({
    "doctor",
    "list",
    "download",
    "research",
    "eval",
    "evaluate",
    "algorithmic",
    "mle",
})


def main_from_checkout(argv: list[str] | None = None) -> None:
    """Dispatch explicit CLI commands or shorthand research dataset names."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        main(["--help"])
    elif args[0] in _COMMANDS:
        main(args)
    else:
        main(["research", *args])


if __name__ == "__main__":
    main_from_checkout()
