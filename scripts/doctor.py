#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from arex_eval.cli import main
if __name__ == '__main__': main(['doctor'])
