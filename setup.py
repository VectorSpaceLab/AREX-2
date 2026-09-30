"""Compatibility shim for environments with an older setuptools build backend."""
from setuptools import find_packages, setup

setup(
    name="arex-evaluation-suite",
    version="0.1.0",
    package_dir={"": "evaluation"},
    packages=find_packages("evaluation", include=["arex_eval", "arex_eval.*"]),
    entry_points={"console_scripts": ["arex=arex_eval.cli:main"]},
)
