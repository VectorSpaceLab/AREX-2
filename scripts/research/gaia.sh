#!/usr/bin/env bash
# GAIA text validation: one dataset, shared refine-equal evaluation profile.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
run_research_dataset "GAIA-2023-validation-text-103" "$@"
