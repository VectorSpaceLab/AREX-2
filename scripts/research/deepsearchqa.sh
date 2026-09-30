#!/usr/bin/env bash
# DeepSearch-QA: one dataset, shared refine-equal evaluation profile.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
run_research_dataset "DeepSearch-QA" "$@"
