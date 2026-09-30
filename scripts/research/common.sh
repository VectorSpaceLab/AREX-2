#!/usr/bin/env bash
# Shared launcher for the four headline research benchmarks.
#
# The evaluator owns the actual benchmark logic. This file only translates
# shell variables into the checkout-level `evaluate.py` command, so the
# benchmark-specific wrappers stay small and auditable.

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Source this file from one of the dataset runners in this directory.\n' >&2
    exit 2
fi

RESEARCH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AREX_REPO_ROOT="$(cd "$RESEARCH_SCRIPT_DIR/../.." && pwd)"

run_research_dataset() {
    if [[ "$#" -lt 1 ]]; then
        printf 'Usage: run_research_dataset DATASET [extra evaluate.py options...]\n' >&2
        return 2
    fi

    local dataset="$1"
    shift

    local dry_run=0
    local option
    for option in "$@"; do
        if [[ "$option" == "--dry-run" ]]; then
            dry_run=1
            break
        fi
    done

    # Defaults mirror the maintained refine-equal profile. The profile is the
    # single source of truth for generation, review, budget, and retry flags;
    # keeping them here as comments makes the wrapper's contract visible.
    # concurrency=1; outer rounds=10; calls per round=300; total calls=1500
    # confidence-tiered review=95/90; temperature=1.0; top_p=0.95; top_k=20
    # min_p=0.0; presence_penalty=1.5; repetition_penalty=1.0
    # context tokens=240000; response tokens=16384; tool retries=20
    # LLM retries=5; general attempts=10; per-case attempts=1
    local num_tasks="${NUM_TASKS:-10}"
    local start_index="${START_INDEX:-0}"
    local concurrency="${CONCURRENCY:-1}"
    local model_name="${AREX_MODEL_NAME:-YOUR_MODEL_NAME}"
    local model_api_key_env="${AREX_API_KEY_ENV:-${MODEL_API_KEY_ENV:-MODEL_API_KEY}}"
    local tokenizer_path="${AREX_TOKENIZER_PATH:-}"
    local base_url="${AREX_BASE_URL:-}"
    local configured_judge_model="${AREX_JUDGE_MODEL:-${JUDGE_MODEL:-}}"
    local configured_judge_base_url="${AREX_JUDGE_BASE_URL:-${JUDGE_BASE_URL:-}}"
    local judge_model="${configured_judge_model:-YOUR_JUDGE_MODEL}"
    local judge_base_url="${configured_judge_base_url:-http://judge.example/v1}"
    local judge_api_key_env="${AREX_JUDGE_API_KEY_ENV:-${JUDGE_API_KEY_ENV:-JUDGE_API_KEY}}"
    local data_root="${AREX_DATA_ROOT:-}"

    if (( dry_run == 0 )); then
        if [[ -z "${AREX_MODEL_NAME:-}" ]]; then
            printf 'Set AREX_MODEL_NAME before a real run.\n' >&2
            return 2
        fi
        if [[ -z "$configured_judge_model" || -z "$configured_judge_base_url" ]]; then
            printf 'Set AREX_JUDGE_MODEL and AREX_JUDGE_BASE_URL before a real run.\n' >&2
            return 2
        fi
        if [[ -z "${!model_api_key_env:-}" ]]; then
            printf 'Set the model key environment variable %s before a real run.\n' "$model_api_key_env" >&2
            return 2
        fi
        if [[ -z "${!judge_api_key_env:-}" ]]; then
            printf 'Set the judge key environment variable %s before a real run.\n' "$judge_api_key_env" >&2
            return 2
        fi
    fi

    local dataset_slug
    dataset_slug="$(printf '%s' "$dataset" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '-')"
    dataset_slug="${dataset_slug%-}"
    local save_path="${SAVE_PATH:-runs/${dataset_slug}-${start_index}-${num_tasks}}"

    local -a command=(
        python3 "$AREX_REPO_ROOT/evaluate.py" "$dataset"
        --profile refine-equal
        --concurrency "$concurrency"
        --n "$num_tasks"
        --start-index "$start_index"
        --model "$model_name"
        --api-key-env "$model_api_key_env"
        --judge-model "$judge_model"
        --judge-base-url "$judge_base_url"
        --judge-api-key-env "$judge_api_key_env"
        --save-path "$save_path"
    )

    if [[ -n "$base_url" ]]; then
        command+=(--base-url "$base_url")
    fi
    if [[ -n "$tokenizer_path" ]]; then
        command+=(--tokenizer-path "$tokenizer_path")
    fi
    if [[ -n "$data_root" ]]; then
        command+=(--data-root "$data_root")
    fi

    # Extra options are intentionally passed through. Typical examples are
    # `--dry-run` and repeated `--extra` evaluator flags. Secrets must still be
    # supplied through the *_API_KEY_ENV variables, never as option values.
    command+=("$@")
    (cd "$AREX_REPO_ROOT" && "${command[@]}")
}
