#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -d "$REPO_ROOT/.bin" ]]; then
    export PATH="$REPO_ROOT/.bin:$PATH"
fi
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

CONFIG_PATH="$REPO_ROOT/evaluation/frontier/profiles/pi-codex-luna-base/harbor_pi_codex_luna_5h.conf"
CONCURRENCY_OVERRIDE=""
PROBLEMS_OVERRIDE=""
TRACK_OVERRIDE=""
DRY_RUN=0
FORCE_BUILD=0
RESUME_EXISTING_RUN=0

usage() {
    cat <<'EOF'
Usage:
  scripts/algorithmic/pi_concurrent.sh [OPTIONS]

Options:
  -c, --config FILE      Bash configuration file
  -t, --track TRACK      Override the Harbor track from the configuration
  -j, --concurrency N    Override the configured concurrency
  -p, --problems LIST    Override problem IDs/references, e.g. 0,1,2 or 2029I
      --force-build      Force image rebuilds; requires a single trial in the queue
      --resume-existing-run
                         Reuse an existing run's runtime configuration and add
                         missing trials
      --dry-run          Print commands only
  -h, --help             Show this help

Examples:
  scripts/algorithmic/pi_concurrent.sh --dry-run
  scripts/algorithmic/pi_concurrent.sh -j 4
  scripts/algorithmic/pi_concurrent.sh -p 0,1,2 -j 3
  scripts/algorithmic/pi_concurrent.sh -t codeforces -p 1772A -j 1
EOF
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

while (( $# > 0 )); do
    case "$1" in
        -c|--config)
            (( $# >= 2 )) || die "$1 requires a file"
            CONFIG_PATH="$2"
            shift 2
            ;;
        -j|--concurrency)
            (( $# >= 2 )) || die "$1 requires a number"
            CONCURRENCY_OVERRIDE="$2"
            shift 2
            ;;
        -t|--track)
            (( $# >= 2 )) || die "$1 requires a track"
            TRACK_OVERRIDE="$2"
            shift 2
            ;;
        -p|--problems)
            (( $# >= 2 )) || die "$1 requires a list"
            PROBLEMS_OVERRIDE="$2"
            shift 2
            ;;
        --force-build)
            FORCE_BUILD=1
            shift
            ;;
        --resume-existing-run)
            RESUME_EXISTING_RUN=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[[ -f "$CONFIG_PATH" ]] || die "config not found: $CONFIG_PATH"
# shellcheck source=/dev/null
source "$CONFIG_PATH"

TRACK="${TRACK:-algorithmic}"
SKILL_PATH="${SKILL_PATH:-}"

if ! declare -p PI_EXTRA_AGENT_ENVS >/dev/null 2>&1; then
    declare -a PI_EXTRA_AGENT_ENVS=()
elif [[ "$(declare -p PI_EXTRA_AGENT_ENVS 2>/dev/null)" != "declare -a "* ]]; then
    die "PI_EXTRA_AGENT_ENVS must be a Bash array"
fi

RUNS_DIR="${RUNS_DIR:-$REPO_ROOT/.frontier-cs/harbor}"
RUN_TAG="${RUN_TAG:-pi-su8}"
RUN_STAMP="${RUN_STAMP:-$(date '+%m%d-%H%M%S')}"
RUN_ID="${RUN_ID:-${RUN_STAMP}-${RUN_TAG}}"
TRIALS_DIR="${TRIALS_DIR:-$RUNS_DIR/$RUN_ID}"
PI_RUNTIME_SNAPSHOT_DIR="${PI_RUNTIME_SNAPSHOT_DIR:-$TRIALS_DIR/pi-runtime/initial}"
PI_RUNTIME_SNAPSHOT_FULL="${PI_RUNTIME_SNAPSHOT_FULL:-0}"
PI_PROVIDER_API_KEY_ENV="${PI_PROVIDER_API_KEY_ENV:-PI_PROVIDER_API_KEY}"
PI_AUTH_MODE="${PI_AUTH_MODE:-api_key}"
PI_CONFIG_DOCKER_DIR="${PI_CONFIG_DOCKER_DIR:-/root/.pi/agent}"
HARBOR_NO_GENERATE="${HARBOR_NO_GENERATE:-0}"
HARBOR_BIN="${HARBOR_BIN:-}"
PI_DOCKER_CLEANUP_ON_SUCCESS="${PI_DOCKER_CLEANUP_ON_SUCCESS:-0}"
PI_DOCKER_CLEANUP_SCRIPT="${PI_DOCKER_CLEANUP_SCRIPT:-}"

if [[ -n "$CONCURRENCY_OVERRIDE" ]]; then
    CONCURRENCY="$CONCURRENCY_OVERRIDE"
fi
if [[ -n "$TRACK_OVERRIDE" ]]; then
    TRACK="$TRACK_OVERRIDE"
fi
if [[ -n "$PROBLEMS_OVERRIDE" ]]; then
    mapfile -t PROBLEMS < <(
        printf '%s\n' "$PROBLEMS_OVERRIDE" | tr ',[:space:]' '\n' | awk 'NF'
    )
fi
[[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]] \
    || die "CONCURRENCY must be a positive integer"
case "$TRACK" in
    algorithmic|codeforces|atcoder|2.0) ;;
    *) die "unsupported TRACK: $TRACK" ;;
esac
[[ "$MODEL" == */* ]] || die "MODEL must use provider/model syntax"
[[ "$RUN_STAMP" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "RUN_STAMP contains unsupported characters"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "RUN_ID contains unsupported characters"
[[ "$HARBOR_NO_GENERATE" =~ ^[01]$ ]] \
    || die "HARBOR_NO_GENERATE must be 0 or 1"
if [[ -n "$HARBOR_BIN" && ! -x "$HARBOR_BIN" ]]; then
    die "HARBOR_BIN is not executable: $HARBOR_BIN"
fi
[[ "$PI_DOCKER_CLEANUP_ON_SUCCESS" =~ ^[01]$ ]] \
    || die "PI_DOCKER_CLEANUP_ON_SUCCESS must be 0 or 1"
[[ "$RESUME_EXISTING_RUN" =~ ^[01]$ ]] \
    || die "RESUME_EXISTING_RUN must be 0 or 1"
(( ${#PROBLEMS[@]} > 0 )) || die "PROBLEMS is empty"

for problem_id in "${PROBLEMS[@]}"; do
    [[ -n "$problem_id" && "$problem_id" != *$'\n'* ]] \
        || die "invalid problem ID/reference: $problem_id"
    if [[ "$TRACK" == "algorithmic" && ! "$problem_id" =~ ^[0-9]+$ ]]; then
        die "algorithmic problem ID must be numeric: $problem_id"
    fi
done
for extra_agent_env in "${PI_EXTRA_AGENT_ENVS[@]}"; do
    [[ "$extra_agent_env" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] \
        || die "invalid PI_EXTRA_AGENT_ENVS entry: $extra_agent_env"
done

provider="${MODEL%%/*}"
model_spec="${MODEL#*/}"
model_id="$model_spec"
[[ -f "$PI_AUTH_JSON_PATH" ]] || die "Pi auth file not found: $PI_AUTH_JSON_PATH"
[[ -f "$PI_MODELS_JSON_PATH" ]] \
    || die "Pi models file not found: $PI_MODELS_JSON_PATH"
[[ -f "$PI_SETTINGS_JSON_PATH" ]] \
    || die "Pi settings file not found: $PI_SETTINGS_JSON_PATH"
if [[ -n "$PI_SYSTEM_PROMPT_PATH" && ! -f "$PI_SYSTEM_PROMPT_PATH" ]]; then
    die "Pi system prompt not found: $PI_SYSTEM_PROMPT_PATH"
fi
[[ -d "$PI_EXTENSIONS_PATH" ]] \
    || die "Pi extensions directory not found: $PI_EXTENSIONS_PATH"
if [[ -n "$SKILL_PATH" ]]; then
    [[ -f "$SKILL_PATH/SKILL.md" ]] \
        || die "skill directory must contain SKILL.md: $SKILL_PATH"
fi
if [[ "$PI_EXTENSIONS_DOCKER_PATH" != /* \
    || "$PI_EXTENSIONS_DOCKER_PATH" == "/" ]]; then
    die "PI_EXTENSIONS_DOCKER_PATH must be an absolute, non-root path"
fi
if [[ "$PI_CONFIG_DOCKER_DIR" != /* \
    || "$PI_CONFIG_DOCKER_DIR" == "/" ]]; then
    die "PI_CONFIG_DOCKER_DIR must be an absolute, non-root path"
fi
if [[ "$PI_LOCAL_PACKAGES_DOCKER_DIR" != /* \
    || "$PI_LOCAL_PACKAGES_DOCKER_DIR" == "/" ]]; then
    die "PI_LOCAL_PACKAGES_DOCKER_DIR must be an absolute, non-root path"
fi
[[ "$PI_PROVIDER_API_KEY_ENV" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] \
    || die "PI_PROVIDER_API_KEY_ENV must be an environment variable name"
command -v uv >/dev/null 2>&1 || die "uv is required"
command -v jq >/dev/null 2>&1 || die "jq is required"
if (( PI_DOCKER_CLEANUP_ON_SUCCESS )); then
    [[ -n "$PI_DOCKER_CLEANUP_SCRIPT" ]] \
        || die "set PI_DOCKER_CLEANUP_SCRIPT when cleanup-on-success is enabled"
    [[ -x "$PI_DOCKER_CLEANUP_SCRIPT" ]] \
        || die "Docker cleanup script is not executable: $PI_DOCKER_CLEANUP_SCRIPT"
fi
case "$PI_AUTH_MODE" in
    api_key)
        jq -e --arg provider "$provider" '
            .[$provider].type == "api_key" and
            (.[$provider].key | type == "string" and length > 0)
        ' "$PI_AUTH_JSON_PATH" >/dev/null \
            || die "provider $provider needs a non-empty api_key credential in auth.json"
        ;;
    oauth)
        jq -e --arg provider "$provider" '
            .[$provider] | .type == "oauth" and
            (.access | type == "string" and length > 0) and
            (.refresh | type == "string" and length > 0) and
            (.expires | type == "number" and . > 0)
        ' "$PI_AUTH_JSON_PATH" >/dev/null \
            || die "provider $provider needs a complete oauth credential in auth.json; log in with Pi first"
        ;;
    *) die "PI_AUTH_MODE must be api_key or oauth" ;;
esac
if [[ "$model_spec" == *:* ]] && ! jq -e \
    --arg provider "$provider" \
    --arg model "$model_spec" \
    '.providers[$provider].models // [] | any(.id == $model)' \
    "$PI_MODELS_JSON_PATH" >/dev/null; then
    thinking_suffix="${model_spec##*:}"
    case "$thinking_suffix" in
        off|minimal|low|medium|high|xhigh|max)
            model_id="${model_spec%:*}"
            ;;
    esac
fi
api_key_reference="\$$PI_PROVIDER_API_KEY_ENV"
validate_runtime_model() {
    jq -e --arg provider "$provider" --arg model "$model_id" \
        --arg auth_mode "$PI_AUTH_MODE" --arg api_key_reference "$api_key_reference" '
        if $auth_mode == "oauth" then
            # Built-in OAuth models need no models.json override or API key.
            type == "object" and ((.providers // {}) | type == "object") and
            (.providers[$provider] == null or
                (.providers[$provider].apiKey == null and
                 (.providers[$provider].models // [] | any(.id == $model))))
        else
            (.providers[$provider].models // [] | any(.id == $model)) and
            .providers[$provider].apiKey == $api_key_reference
        end
    ' "$1" >/dev/null
}
validate_runtime_model "$PI_MODELS_JSON_PATH" \
    || die "models.json is inconsistent with $MODEL and PI_AUTH_MODE=$PI_AUTH_MODE"
printf -v provider_api_key_template '${%s}' "$PI_PROVIDER_API_KEY_ENV"

RESULTS_JSONL="${RESULTS_JSONL:-$TRIALS_DIR/results.jsonl}"

PI_EXTENSIONS_PATH="$(realpath "$PI_EXTENSIONS_PATH")"
if [[ -n "$SKILL_PATH" ]]; then
    SKILL_PATH="$(realpath "$SKILL_PATH")"
fi
MOUNTS_JSON="$(
    jq -cn \
        --arg extension_source "$PI_EXTENSIONS_PATH" \
        --arg extension_target "$PI_EXTENSIONS_DOCKER_PATH" \
        '[
            {
                type: "bind",
                source: $extension_source,
                target: $extension_target,
                read_only: true
            }
        ]'
)"
append_read_only_mount() {
    local source="$1"
    local target="$2"
    MOUNTS_JSON="$(
        jq -cn \
            --argjson mounts "$MOUNTS_JSON" \
            --arg source "$source" \
            --arg target "$target" \
            '$mounts + [{
                type: "bind",
                source: $source,
                target: $target,
                read_only: true
            }]'
    )"
}
LOCAL_PACKAGES_DOCKER_JSON='[]'
LOCAL_PACKAGES_HOST_JSON='[]'
declare -A SEEN_PACKAGE_TARGETS=()
for package_path in "${PI_LOCAL_PACKAGE_PATHS[@]}"; do
    [[ -d "$package_path" ]] || die "local Pi package not found: $package_path"
    package_path="$(realpath "$package_path")"
    package_name="$(basename "$package_path")"
    package_target="${PI_LOCAL_PACKAGES_DOCKER_DIR%/}/$package_name"
    if [[ -n "${SEEN_PACKAGE_TARGETS[$package_target]:-}" ]]; then
        die "duplicate local Pi package target: $package_target"
    fi
    SEEN_PACKAGE_TARGETS[$package_target]=1
    MOUNTS_JSON="$(
        jq -cn \
            --argjson mounts "$MOUNTS_JSON" \
            --arg source "$package_path" \
            --arg target "$package_target" \
            '$mounts + [{
                type: "bind",
                source: $source,
                target: $target,
                read_only: true
            }]'
    )"
    LOCAL_PACKAGES_DOCKER_JSON="$(
        jq -cn \
            --argjson packages "$LOCAL_PACKAGES_DOCKER_JSON" \
            --arg target "$package_target" \
            '$packages + [$target]'
    )"
    LOCAL_PACKAGES_HOST_JSON="$(
        jq -cn \
            --argjson packages "$LOCAL_PACKAGES_HOST_JSON" \
            --arg source "$package_path" \
            '$packages + [$source]'
    )"
done

job_count=${#PROBLEMS[@]}
if (( FORCE_BUILD == 1 && job_count != 1 )); then
    die "--force-build requires exactly one queued trial"
fi

RUNTIME_DIR=""
cleanup() {
    if [[ -n "$RUNTIME_DIR" && -d "$RUNTIME_DIR" ]]; then
        rm -rf -- "$RUNTIME_DIR"
    fi
}
trap cleanup EXIT

if (( DRY_RUN )); then
    RUNTIME_AUTH_JSON_PATH="<temporary-pi-auth.json>"
    RUNTIME_MODELS_JSON_PATH="<run-pi-models.json>"
    RUNTIME_SETTINGS_JSON_PATH="<run-pi-settings.json>"
    RUNTIME_SYSTEM_PROMPT_PATH="<run-pi-SYSTEM.md>"
else
    if (( RESUME_EXISTING_RUN )); then
        [[ -d "$TRIALS_DIR" ]] \
            || die "resume run directory not found: $TRIALS_DIR"
    else
        [[ ! -e "$TRIALS_DIR" ]] \
            || die "run directory already exists: $TRIALS_DIR"
    fi
    RUNTIME_DIR="$(mktemp -d)"
    chmod 700 "$RUNTIME_DIR"
    RUNTIME_AUTH_JSON_PATH="$RUNTIME_DIR/auth.json"
    jq --arg provider "$provider" '{($provider): .[$provider]}' \
        "$PI_AUTH_JSON_PATH" > "$RUNTIME_AUTH_JSON_PATH"
    chmod 600 "$RUNTIME_AUTH_JSON_PATH"

    if [[ "$PI_AUTH_MODE" == "api_key" ]]; then
        provider_api_key="$(
            jq -er --arg provider "$provider" '.[$provider].key' \
                "$PI_AUTH_JSON_PATH"
        )" || die "failed to read the $provider API key from auth.json"
        printf -v "$PI_PROVIDER_API_KEY_ENV" '%s' "$provider_api_key"
        export "$PI_PROVIDER_API_KEY_ENV"
        unset provider_api_key
    fi

    RUNTIME_MODELS_JSON_PATH="$TRIALS_DIR/pi-runtime/container-config/models.json"
    RUNTIME_SETTINGS_JSON_PATH="$TRIALS_DIR/pi-runtime/container-config/settings.json"
    RUNTIME_SYSTEM_PROMPT_PATH="$TRIALS_DIR/pi-runtime/container-config/SYSTEM.md"
    if (( RESUME_EXISTING_RUN )); then
        [[ -f "$RUNTIME_MODELS_JSON_PATH" ]] \
            || die "resume models file not found: $RUNTIME_MODELS_JSON_PATH"
        [[ -f "$RUNTIME_SETTINGS_JSON_PATH" ]] \
            || die "resume settings file not found: $RUNTIME_SETTINGS_JSON_PATH"
        if [[ -n "$PI_SYSTEM_PROMPT_PATH" ]]; then
            [[ -f "$RUNTIME_SYSTEM_PROMPT_PATH" ]] \
                || die "resume system prompt not found: $RUNTIME_SYSTEM_PROMPT_PATH"
        fi
        validate_runtime_model "$RUNTIME_MODELS_JSON_PATH" \
            || die "resume runtime model $MODEL is inconsistent"
    else
        mkdir -p "$TRIALS_DIR/pi-runtime/container-config"
        chmod 700 "$TRIALS_DIR/pi-runtime/container-config"
        cp "$PI_MODELS_JSON_PATH" "$RUNTIME_MODELS_JSON_PATH"
        chmod 600 "$RUNTIME_MODELS_JSON_PATH"
        jq --argjson local_packages "$LOCAL_PACKAGES_DOCKER_JSON" \
            '.packages = ((.packages // []) + $local_packages)' \
            "$PI_SETTINGS_JSON_PATH" > "$RUNTIME_SETTINGS_JSON_PATH"
        chmod 600 "$RUNTIME_SETTINGS_JSON_PATH"
        if [[ -n "$PI_SYSTEM_PROMPT_PATH" ]]; then
            cp "$PI_SYSTEM_PROMPT_PATH" "$RUNTIME_SYSTEM_PROMPT_PATH"
            chmod 600 "$RUNTIME_SYSTEM_PROMPT_PATH"
        fi
    fi
fi

append_read_only_mount \
    "$RUNTIME_MODELS_JSON_PATH" \
    "${PI_CONFIG_DOCKER_DIR%/}/models.json"
append_read_only_mount \
    "$RUNTIME_SETTINGS_JSON_PATH" \
    "${PI_CONFIG_DOCKER_DIR%/}/settings.json"
if [[ -n "$PI_SYSTEM_PROMPT_PATH" ]]; then
    append_read_only_mount \
        "$RUNTIME_SYSTEM_PROMPT_PATH" \
        "${PI_CONFIG_DOCKER_DIR%/}/SYSTEM.md"
fi

build_command() {
    local problem_id="$1"
    local problem_slug track_slug
    local extra_agent_env
    local -a THINKING_ARGS=()
    local -a AUTH_ARGS=()
    local -a SKILL_ARGS=()
    local -a HARBOR_BIN_ARGS=()
    if [[ "$PI_AUTH_MODE" == "api_key" ]]; then
        AUTH_ARGS=(--agent-env "$PI_PROVIDER_API_KEY_ENV=$provider_api_key_template")
    fi
    if [[ -n "${THINKING:-}" ]]; then
        THINKING_ARGS=(--agent-kwarg "thinking=$THINKING")
    fi
    if [[ -n "$SKILL_PATH" ]]; then
        SKILL_ARGS=(--skill "$SKILL_PATH")
    fi
    if [[ -n "$HARBOR_BIN" ]]; then
        HARBOR_BIN_ARGS=(--harbor-bin "$HARBOR_BIN")
    fi
    problem_slug="${problem_id//[^A-Za-z0-9._-]/-}"
    problem_slug="${problem_slug//_/-}"
    problem_slug="${problem_slug,,}"
    track_slug="${TRACK//[^A-Za-z0-9._-]/-}"
    track_slug="${track_slug,,}"
    if [[ "$TRACK" == "algorithmic" ]]; then
        TRIAL_NAME="frontier-cs-algorithm-${problem_slug}__${RUN_STAMP}"
    else
        TRIAL_NAME="frontier-cs-${track_slug}-${problem_slug}__${RUN_STAMP}"
    fi
    COMMAND=(
        uv run frontier harbor trial "$TRACK" "$problem_id"
        -a pi
        -m "$MODEL"
        --agent-timeout "$AGENT_TIMEOUT"
        "${HARBOR_BIN_ARGS[@]}"
        "${THINKING_ARGS[@]}"
        "${AUTH_ARGS[@]}"
        "${SKILL_ARGS[@]}"
        --trials-dir "$TRIALS_DIR"
        --trial-name "$TRIAL_NAME"
        --env "PI_AUTH_JSON_PATH=$RUNTIME_AUTH_JSON_PATH"
        --env "PI_MODELS_JSON_PATH=$RUNTIME_MODELS_JSON_PATH"
        --env "PI_SETTINGS_JSON_PATH=$RUNTIME_SETTINGS_JSON_PATH"
        --env "PI_EXTENSIONS_DIR=$PI_EXTENSIONS_PATH"
        --env "PI_LOCAL_PACKAGES_JSON=$LOCAL_PACKAGES_HOST_JSON"
        --env "PI_SKILL_DIR=$SKILL_PATH"
        --env "PI_RUNTIME_SNAPSHOT_DIR=$PI_RUNTIME_SNAPSHOT_DIR"
        --env "PI_RUNTIME_SNAPSHOT_FULL=$PI_RUNTIME_SNAPSHOT_FULL"
        --mounts "$MOUNTS_JSON"
        --keep-container
        --json
        --verbose
        --jsonl "$RESULTS_JSONL"
    )
    for extra_agent_env in "${PI_EXTRA_AGENT_ENVS[@]}"; do
        COMMAND+=(--agent-env "$extra_agent_env")
    done
    if [[ -n "$PI_SYSTEM_PROMPT_PATH" ]]; then
        COMMAND+=(--env "PI_SYSTEM_PROMPT_PATH=$RUNTIME_SYSTEM_PROMPT_PATH")
    fi
    if (( HARBOR_NO_GENERATE )); then
        COMMAND+=(--no-generate)
    fi
    if (( FORCE_BUILD )); then
        COMMAND+=(--force-build)
    fi
}

printf 'Config: %s\nRun: %s\nTrack: %s\nModel: %s\nProblems: %d\nConcurrency: %d\nArtifacts: %s\nJSONL: %s\n' \
    "$CONFIG_PATH" "$RUN_ID" "$TRACK" "$MODEL" "${#PROBLEMS[@]}" \
    "$CONCURRENCY" "$TRIALS_DIR" "$RESULTS_JSONL"

if (( DRY_RUN )); then
    for problem_id in "${PROBLEMS[@]}"; do
        build_command "$problem_id"
        printf '[dry-run problem=%s] ' "$problem_id"
        printf '%q ' "${COMMAND[@]}"
        printf '\n'
    done
    exit 0
fi

if (( RESUME_EXISTING_RUN )); then
    declare -a pending_problems=()
    for problem_id in "${PROBLEMS[@]}"; do
        build_command "$problem_id"
        trial_name="$TRIAL_NAME"
        if [[ -e "$TRIALS_DIR/$trial_name" ]]; then
            printf '[resume-skip] problem=%s reason=trial-exists trial=%s\n' \
                "$problem_id" "$trial_name"
            continue
        fi
        if [[ -f "$RESULTS_JSONL" ]] && jq -e --arg trial_name "$trial_name" \
            'select(.trial_name == $trial_name)' "$RESULTS_JSONL" >/dev/null; then
            printf '[resume-skip] problem=%s reason=result-exists trial=%s\n' \
                "$problem_id" "$trial_name"
            continue
        fi
        pending_problems+=("$problem_id")
    done
    PROBLEMS=("${pending_problems[@]}")
    job_count=${#PROBLEMS[@]}
    if (( job_count == 0 )); then
        printf 'Resume found no pending trials: %s\n' "$TRIALS_DIR"
        exit 0
    fi
    printf 'Resume will skip existing trials and dispatch only %d pending problems.\n' "$job_count"
fi

validate_trial_success() {
    local trial_dir="$1"
    local exception_file="$trial_dir/exception.txt"
    local pi_log="$trial_dir/agent/pi.txt"
    local result_file="$trial_dir/result.json"
    local terminal_event

    if [[ ! -f "$result_file" ]]; then
        printf '[invalid] missing result.json: %s\n' "$trial_dir" >&2
        return 1
    fi
    if [[ -s "$exception_file" ]]; then
        printf '[invalid] exception.txt is not empty: %s\n' "$exception_file" >&2
        return 1
    fi
    if ! jq -e '
        (.exception_info == null) and
        (.verifier_result.rewards.reward != null)
    ' "$result_file" >/dev/null; then
        printf '[invalid] exception_info is not null or verifier reward is missing: %s\n' \
            "$result_file" >&2
        return 1
    fi
    if [[ -f "$pi_log" ]]; then
        if ! terminal_event="$(
            jq -r '
                if .type == "agent_end" then
                    "agent_end"
                elif .type == "auto_retry_end" and .success == false then
                    "auto_retry_failed"
                else
                    empty
                end
            ' "$pi_log" | tail -n 1
        )"; then
            printf '[invalid] failed to parse Pi log: %s\n' "$pi_log" >&2
            return 1
        fi
        if [[ "$terminal_event" == "auto_retry_failed" ]]; then
            printf '[invalid] Pi automatic retries exhausted: %s\n' "$pi_log" >&2
            return 1
        fi
    fi
}

run_one() {
    local problem_id="$1"
    local rc trial_dir
    build_command "$problem_id"
    trial_dir="$TRIALS_DIR/$TRIAL_NAME"
    printf '[start] problem=%s trial=%s\n' "$problem_id" "$TRIAL_NAME"
    if (cd "$REPO_ROOT" && "${COMMAND[@]}"); then
        rc=0
    else
        rc=$?
    fi
    if (( rc == 0 )) && validate_trial_success "$trial_dir"; then
        if (( PI_DOCKER_CLEANUP_ON_SUCCESS )); then
            if ! "$PI_DOCKER_CLEANUP_SCRIPT" \
                --trial-dir "$trial_dir" --once; then
                printf '[warning] Docker cleanup failed; preserving successful trial status: %s\n' \
                    "$trial_dir" >&2
            fi
        fi
        printf '[success] problem=%s trial=%s\n' "$problem_id" "$TRIAL_NAME"
        return 0
    fi
    if (( rc == 0 )); then
        rc=1
    fi
    printf '[failure] problem=%s trial=%s exit=%d\n' \
        "$problem_id" "$TRIAL_NAME" "$rc" >&2
    return "$rc"
}

declare -A ACTIVE_JOBS=()
failures=0
launched=0
GRACEFUL_STOP_REQUESTED=0

reap_one() {
    local finished_pid rc label
    finished_pid=""
    if wait -n -p finished_pid; then
        rc=0
    else
        rc=$?
    fi
    # A handled USR1 interrupts wait(1) without completing a child. Keep the
    # active-job table intact; the drain loop will wait for it again.
    if [[ -z "${finished_pid:-}" ]]; then
        return 0
    fi
    label="${ACTIVE_JOBS[$finished_pid]:-unknown}"
    unset 'ACTIVE_JOBS[$finished_pid]'
    if (( rc != 0 )); then
        failures=$((failures + 1))
        printf '[failed] %s exit=%d\n' "$label" "$rc" >&2
    fi
}

request_graceful_stop() {
    GRACEFUL_STOP_REQUESTED=1
    printf 'Received USR1: stopping new trial dispatch and waiting for %d active trials to finish.\n' \
        "${#ACTIVE_JOBS[@]}" >&2
}
trap request_graceful_stop USR1

stop_children() {
    trap - INT TERM
    printf 'Interrupt received; stopping child tasks.\n' >&2
    for pid in "${!ACTIVE_JOBS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait || true
    exit 130
}
trap stop_children INT TERM

for problem_id in "${PROBLEMS[@]}"; do
    if (( GRACEFUL_STOP_REQUESTED )); then
        break
    fi
    while (( ${#ACTIVE_JOBS[@]} >= CONCURRENCY )); do
        reap_one
        if (( GRACEFUL_STOP_REQUESTED )); then
            break
        fi
    done
    if (( GRACEFUL_STOP_REQUESTED )); then
        break
    fi
    run_one "$problem_id" &
    ACTIVE_JOBS[$!]="problem=$problem_id"
    launched=$((launched + 1))
done
while (( ${#ACTIVE_JOBS[@]} > 0 )); do
    reap_one
done

if (( GRACEFUL_STOP_REQUESTED )); then
    printf 'Run paused gracefully: planned=%d, launched=%d, failures=%d, artifacts=%s\n' \
        "$job_count" "$launched" "$failures" "$TRIALS_DIR" >&2
    exit 75
fi

if (( failures > 0 )); then
    printf 'Run failed: total=%d, failures=%d, artifacts=%s\n' \
        "$job_count" "$failures" "$TRIALS_DIR" >&2
    exit 1
fi
printf 'Run succeeded: total=%d, failures=0, artifacts=%s\n' \
    "$job_count" "$TRIALS_DIR"
