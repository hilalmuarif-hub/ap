#!/usr/bin/env bash
# daily.sh — Orchestrator for the AP Army anti-piracy pipeline.
#
# Runs once per day via GitHub Actions cron (see .github/workflows/daily.yml).
# All business logic lives in pipeline.py; this script handles:
#   - Environment setup (venv, .env, credential files)
#   - Run ID and log directory management
#   - Exit-code-based step gating (abort on critical, continue on warning)
#   - Log archival
#
# Usage:
#   ./daily.sh [--dry-run] [--platform <name>]
#
#   --dry-run          Crawl and score but skip all writes to Google Sheets
#   --platform <name>  Limit crawl to one platform (e.g. --platform facebook)
#
# Environment variables (sourced from .env if present):
#   GOOGLE_SHEETS_ID          Spreadsheet ID for the operations dashboard
#   GOOGLE_SERVICE_ACCOUNT    Path to service account JSON (default: service_account.json)
#   FB_COOKIE_FILE            Path to Facebook cookies JSON
#   SLACK_WEBHOOK_URL         Slack incoming webhook URL for canary alerts
#   ENABLED_PLATFORMS         Comma-separated platform list (default: facebook)
#   VENV_DIR                  Python venv directory (default: .venv)

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths and run identity
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/${VENV_DIR:-.venv}"
LOG_DIR="${SCRIPT_DIR}/logs"
STATS_DIR="${SCRIPT_DIR}/stats"
RUN_ID="run_$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
STATS_FILE="${STATS_DIR}/${RUN_ID}.json"

# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------

DRY_RUN=false
PLATFORM_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --platform)
            PLATFORM_ARGS+=("--platform" "$2")
            shift 2
            ;;
        *)
            echo "Unknown flag: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG_FILE}"
}

# Run a critical step: abort the pipeline on non-zero exit.
run_critical() {
    local step_name="$1"
    shift
    log ">>> ${step_name}"
    if ! "$@" 2>&1 | tee -a "${LOG_FILE}"; then
        log "ABORT: ${step_name} failed — pipeline cannot continue."
        # Best-effort canary alert on abort (may fail if creds are the problem)
        python "${SCRIPT_DIR}/pipeline.py" canary \
            --run-id "${RUN_ID}" \
            --stats "${STATS_FILE}" 2>>"${LOG_FILE}" || true
        exit 1
    fi
}

# Run a non-critical step: log errors but let the pipeline continue.
run_noncritical() {
    local step_name="$1"
    shift
    log ">>> ${step_name}"
    if ! "$@" 2>&1 | tee -a "${LOG_FILE}"; then
        log "WARNING: ${step_name} failed — continuing pipeline."
    fi
}

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

setup_env() {
    log "========================================"
    log "  AP Army Pipeline — ${RUN_ID}"
    log "========================================"

    # Create required directories
    mkdir -p "${LOG_DIR}" "${STATS_DIR}"

    # Source .env file if present (local development)
    if [[ -f "${SCRIPT_DIR}/.env" ]]; then
        log "Sourcing .env"
        # shellcheck disable=SC1091
        set -a
        source "${SCRIPT_DIR}/.env"
        set +a
    fi

    # Activate Python virtual environment if present
    if [[ -f "${VENV_DIR}/bin/activate" ]]; then
        log "Activating venv: ${VENV_DIR}"
        # shellcheck disable=SC1091
        source "${VENV_DIR}/bin/activate"
    elif [[ -f "${VENV_DIR}/Scripts/activate" ]]; then
        # Windows Git Bash / MSYS2
        # shellcheck disable=SC1091
        source "${VENV_DIR}/Scripts/activate"
    fi

    log "Python: $(python --version 2>&1)"
    log "DRY_RUN=${DRY_RUN}"
    log "Platforms: ${ENABLED_PLATFORMS:-facebook}"
}

# ---------------------------------------------------------------------------
# Step 1: Pre-flight check (critical)
# ---------------------------------------------------------------------------

step_preflight() {
    run_critical "Pre-flight" \
        python "${SCRIPT_DIR}/pipeline.py" preflight
}

# ---------------------------------------------------------------------------
# Steps 2-9: Main pipeline run (critical)
#
# pipeline.py run handles:
#   2. normalize_query  — expand titles to queries
#   3. detection        — crawl platforms
#   4. identity         — resolve permanent IDs
#   5. dedupe_cluster   — deduplicate results
#   6. evidence_scorer  — score 0-100
#   7. offender_registry— update offender database  (skipped if --dry-run)
#   8. sheet_writer     — write to Google Sheets    (skipped if --dry-run)
# ---------------------------------------------------------------------------

step_run_pipeline() {
    local dry_flag=()
    [[ "${DRY_RUN}" == "true" ]] && dry_flag=("--dry-run")

    run_critical "Pipeline run" \
        python "${SCRIPT_DIR}/pipeline.py" run \
            --run-id "${RUN_ID}" \
            --stats-output "${STATS_FILE}" \
            "${dry_flag[@]}" \
            "${PLATFORM_ARGS[@]}"
}

# ---------------------------------------------------------------------------
# Step 9: Post-run canary (non-critical — always runs, even after failures)
# ---------------------------------------------------------------------------

step_postrun_canary() {
    run_noncritical "Post-run canary" \
        python "${SCRIPT_DIR}/pipeline.py" canary \
            --run-id "${RUN_ID}" \
            --stats "${STATS_FILE}"
}

# ---------------------------------------------------------------------------
# Cleanup: remove credentials from disk (security hygiene in CI)
# ---------------------------------------------------------------------------

cleanup_credentials() {
    # Only wipe if running in CI (GitHub Actions sets CI=true)
    if [[ "${CI:-false}" == "true" ]]; then
        log "Removing credential files (CI environment)"
        rm -f \
            "${SCRIPT_DIR}/service_account.json" \
            "${SCRIPT_DIR}/fb_cookies.json"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    setup_env

    # Pre-flight must pass before the expensive crawl begins
    step_preflight

    # Main pipeline run (steps 2-8)
    step_run_pipeline

    log "========================================"
    log "  Pipeline complete: ${RUN_ID}"
    log "========================================"

    # Post-run canary runs regardless of prior step results (trap ensures this)
    step_postrun_canary
}

# Ensure canary and credential cleanup run even if script exits early
trap 'step_postrun_canary; cleanup_credentials' EXIT

main "$@"
