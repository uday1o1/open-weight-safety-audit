#!/usr/bin/env bash
# Run the full audit detached; survives disconnects. Logs outside the repo checkout.
set -euo pipefail
REPO_DIR="${AUDIT_REPO_DIR:-$HOME/open-weight-safety-audit}"
export AUDIT_OUTPUT_DIR="${AUDIT_OUTPUT_DIR:-$HOME/audit-outputs}"
export AUDIT_LABELED_DIR="${AUDIT_LABELED_DIR:-$HOME/audit-labeled}"
export HF_HOME="${HF_HOME:-$HOME/hf}"
export AUDIT_BATCH="${AUDIT_BATCH:-8}"        # conservative for a 24 GB card
export PYTHONPATH="$REPO_DIR/src"
LOG="$HOME/audit-run.log"
cd "$REPO_DIR"
setsid nohup python -u src/run_audit.py "${1:-all}" >"$LOG" 2>&1 &
disown
echo "started run_audit (${1:-all}); log: $LOG"
