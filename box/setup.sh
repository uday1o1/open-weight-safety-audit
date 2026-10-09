#!/usr/bin/env bash
# Provider-specific setup for a vast.ai GPU box. Idempotent; safe to re-run.
# Assumes a CUDA PyTorch base image (torch already present). Installs the rest.
set -euo pipefail

REPO_DIR="${AUDIT_REPO_DIR:-$HOME/open-weight-safety-audit}"
export AUDIT_OUTPUT_DIR="${AUDIT_OUTPUT_DIR:-$HOME/audit-outputs}"   # outside the repo
export AUDIT_LABELED_DIR="${AUDIT_LABELED_DIR:-$HOME/audit-labeled}"
export HF_HOME="${HF_HOME:-$HOME/hf}"
mkdir -p "$AUDIT_OUTPUT_DIR" "$AUDIT_LABELED_DIR" "$HF_HOME"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
pip install -q -U transformers accelerate bitsandbytes scipy pyarrow numpy picklescan

echo "setup OK: repo=$REPO_DIR out=$AUDIT_OUTPUT_DIR hf=$HF_HOME"
