#!/usr/bin/env bash
# Setup for a rented GPU box with a CUDA PyTorch image. Idempotent; safe to re-run.
# Assumes a CUDA PyTorch base image (torch already present). Installs the rest.
set -euo pipefail

REPO_DIR="${AUDIT_REPO_DIR:-$HOME/open-weight-safety-audit}"
export AUDIT_OUTPUT_DIR="${AUDIT_OUTPUT_DIR:-$HOME/audit-outputs}"   # outside the repo
export AUDIT_LABELED_DIR="${AUDIT_LABELED_DIR:-$HOME/audit-labeled}"
export HF_HOME="${HF_HOME:-$HOME/hf}"
mkdir -p "$AUDIT_OUTPUT_DIR" "$AUDIT_LABELED_DIR" "$HF_HOME"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# transformers 4.x: Qwen3 needs >=4.51.0 and GLM-4-0414 (Glm4ForCausalLM) >=4.51.3.
# sentencepiece + protobuf: the HarmBench Llama-2 classifier tokenizer needs them.
pip install -q -U "transformers>=4.51.3,<5" accelerate bitsandbytes scipy pyarrow numpy picklescan hf_transfer sentencepiece protobuf

echo "setup OK: repo=$REPO_DIR out=$AUDIT_OUTPUT_DIR hf=$HF_HOME"
