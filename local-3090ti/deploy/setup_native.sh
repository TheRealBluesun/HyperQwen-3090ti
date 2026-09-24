#!/usr/bin/env bash
# Native (venv) HyperQwen install, per docs/install.md, using uv (no python3-venv on this host).
set -euo pipefail
cd ~/build/HyperQwen
UV=~/.local/bin/uv
$UV venv --seed -p 3.12 venv
VIRTUAL_ENV=$PWD/venv $UV pip install vllm==0.29.0 huggingface_hub hf_transfer ninja \
  --extra-index-url https://flashinfer.ai/whl/ flashinfer-cubin==0.6.18 pandas --index-strategy unsafe-best-match
echo "== pip done"
HF_XET_HIGH_PERFORMANCE=1 venv/bin/hf download dbirks/Qwen3.8-27B-W4A16-AutoRound \
  --local-dir models/Qwen3.8-27B-W4A16-AutoRound
echo "== download done"
