#!/usr/bin/env bash
# Apply the local 3090 Ti patches on top of the HyperQwen-patched vLLM 0.29.0 venv
# (run after the upstream patches/series loop in docs/install.md).
set -euo pipefail
cd "$(dirname "$0")"
V=${VLLM_DIR:-../../venv/lib/python3.12/site-packages/vllm}
for p in 01-spec-attn-tuning.patch 02-unified-attn-sm86-prefill.patch 03-v2-sampler-small-topk.patch 04-split-kv-drafter-and-buffer-sizing.patch; do
  patch -p1 -N -d "$V" < "$p"
done
