#!/usr/bin/env bash
# Apply the local 3090 Ti patches on top of the HyperQwen-patched vLLM 0.29.0 venv
# (run after the upstream patches/series loop in docs/install.md).
set -euo pipefail
cd "$(dirname "$0")"
V=${VLLM_DIR:-../../venv/lib/python3.12/site-packages/vllm}
for p in 01-spec-attn-tuning.patch 02-unified-attn-sm86-prefill.patch 03-v2-sampler-small-topk.patch 04-split-kv-drafter-and-buffer-sizing.patch; do
  patch -p1 -N -d "$V" < "$p"
done
# 06: CUDA split-KV verify attention (sm86). The kernel source is JIT-built on first use.
patch -p1 -N -d "$V" < 06-cuda-verify-attention.patch
cp 06-qwen27_sda.cu "$V/v1/attention/ops/qwen27_sda.cu"
# 07: GDN attention metadata built once per step (shared across the 6 GDN kv-cache groups)
patch -p1 -N -d "$V" < 07-gdn-shared-metadata-build.patch
# 09: small-batch silu_and_mul (Triton custom op)
patch -p1 -N -d "$V" < 09-small-silu-and-mul.patch
cp 09-qwen27_small_ops.py "$V/model_executor/layers/qwen27_small_ops.py"
# 10: GDN spec decode reads q/k/v in place
patch -p1 -N -d "$V" < 10-gdn-strided-qkv.patch
# 11: bf16-cache (CTX=fast) robustness for the split-KV path
patch -p1 -N -d "$V" < 11-bf16-cache-robustness.patch
# 12: prefill attention through the CUDA kernel (direct mode)
patch -p1 -N -d "$V" < 12-prefill-attention-cuda.patch
# 13: conv1d spec-decode guard bounded by conv-state capacity, not this step's length (fixes
#     corrupt output when a verify block is shorter than the previous one: sync scheduling + grammar,
#     adaptive verify length, DFLASH_TOKENS>7)
patch -p1 -N -d "$V" < 13-conv1d-spec-varlen-guard.patch
# 14: lazy GDN state commit for spec decode (opt-in: VLLM_QWEN27_LAZY_GDN=1). New recurrence kernel (JIT-built on
#     first use) that stores 1 state + a token log per step instead of 8 states; bit-identical states.
patch -p1 -N -d "$V" < 14-lazy-gdn-state-commit.patch
cp 14-qwen27_lazy_gdn.py "$V/model_executor/layers/mamba/gdn/qwen27_lazy_gdn.py"
cp 14-qwen27_gdn_rec.cu "$V/model_executor/layers/mamba/gdn/qwen27_gdn_rec.cu"
