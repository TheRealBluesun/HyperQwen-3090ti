# Qwen3.8-27B on a home inference server, :8001 — baseline notes (2026-09-24)

Hardware: GPU0 RTX 3090 Ti (24 GB, power-capped to 300 W on purpose: both cards share an 850 W PSU),
i9-9900K, 31 GB RAM. :8002 (GPU1, RTX 3090) serves the same model for another workload (since migrated to the same setup).
Requirement from the user: at least 128K context in every configuration.

## Baseline chosen: HyperQwen single-user, CTX=long (150K), SPEC=mtp
- Repo: ~/build/HyperQwen (github.com/syv-ai/HyperQwen @ 1cf8665), native venv via uv (host lacks
  python3-venv), vLLM 0.29.0 + the repo's patch stack, model dbirks/Qwen3.8-27B-W4A16-AutoRound
  (+ repo requant: int8 lm_head/embeddings, MTP, 40K draft vocab, int4-GPTQ "fast" variant, DFlash2 drafter).
- Services (since 09-24 evening: SPEC=dflash2 CTX=long, 128K, both GPUs at 350 W): systemd user units `hyperqwen-a.service` (GPU0, :8001) and `hyperqwen-b.service` (GPU1, :8002) -> local-3090ti/deploy/run.sh (
  streamed weight loading, names `Qwen3.8-27B` + `qwen3.8-27b`, thinking off by default).
  `buun-27b-a` (old llama.cpp fork setup) is disabled but installed. Rollback:
  `systemctl --user disable --now hyperqwen-a && systemctl --user enable --now buun-27b-a`.
- Tool calling (OpenAI, qwen3_coder parser) and Anthropic /v1/messages with tools both work.

## Results (bench27.py: streamed, engine-neutral; medians of 3; greedy decode tok/s)

| | buun-llama-cpp EXL3 4.0bpw + DFlash2 (old) | HyperQwen CTX=fast (64K, ref only) | **HyperQwen CTX=long MTP (150K)** | HyperQwen CTX=long DFlash2 (128K) |
|---|---|---|---|---|
| code / prose / explain / json | 118 / 53 / 74 / 163 | 180 / 111 / 147 / 200 | **127 / 88 / 109 / 140** | 230 / 106 / 163 / 264 |
| default sampling code / prose / explain / json | 116 / 58 / 91 / 157 | 163 / 102 / 139 / 192 | 115 / 85 / 106 / 134 | 196 / 101 / 161 / 259 |
| prefill, 18K prompt | ~650 tok/s | 1,222 | **1,181** | 956 |
| 109K prompt: prefill time / tok/s | 217 s / 505 | — | **134 s / 816** | 277 s / 395 |
| decode after the 109K prompt | 41 | — | **75** | 52 |
| context | 2 x 131K | 64K | 150K (202K pool) | 128K (136K pool) |

Raw runs in results/. DFlash2-long is 1.5–2x faster at short context but slower deep in the context;
closing that gap (e.g. a better long-prefill path under DFlash2) is an obvious optimization target.

## Gotchas hit
- FlashInfer JIT (fp8 KV in CTX=long) needs nvcc == the CUDA runtime headers. The venv pulled
  nvcc/crt/nvvm 13.4 + cccl 13.3 against runtime 13.0 -> "CUDA compiler and CUDA toolkit headers are
  incompatible" / "Unsupported .version 9.4". Fix: `uv pip install "nvidia-cuda-nvcc==13.0.*"
  "nvidia-cuda-crt==13.0.*" "nvidia-nvvm==13.0.*" "nvidia-cuda-cccl==13.0.*"` into the venv.
- Docker route not used: no NVIDIA container runtime, and restarting dockerd would kill the
  long-running StigmergyBench containers.
- My early spot checks of the old setup (llama.cpp `timings`, 22–46 tok/s prose) were disturbed by
  something else; bench27.py's streamed medians are the reliable baseline.

## Step 1: traffic + profile (2026-09-24 evening)

Traffic (buun logs, 7 days, 2,748 requests): total context per request median 3.3K, p75 5.1K, p90 14K,
p99 85K; >32K 7%, >64K 2.7%, >128K 1 request. New prompt tokens after prefix cache: median 960, p90 5K.
Output: median 300 tokens, p90 1.3K. Both slots used (some concurrency). => 128K covers everything;
short-context decode dominates; long context matters for ~7% of requests.

Profile: HyperQwen SPEC=dflash2 CTX=long (128K, int8 KV), nsys (copied to ~/tools/nsight-systems on .15),
nsys via `deploy/dev_prof.sh` (capture ranges through /start_profile and /stop_profile).
- Short-context decode (300 tokens): GPU 95% busy, 75 target steps (4.0 tokens/step), ~24 ms/step.
  Marlin W4A16 GEMMs 18.5 ms/step (~75–80% of DRAM bandwidth for ~14.3 GB of target + drafter weights
  per step); GDN recurrence ~1 ms; a 128 µs bf16 GEMM per step; small elementwise/norms; ~1.2 ms idle.
- 109K prefill (270 s): kernel_unified_attention (Triton prefill attention, int8 KV) **190 s (70%)**;
  Marlin 75 s (≈ the 3090 Ti's fp32-accumulate tensor-core limit at 300 W). Rough limit for the
  attention: ~2.4 PFLOP -> ~30 s at peak, ~60 s for an FA2-class kernel.
- Decode at 109K: 53 ms/step: `_spec_attn_partial` (split-KV verify attention, int8 KV) **30 ms/step**
  (1.87 ms per layer vs ~0.22 ms to read the 224 MB of int8 K+V per layer: ~12% of bandwidth);
  Marlin 17.8 ms/step.
  Cause: NUM_SEGMENTS=16 fixed -> grid 1 x 4 KV heads x 16 = 64 CTAs on 84 SMs, num_stages=1,
  TILE=32 at D=256: each CTA walks ~210 tiles serially, latency-bound (~116 GB/s total).
