# Qwen3.8-27B (HyperQwen, DFlash2 128K) on RTX 3090 / 3090 Ti — status and to-do

As of 2026-09-24. Measurements: PROGRESS.md; setup: NOTES.md. Both GPUs at 350 W (the decode knee;
above that decode is bandwidth-bound again).

## Where we are
| Endpoint | GPU | ms/step (short ctx) | greedy code / prose / explain / json (tok/s) | prefill 18K | 109K prefill / decode after |
|---|---|---|---|---|---|
| :8001 | 3090 Ti (1,008 GB/s) | 22.0 | 252 / 131 / 182 / 290 | 1,237 tok/s | not re-measured at 350 W |
| :8002 | 3090 (936 GB/s) | 23.8 | 230 / 114 / 174 / 268 | 1,148 tok/s | 168 s / 65 tok/s |
Concurrency (:8002): 1 / 2 / 4 streams → 183 / 164 / 100 tok/s per stream, 176 / 255 / 317 aggregate.
Versus the old llama.cpp EXL3 setup: ~1.6–2.3x on typical prompts; 109K ready ~50 s sooner.

## How close to the hardware (re-profile of :8002 at 350 W)
- Short-context decode moves ~15.2 GB per verify step (12.6 GB int4 decoder, 1.3 GB drafter, 0.7 GB
  lm_head, ~0.7 GB GDN state traffic): **~68% of peak bandwidth** end to end. Realistic floor at ~92%:
  ~17.7 ms/step on the 3090 (~16.4 ms on the Ti) → **max ~1.35x left**.
- Step breakdown: int4 GEMMs 79% (running at ~79% of peak), GDN recurrence+conv 5%, small bf16 GEMMs 4%,
  norms/activations 3%, glue 2%, attention 2%, drafter/sampling 1%, idle 5%.
  Gap to the floor: GEMM efficiency ~2.8 ms, small kernels ~2.3 ms, idle ~1.1 ms.
- Deep context (109K): decode 42 ms/step, of which **verify attention 18 ms** (reads ~3.6 GB of int8 KV →
  ~21% of bandwidth; tensor-core bound, floor ~5–6 ms). Prefill 166 s = attention 87 s (~38% of bf16
  tensor peak) + GEMMs 74 s (at peak).
- Tokens per verify step (draft acceptance) varies far more than step time: prose ~2.6, explain ~4,
  code ~5.4, JSON ~6.3.

## To-do (highest expected value first)
1. **Draft acceptance on real traffic.** Opt-in logging of request text on one endpoint for a while, then
   tune on it: DFlash2 settings, the 40K draft vocabulary (MTP) / selector, lookup drafting. Each +1
   token/step on prose is ~+38% there; costs no power or bandwidth.
2. **Deep-context verify attention** (patch 01 is config-only): rewrite the split-KV kernel for sm86 with
   int8 tensor cores for Q·Kᵀ (q quantized per row), deeper cp.async pipelining, 16 B-aligned K/V reads
   (the inline scale makes rows only 4 B aligned). Target 18 → ~6–8 ms/step at 109K (decode ~65 → ~90
   tok/s there). Also helps 14–32K (p90) contexts.
3. **Prefill attention** (patch 02 is config-only): an FA2-style kernel for the int8 per-token-head cache
   (or quantize-on-write with bf16 prefill attention). ~38% → ~60–70% of tensor peak would cut the 109K
   prefill from ~166 s to ~120 s.
4. **Small-kernel/idle overhead (~3.4 ms/step):** fewer launches per GDN layer (glue + norms + conv into
   the neighbouring kernels), then the megakernel direction (a persistent per-layer kernel). Big project;
   the only route to most of the remaining ~1.35x.
5. **Custom int4 GEMV for M ≤ 16 reading Marlin's packed layout** (no room for a second weight copy):
   Marlin is 82–91% of peak on the big matrices, 68–74% on the small ones → ~2–4%.
6. **Measure :8001's 109K deep test at 350 W** (skipped while that endpoint's traffic was being moved).
7. **If concurrency grows** (many agents at once): run one endpoint in HyperQwen batch mode (no drafting;
   ~45 tok/s per stream, ~1,000 tok/s aggregate at 64 streams on a 3090).

## Tried and not worth it
- Narrow split-K GEMM for the GDN in_proj_ba: ties cuBLAS (5.3 vs 5.4 µs).
- Fused CUDA GDN decode kernel: needs bf16 recurrent state; only −0.7% (24.32 → 24.14 ms/step).
- DFLASH_TOKENS=15 at 128K: doesn't fit the KV pool (5.73 GiB needed vs 5.19; bigger pool OOMs).
- Marlin knobs: use_fp32_reduce no effect; atomic-add reduce not applicable (N ≥ 2048, bf16 on sm8x).
- Fusing kernels to cut launch gaps (as a fix for GEMMs slower in-server than isolated): the gap was power
  throttling, fixed by 350 W.


## Update after the night of 09-25 (dev tree on :8002; see PROGRESS.md)
Done: verify attention in CUDA (#06: 2.2x at 110K, drafter 2.3x), V2 sampler small-k (#03), split-KV drafter +
buffer sizing (#04, 1.9 GB freed), GDN metadata once per step (#07). Retain decode 24.87 -> 24.13 ms/step;
121K decode 61 -> 82 tok/s. Not yet on :8001 (the deployed venv has #01-#04 only).
Remaining, by expected value:
1. Verify attention v5: the kernel is now CUDA-core bound (int8->fp conversions, masking, softmax, rescale:
   ~14x the tensor instructions). Skip the rescale when no row max changed, a no-mask fast path for interior
   tiles, int8->bf16 via PRMT+FADD instead of I2F. Est. 538 -> ~420 us/layer at 110K (~+5% decode at 121K).
2. A W4A16 GEMV for M<=16 that reads Marlin's packed layout but gives each CTA whole column tiles (no cross-CTA
   split-K, K split across warps in the CTA): the only route to the ~2 ms/step Marlin gap (~8% at short context).
3. DFLASH_TOKENS>7 correctness on CTX=long (the lookup tail is worth up to ~+10% tok/step on copy-heavy JSON).
4. Prefill attention (int8 KV) in CUDA for long prompts: attention is ~50% of a 121K prefill (198 s).
