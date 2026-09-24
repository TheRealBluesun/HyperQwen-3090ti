# :8001 Qwen3.8-27B — progress log

Measured with `bench27.py` (streamed, engine-neutral; medians of 3; greedy unless noted) and
`LONG=1` for the ~109K-token deep test. Each row is a full server restart with that config.

| Date | Config | code / prose / explain / json (tok/s) | prefill 18K (tok/s) | 109K prefill | decode @109K | Notes |
|---|---|---|---|---|---|---|
| 09-24 | buun-llama-cpp EXL3 + DFlash2 (old, 2x131K) | 118 / 53 / 74 / 163 | ~650 | 217 s | 41 | starting point |
| 09-24 | HyperQwen MTP CTX=long (150K) | 127 / 88 / 109 / 140 | 1,181 | 134 s | 75 | current service |
| 09-24 | HyperQwen DFlash2 CTX=long (128K) | 230 / 106 / 163 / 264 | 956 | 277 s | 52 | target base for optimization |
| 09-24 | DFlash2 CTX=long + #1 verify-attention tuning (32 splits, 64-token tiles, 8 warps, 2 stages) | 222 / 114 / 156 / 250 | 935 | 275 s | **66** (+27%) | kernel: 109K 1,827 → 1,063 µs/layer, 14K 250 → 162, 3.3K 72 → 55 (`qwen27-dev/patches/01-spec-attn-tuning.patch`); short-context decode within noise |
| 09-24 | + #2 prefill-attention tuning for sm86 (128-row blocks, 128-token tiles, 8 warps; chunks ≥128 tokens) | 233 / 120 / 165 / 265 | ~1,060 | **162 s** (1.7x) | 62–66 | kernel: 12–13 → 30–33 TFLOP/s, 2.4–2.7x per prefill chunk (`patches/02-unified-attn-sm86-prefill.patch`) |
| 09-24 | #3a narrow split-K GEMM for GDN in_proj_ba (N=96) | — | — | — | — | **no gain**: cuBLAS GEMM+reduce 5.4 µs vs narrow 5.3 µs at M=8 in isolation; reverted (`patches/rejected-03-*`) |
| 09-24 | #4a DFLASH_TOKENS=15 at 128K | — | — | — | — | **doesn't fit**: needs 5.73 GiB KV vs 5.19 pinned; raising KV_MEM to 5.8 GiB OOMs on the verify graphs |
| 09-24 | **service switched to DFlash2 CTX=long (128K) + patches 01+02** | 233 / 120 / 165 / 265 | ~1,060 | 162 s | 62–66 | vs the MTP-150K service: short/medium prompts 1.5–1.9x faster; ~109K still slower (134 s / 75 tok/s) |

### Where the short-context step goes now (DFlash2, ~24 ms/step, 4 tokens/step)
- Marlin W4A16 GEMMs ~18.5 ms for ~14.5 GB of weights (12.6 decoder + 0.66 lm_head + 1.28 drafter):
  ~78% of the 1,008 GB/s peak (gate_up 111 µs ≈ 82%, down 56 µs ≈ 81%). Practical ceiling ~90%
  → up to ~2 ms/step, but needs a new int4 GEMV that reads Marlin's packed layout (a duplicate
  plain copy of 12.6 GB can't fit on 24 GB).
- Per GDN layer ~67 µs outside the GEMMs: recurrence 22.6 µs (writes 8 fp16 states for rollback,
  ~620 GB/s, ≤0.3 ms/step to gain), split/cat/zero glue ~12 µs + launch gaps (~0.8 ms/step total),
  norms/act/conv ~15 µs.
- Idle ~1.2 ms/step, spread over ~1,350 graph nodes (1–2 µs gaps).

### A (int4 GEMM) — measurement before building (09-24)
Marlin alone at M=8 (`tools/bench_marlin.py`, service stopped, 1,008 GB/s peak): lm_head 91%, gate_up 87%,
drafter fc 87%, down 85%, GDN in_proj 82%, attn qkv 82%, GDN out 74%, drafter o 68%. use_fp32_reduce
makes no difference; atomic-add reduce doesn't apply (N ≥ 2048, bf16 on sm8x). In the server the same
GEMMs run ~5–7% slower than isolated (launch ramp/tail, no PDL on Ampere). Realistic upside of a custom
int4 GEMV that reads Marlin's layout: ~0.5–1 ms/step (2–4%), mostly on the small GEMMs, not ~2 ms.
| 09-24 | bench27 now reports **ms per verify step** (from /metrics draft counters): stable to ±0.2 ms, unlike tok/s | — | — | — | — | service baseline: **24.32 ms/step**, 4.38 tokens/step median |
| 09-24 | B: fused CUDA GDN decode (needs bf16 recurrent state; test only) | 233 / 110 / 165 / 264 | — | — | — | **24.14 ms/step (−0.7%)**: not worth bf16 state's quality risk or an fp16 kernel build. Dropped |

### Option 3 premise check: the "in-context" GEMM gap is power throttling (09-24)
- Per-kernel breakdown (prof p1, short decode, 24.1 ms/step): Marlin 18.5 ms/step in the server vs ~16.5 ms
  summed from isolated microbenchmarks; the rest ~4.4 ms of small kernels + ~1.2 ms idle.
- During sustained decode the 3090 Ti sits at its 300 W cap (throttle reason 0x4 = SW power cap):
  SM ~1,500 MHz instead of ~1,950–2,100 boost; memory clock stays at max.
- gate_up Marlin M=8 burst vs sustained (`tools/bench_marlin_sustained.py`): 107 µs (879 GB/s) vs
  **140 µs (659 GB/s) at 1,680 MHz / 297 W**. So decode is power-limited: GEMM speed follows SM clock.
- Implication: fusing kernels to cut launch boundaries (option 3 as planned) targets the wrong cause.
  Levers are (a) the power limit (PSU-constrained, owner decision) and (b) less SM work per weight byte.

### Power-limit test on GPU0 (09-24, temporary; restored to 300 W afterwards)
| GPU0 limit | ms/step | decode vs 300 W | prefill 18K (tok/s) | SM clock (avg busy) | measured draw |
|---|---|---|---|---|---|
| 300 W | 24.32 | — | ~1,060–1,140 | ~1,500 MHz | ~293 W |
| 325 W | 22.47 | +8.2% | 1,201 | 1,767 MHz | 322 W |
| 350 W | 22.00 | +10.5% | 1,237 | 1,876 MHz | 348 W |
| 400 W | 21.95 | +10.8% | 1,266 | 1,927 MHz | 398 W |
Knee at 325–350 W; above that decode is bandwidth-bound again. GPU1 (:8002) idle (~27 W) during the test.
The cap exists because both cards share an 850 W PSU: owner decision (e.g. GPU0 350 / GPU1 250 keeps 600 W total).
