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

### Both GPUs at 350 W (persistent since 09-24: /etc/default/nvidia-power-limit LIMIT_W=350) and :8002 migrated
| Endpoint | GPU | ms/step | greedy code / prose / explain / json | prefill 18K | 109K prefill | decode @109K |
|---|---|---|---|---|---|---|
| :8001 hyperqwen-a | 3090 Ti, 350 W | **22.00** | 252 / 131 / 182 / 290 | 1,237 | (not re-run) | — |
| :8002 hyperqwen-b | 3090, 350 W | **23.83** | 230 / 114 / 174 / 268 | 1,148 | 168 s | 65 |
:8002 was buun-llama-cpp EXL3 (disabled, kept for rollback); now the same HyperQwen DFlash2 128K setup,
same venv and patches (`deploy/run.sh` with GPU/PORT from the unit). The 3090's step is ~8% longer,
matching its ~7% lower memory bandwidth (936 vs 1,008 GB/s).

### Re-profile of :8002 at 350 W (09-24 late, prof/p2-b-350W; RTX 3090, 936 GB/s)
Short-context decode (24.97 ms/step under nsys, 23.8 without; ~15.2 GB moved per step):
int4 GEMMs 19.66 ms (78.7%, ~737 GB/s = 79% of peak) · GDN recurrence+conv 1.13 · bf16 GEMMs 1.07 ·
norms/activations 0.69 · glue 0.54 · attention 0.50 · drafter/sampling 0.15 · other 0.11 · idle 1.13.
Whole step ≈ 638 GB/s = **68% of peak**. Floor at a realistic 92%: ~17.7 ms (1.35x headroom): GEMM
efficiency ~2.8 ms, small kernels ~4.2 ms, idle ~1.1 ms.
Deep context: decode @109K 42.2 ms/step = verify attention 17.9 (reads ~3.6 GB int8 KV per step → ~21% of
bandwidth; compute floor ~5–6 ms) + GEMMs 19.1 + other 4.8 + idle 0.3. 109K prefill 166 s = attention 87 s
(~27 TFLOP/s, ~38% of the 3090's ~71 TFLOP/s bf16 tensor peak) + GEMMs 74 s (~72 TFLOP/s: at peak).

## Decode on memory-service-shaped traffic (09-25, :8002 / RTX 3090 @ 350 W)
Workload: the memory service's fact-extraction calls, rebuilt with its own prompt builder (fixed ~3K-token
system prompt, ~750-token conversation chunk, `response_format: json_object`, T=0.1; model defaults
top_k=20 / top_p=0.95). Synthetic conversations. ~4K context, pure decode. `memory-service/run_workload.py`.

| Step | ms/step | decode (12 req) | tok/step | Notes |
|---|---|---|---|---|
| baseline | 26.21 | 232 tok/s | 6.04 | JSON grammar costs 0.57 ms/step (nofmt: 25.64) |
| + #03 V2 sampler passes k_max | 25.56 | 236 | 6.00 | Triton top-k/top-p 813 us/step -> sort-free torch.topk |
| + #04 split-KV drafter attention | 24.93 | 241 | 5.96 | drafter's 5 non-causal 2048-window layers: 170-180 -> 58 us each |
| greedy A/B of #04 (SWA=0 vs 1) | 25.20 -> 24.73 | 242 -> 247 | 6.06 / 6.05 | acceptance identical at every position |

#04 also fixes split-KV buffer sizing: the vLLM config is never set during forward, so the buffers were
sized for a 256-request fallback (1,935 MiB for the target). Primed from TritonAttentionImpl.__init__ now:
30 MiB (target) + 20 MiB (drafter); ~1.9 GB freed per card after load.
Other findings: two concurrent retains give 227 tok/s aggregate (no gain over one); acceptance on this
workload is high (0.91 ... 0.56 by draft position, ~6 tokens/step); two full-vocab int4 lm_head passes per
step (drafter candidates + target) = 1.5 ms; per-step Marlin 19.2 ms.

## Night of 09-25: dev tree on :8002 (RTX 3090 @ 350 W)
Dev overlay: a copy of the vllm package loaded via PYTHONPATH by a transient unit (`deploy/dev_up.sh`), so
the deployed venv is untouched. Standard check = `nightbench.sh` (retain replay as sent + greedy, greedy chat)
+ greedy-output comparison against a k=7 reference (run-to-run identical at k=7).

Measured / tried:
- Decode is power-capped: 344 W, SM ~1,800 MHz (throttle 0x4), memory 9,501 MHz (CUDA P2 cap; -lmc 9751 is
  ignored). Streaming read ceiling on this card at 350 W: ~825-840 GB/s. SM locked at 1,500 MHz: 257 W, +5% step.
- Big Marlin GEMMs run at 93-94% of that ceiling; lm_head ~100%; GDN out_proj ~74%.
- DFLASH_TOKENS=11 at 128K (fits after the buffer fix, 5.5 GiB pool): output CORRUPT ("restarts every 1/its user
  session"), adaptive and pinned verify length, with and without patch 04. Pre-existing bug in the >7-token
  path on CTX=long; parked.
- Drafter 40,960-token candidate head (reusing mtp.draft_lm_head.*): -0.76 ms/step but tok/step 6.05 -> 5.74
  on retain (the vocab misses copied names/paths) = -2.4% net. Rejected.
- int8 Q.K^T in the verify kernel: +15% only and 16x the error with outlier q channels. Rejected.
  fp16-accumulate P.V: no gain. Triton config sweeps: plateau ~15%.

Verify attention in CUDA (`qwen27_sda.cu`, JIT-built; Nsight showed the Triton kernel at 1 CTA/SM, 254 regs,
1.56 waves, DRAM 18%). v1: 4 warps, 32-key tiles, 16-byte cp.async double buffering of 4-byte-aligned head
rows (K/V interleave per head), key-split QK, D-split PV. v3: Q in registers, row-tile QK, ~45 KB smem ->
2 CTAs/SM, 2x segments. Target 110K: 1,176 -> 552 us/layer; 64K 710 -> 333; 4K 64 -> 49. Drafter (v1): 56 -> 25 us.
Same error as the Triton kernel (rel ~2.4e-3 vs fp32 reference); greedy retain outputs valid JSON 12/12.

| Test (:8002, dev) | before tonight | + CUDA verify attention |
|---|---|---|
| retain greedy tok/s (ms/step) | 245.5 (24.87) | 250.3 (24.29) |
| decode @ 34K / 63K / 121K tok/s | 94 / 78 / 61 | 107 / 98.5 / 82 |
| ms/step @ 34K / 63K / 121K | 30.0 / 34.9 / 44.0 | 26.6 / 28.9 / 33.5 |

### Night of 09-25, continued (all on :8002 dev, RTX 3090 @ 350 W)
| Change | retain greedy ms/step (tok/s) | long-context decode | Notes |
|---|---|---|---|
| start of night (patches 01-04) | 24.87 (245.5) | 94 / 78 / 61 tok/s @ 34K/63K/121K | |
| #06 CUDA verify attention v1/v3 | 24.29 (250.3) | 107 / 98.5 / 82 | per layer @110K 1,176 -> 552 us |
| #06 race fix + v4 (fp16 Q.K^T) + adaptive segments | 24.29 (249.3) | 28.7 ms/step @63K, 33.1 @121K | 110K 538 us; 4K 51 -> 43 us |
| #07 GDN metadata built once per step (6 groups) | 24.13 (250.7) | - | outputs identical 12/12 |

Measured and rejected tonight (details in NOTES.md):
- Stock Marlin at M=8 is at 90-95% of the card's ~865 GB/s ceiling on the big matrices, 70% on the 16 MB o_proj.
  A standalone build of vLLM 0.29's Marlin (bit-identical output) with explicit thread_k/thread_n/sms: no config
  beats the default; 2-3 CTAs/SM is slower. The ~2 ms/step gap is Marlin's split-K scheduling, not its tiles.
- GDN recurrent update: the 8 per-token state writes cost only ~6.6 of 22.7 us/layer; the rest is the sequential
  recurrence. A replay/materialize redesign would save ~0.13 ms/step. Launch-config sweep: best -8%.
- CPU energy-performance preference (performance vs balance_performance): no difference.
- INT8_ACT=int8 (W4A8): retain TTFT 1.06 -> 0.56 s, but decode +5.8% ms/step (the W4A8 path covers the decode
  GEMMs too; weights are repacked for it) and greedy outputs diverge from W4A16 early (upstream: ppl +4.1%).
  Net ~-6% per memory-service request; left off, the user's call.
- DFLASH_TOKENS>7 at CTX=long: corrupt output (not caused by our patches; the GDN state-slot count is right).
