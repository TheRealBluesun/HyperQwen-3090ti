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

### Night of 09-25, late: small-kernel and GEMM-config work (dev, :8002, RTX 3090 @ 350 W)
| Change | retain greedy ms/step (tok/s) | chat ms/step | Notes |
|---|---|---|---|
| after #07 | 24.13 (250.7) | 23.73 | |
| #08 Marlin M<=8 table: CTA count + tile config per shape | 23.54 (259.4) | 23.16 | standalone vLLM-0.29 Marlin, `VLLM_MARLIN_TUNE=1`; <= 1 bf16 ulp vs stock |
| #09 small-batch silu_and_mul (Triton) | 23.34 (261.5) | 22.94 | 4.9 -> 1.8 us, bit-identical |
| #10 GDN spec decode reads q/k/v in place | 23.06 (264.6) | 22.69 | -4 kernels/layer, bit-identical |

Full dev stack vs the start of the night (single session, uncached prompt, ~400-token summary):
| ctx | ms/step before -> now | decode tok/s before -> now |
|---|---|---|
| 1K | 24.2 -> 22.7 | 157 -> 172 |
| 18K | 27.3 -> 24.1 | 103 -> 117 |
| 34K | 30.0 -> 25.4 | 94 -> 103 |
| 63K | 34.9 -> 27.7 | 78 -> 96 |
| 121K | 44.0 -> 32.1 | 61 -> 84 |
Prefill unchanged (compute-bound at ~89% of the bf16 tensor peak): 1,297 / 1,146 / 1,008 / 824 / 609 tok/s.

Tried and dropped: verify attention v5 (rescale skip, mask-free tiles, FADD int8 decode: +1.5%); a Triton GEMV
for the bf16 in_proj_ba (2x slower than cuBLAS split-K); strided a/b into the recurrent kernel (they were already
contiguous there; the real copy is one fused inductor kernel).

### Night of 09-25, v6/v7 verify attention
| kernel | 110K per layer | 64K | in-server decode 64K / 121K (ms/step) |
|---|---|---|---|
| Triton at the start of the night | 1,176 us | 710 | 34.9 / 44.0 |
| v4 | 536 | 325 | 27.70 / 32.12 |
| v6 (softmax in registers) | 508 | 306 | 27.40 / - |
| v7 (two-level int8 Q, s8 MMA Q.K^T) | 355 | 221 | 26.10 / 29.18 |
v7: 121K decode 94 tok/s (61 at the start of the night); short context unchanged (22.92 ms/step); valid JSON 12/12;
needles at 100K@10%/90% retrieved. Error vs fp32 reference unchanged (~2.4e-3).

### Night of 09-25: prefill attention in CUDA (#12)
The v7 kernel in direct mode (one segment per query tile, normalized bf16 output) replaces vLLM's Triton
unified_attention for all non-verify attention on the int8 cache: 64-67 TFLOP/s effective vs 31 (2.1-2.4x).
| ctx (uncached prompt) | TTFT before -> now | prefill tok/s |
|---|---|---|
| 18K | 15.5 -> 14.3 s | 1,146 -> 1,242 |
| 34K | 33.3 -> 28.9 s | 1,008 -> 1,164 |
| 63K | 76.9 -> 60.7 s | 824 -> 1,044 |
| 121K | 199 -> 139 s | 609 -> 873 |
Checks: replay valid JSON 12/12, needles 30K@50% + 100K@10/50/90% retrieved, GSM8K 200 95.5% (96.0% before).
Prefill is now GEMM-bound (Marlin at the bf16 tensor peak); the remaining lever there is INT8_ACT (quality trade).

## Patch 13: verify-length bug fixed (09-25)
Symptom: DFLASH_TOKENS>7 produced corrupt greedy output in JSON mode ("restarts every 15 seconds" ->
"restarts every 1/its user session"). Not specific to k>7: k=7 with synchronous scheduling breaks the same way,
on the original venv too (pre-existing, not from patches 06-12), in eager mode too.
Root cause: `_causal_conv1d_update_kernel` (GDN conv, spec-decode path) rejects `num_accepted > seqlen`,
zeroing the conv output and skipping the state update. `num_accepted` is from the PREVIOUS step, so any
verify block shorter than the previous step's accepted count trips it in every GDN layer. Block lengths
vary when synchronous scheduling trims drafts at the first grammar-invalid token, under the adaptive
lookup length, and near max_model_len. Async scheduling at k=7 (production) pads rejected drafts
with -1, so the length stays constant and production was not affected.
Fix: bound the check by the conv-state capacity (columns - (width - 1) + 1). The rest of the rolling-state
arithmetic is independent of the previous length. Likely also the root cause of upstream's
"adaptive length corrupts a prefix-cache hit under KVarN" note (length 16 -> 8 after >8 accepted).
Verification: greedy outputs teacher-forced through a non-speculative bf16 oracle (prompt_logprobs);
count tokens that are not the oracle's argmax by >1 nat (`lossless.py`):
| config (8 req x 600 tok, JSON mode) | before | after |
|---|---|---|
| k=7, sync scheduling | 24 bad (max 27 nats) | 1* |
| k=11 pinned, sync | 264 bad | 1* |
| k=11 adaptive, sync | - | 1* (0 without JSON mode) |
| k=11 pinned, async | - | 1* |
| k=7 async (production) | 0 without JSON mode | unchanged |
*the grammar forcing `{` where the model wants a code fence (identical in every JSON run).
Speed on the memory-service replay (greedy, 12 req): k=7 async 262.6 tok/s (22.87 ms x 5.95);
k=7 sync 208.4 (29.16 ms: synchronous scheduling costs 6.3 ms/step); k=11 adaptive (sync) 215.5
(29.52 x 6.29); k=11 pinned async 262.5 (25.55 x 6.63). So k>7 is now correct but break-even: +11%
tok/step, and the 12-token verify costs +2.7 ms in the int4 GEMMs. It pays only if M=12 gets as cheap
as M=8 (the custom W4A16 GEMV item).

## Patch 14: lazy GDN state commit (09-25, both endpoints, `VLLM_QWEN27_LAZY_GDN=1`)
A spec-decode step of M tokens stores the recurrent state after every token (M x 1.57 MB per GDN layer) so the
next step can start from whichever one gets accepted. Patch 14 replaces the spec-path recurrence with a CUDA
kernel (8 lanes per state row, k/q slices in registers, every input loaded in one prologue) that can run *lazy*:
store only the state the step started from plus a 133 KB log of its tokens (post-conv k, v and raw a, b), and have
the next step replay its accepted tokens from there. The replay rounds through fp16 exactly where the full path
stores/reloads, so every state is bit-identical to what the full step would have written.
- Everything else still sees the full layout: align-mode prefix-cache copies only fire at block boundaries, so
  steps within 32/24 tokens of one run full; a request whose last step was lazy and that is read by anything
  else first (e.g. a zero-draft step) is "materialized" (lazy step replayed, per-token states stored).
- Hooks live in the V2 model runner (`model_states/mamba_hybrid.py`: slot reset, per-step decision, capture);
  per-row modes reach the layers through persistent buffers filled by the GDN metadata builder (graph safe).
- Validation: 8 long greedy requests token-identical across lazy / full / full-every-3rd-step /
  materialize-every-2nd-step; oracle scoring clean (max gap 0.12-0.13, as before) incl. 4 concurrent requests;
  ~91% of decode rows run lazy in production.
- Speed: in-server recurrence 21.2 us (Triton) / 18.9 us (new kernel, full) / 15.8 us (lazy) per GDN layer:
  about +0.8-1% decode tok/s. Smaller than the isolated benchmark suggested (the state writes overlap other work in
  the live pipeline; the lazy kernel is latency-bound: prologue loads + the serial token chain).

## Patch 15 + what didn't work (09-26)
Production profile at ~100K context, 28.7 ms/step: int4 GEMMs 19.3 ms (92% of the weight-read floor), target
verify attention 16 x 342 us = 5.5 ms, drafter attention already windowed (5 x 18 us), and the DFlash2 lookup's
suffix scan -- a single Triton program walking the whole history -- 265 us/step.
- Patch 15: the scan split over 64 programs per request, folded with an int64 atomic max on the same packed
  (match length, position) score, so the proposal is exactly the old one (40 randomized trials). 124 -> 6 us at
  105K in isolation. Scratch buffer is allocated by the speculator with its other lookup buffers (a lazily
  allocated one can land in a CUDA-graph memory pool). Oracle scoring clean (max gap 0.12). ~1% at 100K.
- Not worth it:
  - A single-kernel GEMV for the GDN in_proj_ba: slower than cuBLAS + split-K reduce at M >= 8 (every CTA
    re-reads the activations from L2).
  - Drafter-side int4 (its fused K/V projection through Marlin, RTN int4 conv projection): no measurable
    ms/step change at 64K / 100K.
  - Verify attention past v7 (358 us vs a 257 us read floor at 105K): lazy softmax rescale, fp16-accumulated
    P.V, a deeper cp.async ring at one CTA per SM (+55%), and a warp-specialized kernel (Q.K^T/softmax warps
    handing P to P.V warps through named barriers; 20-25% slower). Skipping either phase in the specialized
    kernel gives ~285 us, both together ~the sum: the phases contend for the same SM resources rather than
    waiting on each other, which is also why overlap tricks don't help v7.
- Trap: `spec_decode_attn._sda_ext()` swallows a failed JIT build and returns None, and the wrapper then runs the
  Triton kernel. Check the kernel name when benchmarking a modified `qwen27_sda.cu`.

## Agent-level profile, and patch 16 (09-26)
Real Hermes (headless, an isolated profile with memory off) on 12 synthetic coding / ops tasks against :8002, every
LLM call timed by a logging proxy. Wall time: decode 62% (hidden reasoning alone ~49%: 77% of generated tokens are
thinking), prefill 26%, tool execution 11%, Hermes itself 1%. Repeated tool calls ~1% -- no duplicate-work problem.
Of the prompt tokens actually prefilled, 37% had been seen before: every prefix-cache hit came up one full 864-token
block short, even for block-aligned prompts.
- Cause: with a speculative method that `use_eagle()` covers (DFlash included), prefix hits drop their last block,
  because an EAGLE drafter's KV at position p mixes in token p+1; the Mamba align checkpoint backs off one block to
  match. DFlash's context K/V at p is a projection of the target hidden state at p alone (its grouped conv is causal
  and applies inside draft blocks), so nothing needs dropping; target KV / SSM state are plain prefix functions.
- Patch 16: a separate `eagle_prefix_drop` flag (false for DFlash; `VLLM_QWEN27_DFLASH_KEEP_LAST_BLOCK=0` restores
  the drop) drives the KV-manager drop and the Mamba back-off; everything else EAGLE-related is unchanged.
- Result, same six tasks, per LLM call: re-prefilled seen tokens 45% -> 13% of new tokens (what remains is the
  partial tail below the next 864 boundary), new prompt tokens ~3,070 -> ~2,345, TTFT 2.81 -> 2.14 s (-24%);
  draft acceptance unchanged (3.05 -> 3.14 accepted per step). Warm vs cold greedy outputs identical (7 of 8; the
  eighth diverges at a near-tie), oracle scoring clean for both (max gap 0.12).
- Remaining: the partial tail (~300 tokens per turn) needs `--prefix-match-unit`, which vLLM disables here because
  the drafter's sliding-window group has no fine-grained lookup.
