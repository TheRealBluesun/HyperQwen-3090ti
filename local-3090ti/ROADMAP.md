# Status and what is left (2026-09-26)

Numbers: README.md (summary) and PROGRESS.md (every measurement). This replaces the 09-24 to-do list, which is
done or closed.

## Where the decode step stands (3090, 100K context, 28.7 ms/step)
- Weight GEMMs (Marlin W4A16): 19.3 ms, ~92% of the memory-read floor. Going under it needs fewer weight bytes,
  i.e. lower-bit or lossy weights: out of scope (quality first).
- Verify attention (CUDA v7, patch 06): 5.5 ms; four further variants failed to beat it.
- Everything else: ~3.9 ms, mostly a per-kernel launch/ramp tax (~2.5-3 us x ~700 kernels per step) that only
  a persistent kernel removes, and the persistent-kernel prototype measured slower.
- Tokens per step: ~3.9 on agent traffic, ~3.5 on hidden reasoning. Qwen's own jointly trained MTP head has the
  same per-position acceptance as DFlash2 (~73% on the first token), which points at the model's own
  predictability rather than the drafter.
Single-card lossless decode is at its practical ceiling.

## Closed (measured, not worth it on this card)
- Tree verification: ceiling +7-12% tokens/step on sampled text, but a 16-token verify step costs +49% (32K) /
  +59% (64K) against 8 (attention re-reads the KV per 64-row query tile, GDN loses the lazy commit, Marlin +1.3 ms,
  plus idle gaps). Wider verify (DFLASH_TOKENS=15) likewise.
- Cross-request lookup corpus (draft from other requests' output): offline replay of 320K generated tokens,
  ceiling +1.6%, implementable policies -0.1..-1.9%.
- Drafter fine-tuning: low expected value (see above); the earlier MTP-head fine-tune did not move acceptance.
- Small-kernel fusion pass: drafter K/V projection through Marlin -0.13 ms/step, invisible in tok/s; int4 conv
  projection a wash.
- Persistent megakernel (GDN + MLP layers): correct but ~18% slower than the kernel sequence.
- Custom W4A16 GEMV for M <= 16: ties Marlin at best.
- INT8 activations (W4A8): TTFT -47% but decode +5.8% and upstream perplexity +4.1%.
- Reasoning effort below the default: "low" makes this model think more; "none" halves agent time at a
  quality cost.

## Still open
- Borrow the idle twin GPU for cold long prefills (layer split over PCIe, only while the other endpoint is idle):
  ~2x on 100K first turns; the most engineering of anything left.
- Agent level: tool execution was 14-23% of agent wall time in the benchmarks.
- If concurrency grows: one endpoint in HyperQwen batch mode (no drafting; ~1,000 tok/s aggregate at 64 streams).

## Someday: a per-card persistent megakernel for decode (design note, 2026-09-29)

Goal: squeeze single-stream decode toward the weight-streaming floor. Everything else in this document is
already at or near its limit; this is the one big software lever left on this hardware.

### The prize (be honest about it)
- Today (PP=2, int4, DFlash2, short context): 21.4 ms per verify step, about 8.6 ms on the first card (layers
  0-31) + 12.5 ms on the second (layers 32-63, lm_head, drafter, sampling) + ~0.35 ms idle.
- Floor: ~15 GB of weights stream per step; at ~92-95% of each card's DRAM bandwidth that is ~15-16 ms.
- A near-perfect megakernel might land at ~17-18 ms per step: **~1.2-1.3x single-stream decode**. The bytes still
  have to cross the memory bus, so there is no 2x here. Prefill is compute-bound and would not change.

### What the first attempt taught us (megakernel/mk_gdn.cu, mk_mlp.cu; 09-25)
- Correct end to end for GDN + MLP layers, but ~18% slower than the kernel sequence (350 vs 295 us/layer).
- The losses: **grid-wide barriers** between phases (every barrier waits for the slowest SM; tail skew), and
  DRAM sitting idle during latency-bound phases (norms, glue, GDN recurrence). Four barriers per layer cost ~15 us.
- Dataflow (per-chunk counters instead of barriers) did not help: with a **static** split of work across CTAs,
  slow CTAs stay slow.
- Measured ceiling on what fusion alone buys: one continuous read of all decode weights vs today's ~280 separate
  read kernels differs by only ~0.43 ms/step. The win must come from **overlapping** latency-bound work with the
  weight stream, not from removing launches.
- In-kernel phase timestamps lied (they showed phases at/above the read ceiling); **only CUDA-graph event timing of
  the whole step is trustworthy**.
- Hardware facts learned the hard way (gemv/, megakernel/ notes in PROGRESS.md):
  - a CTA-level cp.async ring streams Marlin's packed layout at 815-875 GB/s; the address pattern does not matter;
  - a lockstep K sweep across all CTAs (all CTAs in one contiguous row band) is what keeps the stream fast;
  - a memory fence waits for the thread's in-flight cp.async;
  - 64-bit division in a per-stage path doubles runtime;
  - mbarrier waiters must never get more than one phase ahead (stages a multiple of rows x producers);
  - `discard.global.L2` is NOT ordered by release/acquire;
  - Marlin already sits at ~91-92% of the read ceiling at M=8, so the GEMMs themselves are not where time goes.

### The design (the one that should work)
Model: Hazy Research's Llama megakernel / ThunderKittens-style "megakernels", and Mirage's persistent-kernel
compiler (MPK).
1. **One persistent kernel per card per verify step.** Each SM runs a small interpreter over a precomputed
   instruction list (op, tile range, dependencies). No global barriers at all.
2. **Fine-grained dependencies.** Each instruction waits on counters for exactly the tiles it reads
   (e.g. the down-projection tile waits for the gate_up tiles it needs, not for the whole layer).
3. **The weight stream never stops.** A producer warp group per SM keeps streaming the *next* instructions'
   weights into shared memory while consumers finish latency-bound work. This is the core idea the first
   attempt was missing: overlap across op AND layer boundaries.
4. **Dynamic, not static, work assignment** for anything with variable cost (attention over long KV, the GDN
   recurrence): a global work queue so fast SMs take more.
5. **Rebuild the latency-bound pieces for parallelism, not just fuse them.** The verify attention (~15-20 us fixed
   cost per layer, 255 registers, 12% occupancy), the GDN recurrence (~21 us/layer, serial over tokens), and the
   norms/glue each need a formulation that uses many SMs briefly instead of few SMs for long.
6. **Pipeline parallel stays.** One megakernel per card; the stage boundary is still the ~80 KB handoff (NCCL
   send/recv or a host-mapped flag, ~5 us latency measured). The second card's kernel also runs lm_head,
   sampling and the DFlash drafter.
7. Shapes are compile-time constants (M = 8 verify tokens, all dims known): no runtime indexing math.

### Milestones (each one must pay for itself before the next starts)
- **M1, the go/no-go:** one fused GDN layer + its MLP (the dominant layer type: 48 of 64), weights streaming
  continuously across the layer's internal boundaries, that **beats the existing kernel sequence for that layer
  under CUDA-graph event timing**, bit-for-bit or within the existing numeric noise. If M1 cannot beat ~295
  us/layer, stop: the approach does not work on this hardware.
- **M2:** the full-attention layer type (verify attention inside the kernel), then a 4-layer block (3 GDN + 1
  attention) with cross-layer weight prefetch.
- **M3:** all 32 layers of one card as one kernel; plug into the V2 runner as a custom op behind a flag; the PP
  gate (tools-pp2/gate.sh) must pass.
- **M4:** the second card's kernel adds lm_head + sampling + the DFlash drafter step.
- Success bar for the whole project: >= 10% on single-stream ms/step at short AND 32K context, gate PASS, with no
  quality change.

### How to measure (non-negotiable)
- Whole-step CUDA-graph event timing and `ms_step.py` / `bench27.py` end to end. Never in-kernel stamps.
- Correctness: the margin-aware PP gate (reference top-2 logprobs), plus the quality battery (perplexity + GSM8K).
- Profile both cards with tools-pp2's nsys helpers (ppidle.py / pptimeline.py) to confirm overlap is real.
