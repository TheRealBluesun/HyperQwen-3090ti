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

## Still open (outside decode)
- Borrow the idle twin GPU for cold long prefills (layer split over PCIe, only while the other endpoint is idle):
  ~2x on 100K first turns; the most engineering of anything left.
- Agent level: tool execution was 14-23% of agent wall time in the benchmarks.
- If concurrency grows: one endpoint in HyperQwen batch mode (no drafting; ~1,000 tok/s aggregate at 64 streams).
