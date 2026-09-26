# HyperQwen on an RTX 3090 / 3090 Ti: Qwen3.8-27B W4A16, DFlash2, 128K context

Local deployment and optimization work on top of upstream HyperQwen (syv-ai/HyperQwen). Two single-user
endpoints, one per GPU (3090 Ti and 3090, both capped at 350 W), each with at least 128K of context.

## Result (3090 Ti, `tools/bench27.py` with the 109K test, greedy medians)
| metric | HyperQwen as cloned (DFlash2, 128K) | with patches 01-18 | change |
|---|---|---|---|
| decode, code / prose / explain / json | 230 / 106 / 163 / 264 tok/s | 266 / 132 / 193 / 305 | 1.16-1.25x |
| prefill, 18K prompt | 956 tok/s | 1,313 | 1.37x |
| prefill, 109K prompt | 277 s | 113.6 s | 2.44x |
| decode after 109K of context | 52 tok/s | 94 | 1.81x |

The baseline ran at a 300 W cap, the final numbers at 350 W. For agent workloads, patches 16 and 17 matter as much:
a turn re-prefills ~26 already-seen tokens instead of ~394 plus a dropped 864-token block. Details: PROGRESS.md.

## Files
- ROADMAP.md: where things stand, what was closed and why, what is left.
- PROGRESS.md: every measured configuration and experiment, in order (including the ones that did not work).
- NOTES.md: the 09-24 baseline (setup, comparison with the previous llama.cpp endpoint, gotchas).
- patches/: changes on top of the HyperQwen-patched vLLM 0.29.0; `patches/apply.sh` applies them in order
  (after the upstream patch series). Patch 08 (the Marlin per-shape config table) lives in marlin_tune/.
- deploy/: `run.sh` (launcher: SPEC=dflash2 CTX=long, `--prefix-match-unit 32`), the two systemd user units,
  `dev_up.sh` / `dev_prof.sh` / `prof_stop.sh` (a test instance on GPU1 from an overlay copy, with or without
  nsys), and the native install scripts.
- marlin_tune/: the tuning pipeline behind patch 08 (`VLLM_MARLIN_TUNE=1`).
- tools/: `bench27.py` (engine-neutral streamed benchmark; `LONG=1` adds the 109K test), `prose_acc.py` (draft
  acceptance by position on prose), `bench_marlin*.py` (Marlin microbenchmarks).
- results/: raw `bench27.py` runs.

## Patches
| # | change |
|---|---|
| 01, 02 | Triton split-KV verify / sm86 prefill attention configs (now the fallback path) |
| 03 | V2 sampler: small top-k fast path |
| 04 | split-KV attention for the drafter's sliding-window layers + buffer sizing (1.9 GB freed per card) |
| 06 | CUDA split-KV verify attention for the int8 per-token-head cache (v7: 3.3x the Triton kernel at 110K) |
| 07 | GDN attention metadata built once per step |
| 08 | Marlin config table (tile + CTA count) per decode shape, M <= 8 and M <= 16 (marlin_tune/) |
| 09 | small-batch silu_and_mul |
| 10 | GDN spec decode reads q/k/v in place |
| 11 | bf16-cache robustness for the split-KV path |
| 12 | prefill attention through the CUDA kernel (121K prefill 199 -> 139 s) |
| 13 | conv1d spec-decode guard (fixes corrupt output when the verify block shrinks) |
| 14 | lazy GDN state commit: 1 state + a token log per step instead of 8 states, bit-identical (`VLLM_QWEN27_LAZY_GDN=1`) |
| 15 | DFlash2 lookup drafting: suffix scan split over 64 programs per request |
| 16 | prefix-cache hits keep their last block with DFlash |
| 17 | fine-grained prefix-cache hits with the drafter's sliding-window group (`--prefix-match-unit 32`) |
| 18 | lazy GDN turns itself off above an 8-token verify block |
