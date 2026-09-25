# Night of 2026-09-25: the 3090 endpoint optimization summary

## Where things stand
Both endpoints run the patched stack (deployed 09-25 morning): the units set `PYTHONPATH` to a vLLM overlay
(`~/build/HyperQwen/dev/site`, a copy of the venv's package with patches 03-12 applied) and `VLLM_MARLIN_TUNE=1`.
Rollback: remove those two lines from the unit and restart. The 3090 Ti endpoint measured 22.0 -> 21.0 ms/step
on the chat benchmark after deployment.

| Measure (RTX 3090 @ 350 W) | start of night | dev stack now |
|---|---|---|
| memory-service replay, greedy (12 req) | 24.87 ms/step, 245.5 tok/s | 22.92 ms/step, 265 tok/s |
| chat benchmark step | 23.74 ms | 22.69 ms |
| decode @ 1K / 18K / 34K / 63K / 121K | 157 / 103 / 94 / 78 / 61 tok/s | 172 / 117 / 103 / 109 / 94 tok/s |
| ms/step @ 63K / 121K | 34.9 / 44.0 | 26.1 / 29.2 |
| time to first token (uncached) @ 34K / 63K / 121K | 33 / 77 / 199 s | 29 / 61 / 139 s |
| GSM8K 200 (greedy) | 96.0% (deployed build) | 96.0% (95.5% with #12) |
| needle 30K/100K @ 10-90% depth | - | all retrieved |
| 2 concurrent memory-service requests (wall clock) | 227 tok/s | ~240 tok/s |

## What changed (repo: TheRealBluesun/HyperQwen-3090ti, local-3090ti/)
- 06 CUDA split-KV verify attention (int8 per-token-head cache), now v7 (int8 tensor cores for Q.K^T with
  two-level Q): 3.3x the Triton kernel at 110K (1,176 -> 355 us/layer), 1.5x at 4K, drafter 2.3x
- 07 GDN attention metadata built once per step instead of once per kv-cache group (6x)
- 08 Marlin config table (tile + CTA count) per decode shape, M<=8 and M<=16 (standalone vLLM-0.29 Marlin build)
- 09 small-batch silu_and_mul (Triton, bit-identical)
- 10 GDN spec decode reads q/k/v in place (4 fewer kernels per GDN layer, bit-identical)
- 11 bf16-cache robustness fixes (only CTX=fast was affected)
- 12 prefill attention through the same CUDA kernel (direct mode): 2.1-2.4x the Triton prefill attention
- (03/04 from yesterday: sampler small-k, split-KV drafter attention + 1.9 GB buffer fix)

## Options that are your call
- INT8_ACT=int8 (W4A8): memory-service TTFT 1.06 -> 0.56 s, but decode +5.8% ms/step and different outputs
  (upstream: perplexity +4.1%). Net ~-6% per request. Left off.
- The memory service's 2 concurrent requests is fine: 2-at-a-time is +40% wall-clock throughput, and the crash that
  prompted the question is fixed (KV_MEM pin + Restart=always on both units).

## Dead ends (measured)
DFLASH_TOKENS>7 (wrong output on CTX=long and CTX=fast, attention ruled out), drafter 40K-vocab head (-2.4%),
int8 Q.K^T, Marlin at >82 CTAs, GDN state-write replay (~0.13 ms), CPU EPP, SM/mem clock locks, triton GEMV for
in_proj_ba, verify-attention v5.
