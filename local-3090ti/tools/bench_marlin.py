"""Marlin W4A16 (sym int4, group 128) on the real Qwen3.8-27B decode GEMM shapes.
Weights repacked exactly as vLLM does; several distinct copies per shape (L2 is only 6 MB)."""
import os, torch
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear, marlin_permute_scales, marlin_make_workspace_new, marlin_make_empty_g_idx)
dev = "cuda"; PEAK = 1008e9
SHAPES = [("gate_up", 5120, 34816), ("down", 17408, 5120), ("gdn_in", 5120, 16384), ("gdn_out", 6144, 5120),
          ("attn_qkv", 5120, 14336), ("drafter_fc", 25600, 5120), ("drafter_o", 4096, 5120), ("lm_head", 5120, 248320)]
WS = marlin_make_workspace_new(torch.device(dev))
EMPTY = marlin_make_empty_g_idx(torch.device(dev))
def make(K, N):
    q = torch.randint(0, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=dev)
    s = (torch.rand(K // 128, N, device=dev) * 0.01 + 0.001).to(torch.bfloat16)
    w = ops.gptq_marlin_repack(q, torch.empty(0, dtype=torch.int, device=dev), K, N, 4)
    return w, marlin_permute_scales(s, K, N, 128)
def gemm(x, w, s, K, N, fp32_reduce):
    return apply_gptq_marlin_linear(x, w, s, EMPTY, EMPTY, EMPTY, WS, scalar_types.uint4b8, N, K, True,
                                    use_fp32_reduce=fp32_reduce)
def bench(fn, reps=20):
    fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): fn()
    g.replay(); torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record(); g.replay(); b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / reps * 1e3
for name, K, N in SHAPES:
    copies = max(2, int(400e6 // (K * N // 2)))
    Ws = [make(K, N) for _ in range(copies)]
    nbytes = K * N // 2 + K // 128 * N * 2
    for M in (8, 1):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        row = []
        for fp32r in (True, False):
            t = bench(lambda: [gemm(x, w, s, K, N, fp32r) for w, s in Ws]) / copies
            row.append(f"fp32_reduce={fp32r!s:5s} {t:7.1f} us {nbytes / t / 1e3:5.0f} GB/s ({nbytes / t / 1e3 / (PEAK / 1e9) * 100:3.0f}%)")
        print(f"{name:10s} M={M}  K={K:5d} N={N:6d} {nbytes/1e6:6.1f} MB | " + " | ".join(row), flush=True)
    del Ws; torch.cuda.empty_cache()
