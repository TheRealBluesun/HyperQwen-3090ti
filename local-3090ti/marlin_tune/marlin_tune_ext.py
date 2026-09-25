"""qwen27-dev: per-shape Marlin CTA count for M<=8 decode/verify GEMMs (RTX 3090, bf16 x uint4b8, group 128).

At M=8 stock Marlin launches one CTA per SM and splits K across CTAs, so several CTAs share each output
column and hand partial sums through lock-ordered global reductions. For some shapes, a CTA count that
gives every CTA whole column tiles is faster even with SMs idle (gate_up 117.7 -> 114.0 us). This module
is loaded by vllm._custom_ops when VLLM_MARLIN_TUNE=1 (HyperQwen's marlin-tune-table wiring): it registers
_C_marlin_tune::marlin_gemm with the stock schema, runs a standalone build of vLLM 0.29.0's Marlin
(bit-identical kernels, explicit `sms`) for tabled shapes, and forwards every other call to _C::marlin_gemm.
"""
import os

import torch

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "marlin_tune_src")
_TABLE: dict = {}
_EXT = None
_CTMP = {}
_U4B8 = None

_lib = torch.library.Library("_C_marlin_tune", "DEF")
_lib.define(
    "marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, Tensor? b_bias_or_none, Tensor b_scales, "
    "Tensor? a_scales, Tensor? global_scale, Tensor? b_zeros_or_none, Tensor? g_idx_or_none, Tensor? perm_or_none, "
    "Tensor workspace, int b_type_id, SymInt size_m, SymInt size_n, SymInt size_k, bool is_k_full=True, "
    "bool use_atomic_add=False, bool use_fp32_reduce=False, bool is_zp_float=False) -> Tensor")


def _load():
    global _EXT, _U4B8
    if _EXT is None:
        from torch.utils.cpp_extension import load

        from vllm.scalar_type import scalar_types

        _U4B8 = scalar_types.uint4b8.id
        with open(os.path.join(_SRC, "host.cu")) as f1, open(os.path.join(_SRC, "kernels.cu")) as f2:
            allsrc = f1.read() + "\n" + f2.read()
        allp = os.path.join(_SRC, "all.cu")
        if not os.path.exists(allp) or open(allp).read() != allsrc:
            with open(allp, "w") as f:
                f.write(allsrc)
        _EXT = load(name="marlin_mt", sources=[allp, os.path.join(_SRC, "binding.cu")],
                    extra_include_paths=[_SRC, os.path.join(_SRC, "csrc")],
                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_86,code=sm_86", "--expt-relaxed-constexpr",
                                       "-std=c++17", "-DMARLIN_NAMESPACE_NAME=marlin"],
                    extra_cflags=["-O3", "-std=c++17"])
    return _EXT


def set_table(table: dict) -> None:
    """table: {(size_n, size_k): (thread_k, thread_n, sms)} for M<=8 (a bare int = sms, stock tiles)."""
    _TABLE.clear()
    _TABLE.update(table)
    _load()


def _impl(a, c, b_q_weight, b_bias, b_scales, a_scales, global_scale, b_zeros, g_idx, perm, workspace,
          b_type_id, size_m, size_n, size_k, is_k_full=True, use_atomic_add=False, use_fp32_reduce=False,
          is_zp_float=False):
    cfg = _TABLE.get((size_n, size_k))
    tk, tn, sms = (cfg if isinstance(cfg, tuple) else (-1, -1, cfg)) if cfg is not None else (-1, -1, None)
    if (sms is not None and size_m <= 8 and b_type_id == _U4B8 and a.dtype == torch.bfloat16
            and b_scales.dtype == torch.bfloat16 and b_scales.size(0) * 128 == size_k
            and b_bias is None and a_scales is None and global_scale is None
            and (b_zeros is None or b_zeros.numel() == 0) and (g_idx is None or g_idx.numel() == 0)
            and (perm is None or perm.numel() == 0) and is_k_full and not use_atomic_add and not is_zp_float
            and a.stride(1) == 1 and a.stride(0) % 8 == 0 and a.data_ptr() % 16 == 0
            and workspace.numel() >= sms):
        out = c if c is not None else torch.empty((size_m, size_n), dtype=a.dtype, device=a.device)
        ctmp = _CTMP.get(a.device)
        if ctmp is None:
            ctmp = _CTMP[a.device] = torch.empty(82 * 16 * 256, dtype=torch.float32, device=a.device)
        _EXT.run(a, b_q_weight, b_scales, workspace, out, ctmp, size_m, size_n, size_k, tk, tn, sms, use_fp32_reduce)
        return out
    return torch.ops._C.marlin_gemm(a, c, b_q_weight, b_bias, b_scales, a_scales, global_scale, b_zeros, g_idx, perm,
                                    workspace, b_type_id, size_m, size_n, size_k, is_k_full, use_atomic_add,
                                    use_fp32_reduce, is_zp_float)


_lib.impl("marlin_gemm", _impl, "CUDA")
