# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-K Triton GEMM for very narrow BF16 linears at decode batch sizes.

cuBLAS picks 8-CTA WMMA kernels for layers such as Qwen4Exp's GDN
``in_proj_ba`` (N=96, K=2560) or hyper-connection ``block_inject`` (N=4),
spending ~14 us on 0.5 MB of weights. Splitting K across the whole GPU brings
those to ~3 us. Enabled with ``VLLM_NARROW_GEMM=1``; only layers with
``N <= NARROW_MAX_N`` and no bias are routed here, and only for ``M <= 64``.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

NARROW_MAX_N = 256
_MAX_M = 64


@triton.jit
def _narrow_gemm_kernel(
    x_ptr, w_ptr, y_ptr, M, N, K, stride_xm, stride_wn, stride_ym,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = pid_k * K_PER_SPLIT + kk + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w))
    tl.atomic_add(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        sem="relaxed",
    )


_NUM_SMS: int | None = None


def _narrow_gemm_impl(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    global _NUM_SMS
    m = x.shape[0]
    if m == 0 or m > _MAX_M:
        return torch.nn.functional.linear(x, weight)
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(x.device).multi_processor_count
    x = x.contiguous()
    n, k = weight.shape
    block_m = 16 if m <= 16 else 32 if m <= 32 else 64
    block_n, block_k = (16, 64) if n <= 64 else (32, 64)
    tiles = triton.cdiv(n, block_n)
    split = 1
    while tiles * split < 2 * _NUM_SMS and k // (block_k * split * 2) >= 1:
        split *= 2
    k_per_split = triton.cdiv(triton.cdiv(k, split), block_k) * block_k
    split = triton.cdiv(k, k_per_split)
    acc = torch.zeros((m, n), device=x.device, dtype=torch.float32)
    _narrow_gemm_kernel[(tiles, split)](
        x, weight, acc, m, n, k, x.stride(0), weight.stride(0), acc.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, K_PER_SPLIT=k_per_split,
        num_warps=4, num_stages=3,
    )
    return acc.to(x.dtype)


def _narrow_gemm_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


direct_register_custom_op(
    op_name="cuda_narrow_bf16_gemm",
    op_func=_narrow_gemm_impl,
    fake_impl=_narrow_gemm_fake,
)


def narrow_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        bias is not None
        or weight.dim() != 2
        or weight.shape[0] > NARROW_MAX_N
        or weight.dtype != torch.bfloat16
        or x.dtype != torch.bfloat16
        or weight.stride(1) != 1
    ):
        return torch.nn.functional.linear(x, weight, bias)
    lead = x.shape[:-1]
    y = torch.ops.vllm.cuda_narrow_bf16_gemm(x.reshape(-1, x.shape[-1]), weight)
    return y.reshape(*lead, weight.shape[0])



def pdl_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """VLLM_PDL_GEMV: narrow layers (N <= NARROW_MAX_N) keep the split-K narrow
    kernel if VLLM_NARROW_GEMM is also set; wider BF16 layers use the PDL GEMV."""
    if (
        bias is not None
        or weight.dim() != 2
        or weight.dtype != torch.bfloat16
        or x.dtype != torch.bfloat16
        or weight.stride(1) != 1
    ):
        return torch.nn.functional.linear(x, weight, bias)
    if weight.shape[0] <= NARROW_MAX_N and os.environ.get("VLLM_NARROW_GEMM", "0") == "1":
        return narrow_unquantized_gemm(layer, x, weight, bias)
    from vllm.model_executor.layers.pdl_gemv import evict_policy, pdl_bf16_linear

    return pdl_bf16_linear(x, weight, evict_policy(layer))
