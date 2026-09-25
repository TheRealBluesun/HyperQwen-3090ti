# SPDX-License-Identifier: Apache-2.0
"""qwen27-dev: small-batch replacements for launch-shaped decode kernels.

silu_and_mul: vLLM's act_and_mul_kernel launches one CTA per token, i.e. 8 CTAs at a verify step on an 82-SM
3090 (~5 us for <1 MB of data, 69 calls per step). This Triton kernel spreads each row over 1024-wide
chunks. Rounding matches the CUDA kernel: silu in fp32, rounded to the activation dtype, then the product in
fp32 rounded once. Larger batches (prefill) keep the stock op; the choice is made at run time inside the
custom op, so the compiled graph has no shape branch.
"""
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

SMALL_TOKENS = 64


@triton.jit
def _silu_mul_kernel(x_ptr, out_ptr, d, stride_x, stride_o, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = j < d
    g = tl.load(x_ptr + t * stride_x + j, mask=m, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + t * stride_x + d + j, mask=m, other=0.0)
    s = (g / (1.0 + tl.exp(-g))).to(u.dtype)
    tl.store(out_ptr + t * stride_o + j, (s.to(tl.float32) * u.to(tl.float32)).to(u.dtype), mask=m)


def _silu_and_mul_impl(out: torch.Tensor, x: torch.Tensor) -> None:
    n = x.numel() // x.shape[-1]
    if n > SMALL_TOKENS or x.stride(-1) != 1 or out.stride(-1) != 1 or x.dim() != 2:
        torch.ops._C.silu_and_mul(out, x)
        return
    d = x.shape[-1] // 2
    BLOCK = 1024
    _silu_mul_kernel[(n, triton.cdiv(d, BLOCK))](x, out, d, x.stride(0), out.stride(0), BLOCK=BLOCK, num_warps=4)


def _silu_and_mul_fake(out: torch.Tensor, x: torch.Tensor) -> None:
    return


direct_register_custom_op(
    op_name="qwen27_silu_and_mul",
    op_func=_silu_and_mul_impl,
    mutates_args=["out"],
    fake_impl=_silu_and_mul_fake,
)
