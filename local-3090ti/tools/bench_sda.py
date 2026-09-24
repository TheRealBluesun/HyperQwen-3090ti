"""Microbenchmark: split-KV verify attention on the int8_per_token_head cache (one layer),
Qwen3.8-27B geometry (Hq=24, Hkv=4, D=256), block_size 864, q_len 8 (DFlash2 verify), 1 request."""
import os, sys, math, torch, importlib
sys.path.insert(0, os.environ.get("QWEN27_DEV", os.path.expanduser("~/qwen27-dev")))  # sda_exp.py / ua_exp.py: env-tunable copies of the vLLM kernels
mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "sda_exp")
torch.manual_seed(0)
Hq, Hkv, D, BS, QL = 24, 4, 256, 864, 8
dev = "cuda"

def make(ctx):
    nb = math.ceil(ctx / BS) + 1
    raw_k = torch.randint(-127, 127, (nb, BS, Hkv, D + 4), dtype=torch.int8, device=dev)
    raw_v = torch.randint(-127, 127, (nb, BS, Hkv, D + 4), dtype=torch.int8, device=dev)
    sk = (torch.rand(nb, BS, Hkv, device=dev) * 0.02 + 0.005)
    sv = (torch.rand(nb, BS, Hkv, device=dev) * 0.02 + 0.005)
    raw_k[..., D:] = sk.view(torch.int8).view(nb, BS, Hkv, 4) if False else raw_k[..., D:]
    # inline fp32 scales in the 4 padding bytes, like TRITON_ATTN's int8_per_token_head layout
    raw_k[..., D:].view(torch.float32).copy_(sk.unsqueeze(-1)); raw_v[..., D:].view(torch.float32).copy_(sv.unsqueeze(-1))
    k, v = raw_k[..., :D], raw_v[..., :D]
    ksc, vsc = raw_k[..., D:].view(torch.float32).squeeze(-1), raw_v[..., D:].view(torch.float32).squeeze(-1)
    bt = torch.randperm(nb, device=dev, dtype=torch.int32)[None, :]
    q = torch.randn(QL, Hq, D, device=dev, dtype=torch.bfloat16)
    return q, k, v, ksc, vsc, bt

def reference(q, k, v, ksc, vsc, bt, ctx):
    # dequantized, gathered K/V in fp32, causal on the last QL positions
    blocks = bt[0, : math.ceil(ctx / BS)].long()
    K = (k[blocks].float() * ksc[blocks].unsqueeze(-1)).reshape(-1, Hkv, D)[:ctx]
    V = (v[blocks].float() * vsc[blocks].unsqueeze(-1)).reshape(-1, Hkv, D)[:ctx]
    G = Hq // Hkv
    out = torch.empty(QL, Hq, D, device=dev)
    for i in range(QL):
        pos = ctx - QL + i
        for h in range(Hq):
            s = (q[i, h].float() @ K[: pos + 1, h // G].T) / math.sqrt(D)
            out[i, h] = torch.softmax(s, -1) @ V[: pos + 1, h // G]
    return out

CU = torch.tensor([0, QL], device=dev, dtype=torch.int32)
def run(att, q, k, v, ksc, vsc, bt, ctx, out, seq):
    att.run(q, k, v, out, CU, seq, bt, 1 / math.sqrt(D), 1, QL, k_scale_cache=ksc, v_scale_cache=vsc)

def bench(fn, reps=50):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): fn()
    g.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record(); g.replay(); b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1e3

import itertools, os
configs = [dict(nseg=16)] + [dict(nseg=64, stages=2, tile=64, warps=8, min_tiles=m) for m in (1, 2, 4, 8)] + [dict(nseg=32, stages=2, tile=64, warps=8, min_tiles=m) for m in (1, 2)]
for ctx in [int(x) for x in os.environ.get('CTXS', '109000,14000,3300').split(',')]:
    q, k, v, ksc, vsc, bt = make(ctx)
    kv_bytes = ctx * Hkv * (D + 4) * 2
    ref = reference(q, k, v, ksc, vsc, bt, ctx) if ctx <= 14_000 else None
    print(f"ctx {ctx}: KV read {kv_bytes/1e6:.0f} MB -> {kv_bytes/1.008e12*1e6:.0f} us at 1008 GB/s")
    for c in configs:
        att = mod.SpecDecodeAttention(1, Hq, D, dev, qmax=mod.QMAX_TOKENS, num_segments=c["nseg"])
        att.stages = c.get("stages", 1); att.tile_override = c.get("tile"); att.warps_override = c.get("warps"); att.min_tiles = c.get("min_tiles", 1)
        out = torch.empty(QL, Hq, D, device=dev, dtype=torch.bfloat16)
        seq = torch.tensor([ctx], device=dev, dtype=torch.int32)
        try:
            t = bench(lambda: run(att, q, k, v, ksc, vsc, bt, ctx, out, seq))
        except Exception as e:
            print(f"   {c}: FAILED {type(e).__name__}: {str(e)[:120]}"); continue
        err = "" if ref is None else f"  max err {(out.float() - ref).abs().max().item():.2e}"
        print(f"   {str(c):52s} {t:8.1f} us  ({kv_bytes / t / 1e3:6.0f} GB/s){err}", flush=True)
    del q, k, v, ksc, vsc, bt; torch.cuda.empty_cache()
