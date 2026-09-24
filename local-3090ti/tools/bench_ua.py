"""Prefill-chunk attention (vLLM Triton unified attention) on the int8_per_token_head cache,
Qwen3.8-27B geometry: Hq=24, Hkv=4, D=256, block 864. One sequence: C new tokens at depth L.
Config via UA_* env vars (read inside ua_exp.unified_attention). Prints us per call."""
import math, os, sys, torch
sys.path.insert(0, os.environ.get("QWEN27_DEV", os.path.expanduser("~/qwen27-dev")))  # sda_exp.py / ua_exp.py: env-tunable copies of the vLLM kernels
import ua_exp as ua
from vllm.v1.attention.ops.triton_unified_attention import KVQuantMode
torch.manual_seed(0)
Hq, Hkv, D, BS = 24, 4, 256, 864
dev = "cuda"
C = int(os.environ.get("CHUNK", "1304"))

def make(total):
    nb = math.ceil(total / BS) + 1
    raw_k = torch.randint(-127, 127, (nb, BS, Hkv, D + 4), dtype=torch.int8, device=dev)
    raw_v = torch.randint(-127, 127, (nb, BS, Hkv, D + 4), dtype=torch.int8, device=dev)
    raw_k[..., D:].view(torch.float32).copy_((torch.rand(nb, BS, Hkv, 1, device=dev) * 0.02 + 0.005))
    raw_v[..., D:].view(torch.float32).copy_((torch.rand(nb, BS, Hkv, 1, device=dev) * 0.02 + 0.005))
    k, v = raw_k[..., :D], raw_v[..., :D]
    ksc, vsc = raw_k[..., D:].view(torch.float32).squeeze(-1), raw_v[..., D:].view(torch.float32).squeeze(-1)
    bt = torch.randperm(nb, device=dev, dtype=torch.int32)[None, :]
    return k, v, ksc, vsc, bt

def call(q, k, v, ksc, vsc, bt, L, out, cu, seq):
    ua.unified_attention(q, k, v, out, cu, C, seq, L + C, 1 / math.sqrt(D), True, (-1, -1), bt, 0,
                         None, None, None, kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
                         k_scale_cache=ksc, v_scale_cache=vsc)

def reference(q, k, v, ksc, vsc, bt, L):
    total = L + C; blocks = bt[0, : math.ceil(total / BS)].long()
    K = (k[blocks].float() * ksc[blocks].unsqueeze(-1)).reshape(-1, Hkv, D)[:total]
    V = (v[blocks].float() * vsc[blocks].unsqueeze(-1)).reshape(-1, Hkv, D)[:total]
    G = Hq // Hkv; out = torch.empty(C, Hq, D, device=dev)
    for h in range(Hq):
        s = (q[:, h].float() @ K[:, h // G].T) / math.sqrt(D)
        mask = torch.arange(total, device=dev)[None, :] > (L + torch.arange(C, device=dev))[:, None]
        out[:, h] = torch.softmax(s.masked_fill(mask, float("-inf")), -1) @ V[:, h // G]
    return out

def bench(fn, reps=5):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1e3

L = int(os.environ.get("DEPTH", "50000"))
k, v, ksc, vsc, bt = make(L + C)
q = torch.randn(C, Hq, D, device=dev, dtype=torch.bfloat16)
out = torch.empty(C, Hq, D, device=dev, dtype=torch.bfloat16)
cu = torch.tensor([0, C], device=dev, dtype=torch.int32); seq = torch.tensor([L + C], device=dev, dtype=torch.int32)
t = bench(lambda: call(q, k, v, ksc, vsc, bt, L, out, cu, seq))
flops = 4 * C * Hq * D * (L + C / 2)
err = ""
if os.environ.get("CHECK"):
    err = f" max err {(out.float() - reference(q, k, v, ksc, vsc, bt, L)).abs().max().item():.2e}"
cfg = {k2: os.environ[k2] for k2 in ("UA_BLOCK_M", "UA_TILE", "UA_WARPS", "UA_STAGES") if k2 in os.environ}
print(f"depth {L:6d} chunk {C}: {t/1e3:8.2f} ms  {flops / t / 1e6:6.1f} TFLOP/s  {cfg or 'default'}{err}", flush=True)
