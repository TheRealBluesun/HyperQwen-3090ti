"""gate_up Marlin GEMM at M=8: burst vs sustained (power-capped) timing, with SM clock readout."""
import subprocess, time, torch
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear, marlin_permute_scales, marlin_make_workspace_new, marlin_make_empty_g_idx)
dev = "cuda"; K, N = 5120, 34816
WS = marlin_make_workspace_new(torch.device(dev)); E = marlin_make_empty_g_idx(torch.device(dev))
Ws = []
for _ in range(5):
    q = torch.randint(0, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=dev)
    s = (torch.rand(K // 128, N, device=dev) * 0.01 + 0.001).to(torch.bfloat16)
    Ws.append((ops.gptq_marlin_repack(q, torch.empty(0, dtype=torch.int, device=dev), K, N, 4), marlin_permute_scales(s, K, N, 128)))
x = torch.randn(8, K, device=dev, dtype=torch.bfloat16)
f = lambda: [apply_gptq_marlin_linear(x, w, s, E, E, E, WS, scalar_types.uint4b8, N, K, True) for w, s in Ws]
f(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(10): f()
nbytes = K * N // 2 + K // 128 * N * 2
def timed():
    a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record(); g.replay(); b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / 50 * 1e3
def clock():
    return subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=clocks.sm,power.draw", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
time.sleep(3)
print(f"burst:     {timed():6.1f} us/call ({nbytes / timed() / 1e3:.0f} GB/s)")
t0 = time.time()
while time.time() - t0 < 4: g.replay()
torch.cuda.synchronize()
t = timed(); c = clock()
while time.time() - t0 < 5: g.replay()
torch.cuda.synchronize(); c = clock()
print(f"sustained: {t:6.1f} us/call ({nbytes / t / 1e3:.0f} GB/s)  [SM clock, power: {c}]")
