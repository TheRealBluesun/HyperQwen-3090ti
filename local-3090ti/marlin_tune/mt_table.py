import json
exec(open("./mt_test.py").read().split("W = {name")[0])
shapes = {"gdn_in_qkvz": (16384, 5120), "o_proj": (5120, 6144), "gate_up": (34816, 5120), "down": (5120, 17408),
          "attn_qkv": (14336, 5120), "lm_head": (248320, 5120), "dr_qkv": (6144, 5120), "dr_o": (5120, 4096), "dr_fc": (5120, 25600)}
ws = torch.zeros(82 * 16, dtype=torch.int32, device=dev)
ctmp = torch.empty(82 * 16 * 256 * 2, dtype=torch.float32, device=dev)
ncopy = {name: (1 if n * k > 5e8 else 4) for name, (n, k) in shapes.items()}
W = {name: [mk(n, k) for _ in range(ncopy[name])] for name, (n, k) in shapes.items()}
a = {name: torch.randn(M, k, device=dev, dtype=torch.bfloat16) for name, (n, k) in shapes.items()}
outs = {name: torch.empty(M, n, device=dev, dtype=torch.bfloat16) for name, (n, k) in shapes.items()}
def run(name, i, sms):
    n, k = shapes[name]; q, s = W[name][i]
    mt.run(a[name], q, s, ws, outs[name], ctmp, M, n, k, -1, -1, sms, True)
def timed(name, sms, reps=40):
    c = ncopy[name]
    for _ in range(2): [run(name, i, sms) for i in range(c)]
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(reps): [run(name, i, sms) for i in range(c)]
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / reps / c * 1e3
table = {}
for name, (n, k) in shapes.items():
    base = min(timed(name, 82) for _ in range(2))
    res = sorted((timed(name, s), s) for s in range(30, 83))
    best_t, best_s = res[0]
    # confirm the winner against the default with a re-measure
    best_t = min(best_t, timed(name, best_s)); base = min(base, timed(name, 82))
    gain = (base - best_t) / base * 100
    print(f"{name:12s} N={n:6d} K={k:5d}: default {base:7.1f} us, best sms {best_s:2d} {best_t:7.1f} us ({gain:+.1f}%)", flush=True)
    if gain > 1.0: table[f"{n},{k}"] = best_s
json.dump(table, open("./sms_table_m8.json", "w"), indent=1)
print("table", table)
