import json, itertools
exec(open("./mt_test.py").read().split("W = {name")[0].replace('name="marlin_mt"', 'name="marlin_mt_cfg"'))
ws = torch.zeros(82 * 16, dtype=torch.int32, device=dev)
ctmp = torch.empty(246 * 16 * 256 * 2, dtype=torch.float32, device=dev)
shapes = {"gdn_in_qkvz": (16384, 5120), "o_proj": (5120, 6144), "gate_up": (34816, 5120), "down": (5120, 17408),
          "attn_qkv": (14336, 5120), "dr_qkv": (6144, 5120), "dr_o": (5120, 4096), "dr_fc": (5120, 25600)}
W = {name: [mk(n, k) for _ in range(4)] for name, (n, k) in shapes.items()}
a = {name: torch.randn(M, k, device=dev, dtype=torch.bfloat16) for name, (n, k) in shapes.items()}
outs = {name: torch.empty(M, n, device=dev, dtype=torch.bfloat16) for name, (n, k) in shapes.items()}
def timed(name, tk, tn, sms, reps=40):
    n, k = shapes[name]
    f = lambda: [mt.run(a[name], W[name][i][0], W[name][i][1], ws, outs[name], ctmp, M, n, k, tk, tn, sms, True) for i in range(4)]
    for _ in range(2): f()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(reps): f()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / reps / 4 * 1e3
best = {}
for name, (n, k) in shapes.items():
    base = min(timed(name, -1, -1, 82) for _ in range(2))
    res = []
    for (tk, tn) in ((128, 128), (64, 128), (128, 64)):
        if n % tn or k % tk: continue
        for sms in range(24, 83, 2):
            try: res.append((timed(name, tk, tn, sms), tk, tn, sms))
            except Exception as e: pass
    res.sort()
    t, tk, tn, sms = res[0]; t = min(t, timed(name, tk, tn, sms))
    print(f"{name:12s} default {base:6.1f} us | best ({tk},{tn}) sms {sms:2d}: {t:6.1f} us ({(base - t) / base * 100:+.1f}%) | runner-up " +
          ", ".join(f"({a2},{b2})x{c2} {t2:.1f}" for t2, a2, b2, c2 in res[1:4]), flush=True)
    best[name] = (tk, tn, sms, base, t)
json.dump(best, open("./cfg_best_m8.json", "w"), indent=1)
