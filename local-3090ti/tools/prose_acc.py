"""Prose draft-acceptance harness. Usage: prose_acc.py URL LABEL [modes]  (modes: greedy,default,t07)
default = no sampling params (server default: T=1.0 top_k 20 top_p 0.95, what the agent sends).
Reports tok/step, ms/step, tok/s and acceptance by draft position per mode; saves results/<t>-prose-<label>.json"""
import json, os, sys, time, urllib.request
URL, LABEL = sys.argv[1].rstrip("/"), sys.argv[2]
MODES = (sys.argv[3] if len(sys.argv) > 3 else "greedy,default").split(",")
MAXTOK = int(os.environ.get("MAXTOK", "400")); SEEDS = int(os.environ.get("SEEDS", "1"))
PREFIX_CH = int(os.environ.get("PREFIX_CHARS", "0"))   # prepend an unrelated document of this many characters
PREFIX = ("Here is a document for reference:\n\n" + open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "longtext110k.txt")).read()[:PREFIX_CH] + "\n\n---\n\nUnrelated request: ") if PREFIX_CH else ""
P = [
 "Write a 400-word essay on why the Roman Republic fell.",
 "Write a short story about a lighthouse keeper who finds a message in a bottle.",
 "Write a friendly email to a colleague explaining that the project deadline is moving by two weeks and why.",
 "Explain to a curious teenager how vaccines train the immune system.",
 "Describe a rainy afternoon in a small Japanese mountain town, in vivid literary prose.",
 "Write a product description for a handmade walnut desk organizer.",
 "Give me advice on how to prepare for my first half marathon, in a conversational tone.",
 "Summarize the causes and consequences of the 2008 financial crisis for a general audience.",
 "Write a reflective journal entry from someone who just moved to a new city alone.",
 "Write a persuasive op-ed arguing that cities should invest more in public libraries.",
 "Explain the difference between weather and climate, with everyday examples.",
 "Write a eulogy-style tribute to a beloved old family dog named Biscuit.",
 "I'm feeling burned out at work and can't focus. What are some practical things I can try this week?",
 "Write the opening scene of a mystery novel set on an overnight train.",
 "Compare living in a big city versus a small town, weighing pros and cons thoughtfully.",
 "Write a letter of recommendation for a former student applying to a nursing program.",
]
def metrics():
    m = {}
    for line in urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode().splitlines():
        if line.startswith("vllm:spec_decode_num_drafts_total"): m["drafts"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            m["pos" + line.split('position="')[1].split('"')[0]] = float(line.rsplit(" ", 1)[1])
    return m
model = json.load(urllib.request.urlopen(URL + "/v1/models", timeout=30))["data"][0]["id"]
out = {"label": LABEL, "modes": {}}
for mode in MODES:
    m0 = metrics(); toks = 0; dt = 0.0; runs = []
    for seed in range(SEEDS):
        for i, p in enumerate(P):
            b = {"model": model, "messages": [{"role": "user", "content": PREFIX + p}], "max_tokens": MAXTOK, "stream": True,
                 "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
            if mode == "greedy": b["temperature"] = 0
            elif mode == "t07": b["temperature"] = 0.7
            if mode != "greedy": b["seed"] = 1000 * seed + i
            t0 = time.time(); tf = tl = None; usage = None; txt = []
            with urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=600) as r:
                for line in r:
                    line = line.strip()
                    if not line.startswith(b"data:") or line == b"data: [DONE]": continue
                    d = json.loads(line[5:]); usage = d.get("usage") or usage
                    ch = d.get("choices") or []
                    if ch and (ch[0].get("delta") or {}).get("content"):
                        now = time.time(); tf = tf or now; tl = now; txt.append(ch[0]["delta"]["content"])
            ct = usage["completion_tokens"]; toks += ct; dt += (tl - tf) if tl and tf else 0
            runs.append({"i": i, "seed": seed, "ct": ct, "text": "".join(txt)})
    m1 = metrics(); steps = m1["drafts"] - m0["drafts"]
    pos = [(m1.get(f"pos{k}", 0) - m0.get(f"pos{k}", 0)) / steps for k in range(16) if f"pos{k}" in m1]
    res = {"tok_per_step": toks / steps, "ms_per_step": dt * 1e3 / steps, "tok_s": toks / dt, "pos": pos, "tokens": toks, "runs": runs}
    out["modes"][mode] = res
    print(f"{LABEL} [{mode:7s}] {toks:5d} tok | tok/step {res['tok_per_step']:.2f} | {res['ms_per_step']:.2f} ms/step | {res['tok_s']:.1f} tok/s | accept by pos: " + " ".join(f"{p:.2f}" for p in pos), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", time.strftime("%Y%m%d-%H%M%S") + f"-prose-{LABEL}.json"), "w"), indent=1)
