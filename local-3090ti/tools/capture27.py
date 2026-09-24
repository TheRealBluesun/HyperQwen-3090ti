"""Drive nsys capture ranges on a HyperQwen server launched with prof_a.sh.
Range 1: short-context decode (chat prompt, ~300 tokens). Range 2: one ~109K-token request (prefill + 200 tokens)."""
import json, os, sys, time, urllib.request
URL = sys.argv[1].rstrip("/")
def post(path, body=None):
    req = urllib.request.Request(URL + path, json.dumps(body).encode() if body else b"", {"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=1800).read()
def chat(content, n):
    t = time.time()
    r = json.loads(post("/v1/chat/completions", {"model": "Qwen3.8-27B", "messages": [{"role": "user", "content": content}], "max_tokens": n, "temperature": 0}))
    return r["usage"], time.time() - t
p = "Explain how a B-tree differs from a binary search tree and why databases use B-trees. Be thorough."
chat("Say hi.", 8); chat(p, 64); chat("Write a haiku about rain.", 40)          # warm-up
post("/start_profile"); u, t = chat(p, 300); post("/stop_profile"); print("range 1 (short decode):", u, f"{t:.1f}s", flush=True)
time.sleep(5)
long = f"[{time.time()}]\n" + open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "longtext110k.txt")).read() + "\n\nSummarize the material above in about 150 words."
post("/start_profile"); u, t = chat(long, 200); post("/stop_profile"); print("range 2 (109K prefill + decode):", u, f"{t:.1f}s", flush=True)
