#!/bin/bash
# Check that this repo is installed the way the README numbers assume:
# venv + vLLM version, every compatible patch applied, the model requantized (lm_head,
# embed_tokens, MTP module, draft head), keys/files present, and — if a
# server is running — that it answers and which backend/pool it came up with.
#
#   bash verify.sh            # everything
#   bash verify.sh --no-server
#   bash verify.sh --install  # only the install (venv, vLLM, patches, KVarN): no GPU,
#                             # model or server checks — what the Docker build runs
# Exit code: 0 all PASS (WARNs allowed), 1 if anything FAILs.
# PY=/path/to/python overrides the interpreter (default: this repo's venv).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
NOSRV=0; INSTALL=0
for a in "$@"; do case "$a" in --no-server) NOSRV=1;; --install) INSTALL=1; NOSRV=1;; esac; done
FAILS=0
ok()   { printf "  PASS  %s\n" "$1"; }
warn() { printf "  WARN  %s\n" "$1"; }
fail() { printf "  FAIL  %s\n" "$1"; FAILS=$((FAILS+1)); }
MODEL=${MODEL:-$HERE/models/Qwen3.8-27B-W4A16-AutoRound}
PY=${PY:-$HERE/venv/bin/python}

echo "== environment"
[ -x "$PY" ] && ok "python: $PY" || { fail "no $PY (see README Setup)"; exit 1; }
VER=$($PY -c "import vllm; print(vllm.__version__)" 2>/dev/null | tail -n1)
[ "$VER" = "0.28.0" ] && ok "vllm $VER" || warn "vllm ${VER:-missing} (patches were written against 0.28.0)"
SP=$($PY -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null | tail -n1)
[ -n "$SP" ] && [ -d "$SP" ] && ok "vllm package at $SP" || { fail "cannot import vllm with $PY"; exit 1; }
if [ $INSTALL = 0 ]; then
$PY - <<'EOF' 2>/dev/null || fail "torch cannot see a CUDA GPU"
import torch; assert torch.cuda.is_available()
p=torch.cuda.get_device_properties(0)
print(f"  PASS  GPU: {p.name}, {p.total_memory/2**30:.1f} GiB, sm{p.major}{p.minor}, torch {torch.__version__}")
EOF
command -v nvidia-smi >/dev/null && { PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1); ok "power limit ${PL} W (README numbers are at 250 W)"; }
fi
for t in triton compressed_tensors; do $PY -c "import $t" 2>/dev/null && ok "python module $t" || fail "python module $t missing"; done
# a bare `import flashinfer` passes while vLLM still falls back to torch.topk:
# has_flashinfer() additionally wants nvcc on PATH or the flashinfer-cubin
# package (#35). Test what the server will actually use.
export FLASHINFER_DISABLE_VERSION_CHECK=1  # cubin publishes 0.6.13 vs python 0.6.16.post3; the launchers export this too
$PY -c "from vllm.utils.flashinfer import has_flashinfer; assert has_flashinfer()" 2>/dev/null \
  && ok "flashinfer usable by vLLM (nvcc or flashinfer-cubin present)" \
  || fail "flashinfer unusable: DFlash2 selector will run torch.topk at ~half speed. pip install flashinfer-python flashinfer-cubin==0.6.13 (#35)" 

echo "== vLLM patches (order: patches/series)"
# A later patch can rewrite the region an earlier one added -- both still apply, in
# order, but the earlier one's lines are no longer in the tree, so neither check
# above can see it. The later patch declares "Supersedes: <basename>" in its header;
# that only counts if the later patch is itself applied (#67 over #57).
SERIES=()
while IFS= read -r name; do
  SERIES+=("$name")
done < <(sed -e 's/#.*//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//' -e '/^$/d' patches/series)
ON_DISK=$(for f in patches/*.patch; do basename "$f"; done | sort)
IN_SERIES=$(printf '%s\n' "${SERIES[@]}" | sort)
if [ "$ON_DISK" = "$IN_SERIES" ]; then
  ok "patches/series lists all ${#SERIES[@]} patches"
else
  fail "patches/series out of sync with patches/ (a patch not in series is never applied):"
  comm -3 <(printf '%s\n' "$ON_DISK") <(printf '%s\n' "$IN_SERIES") | sed 's/^/    /'
fi
superseded_by() {
  local target="$1" q
  for q in "${SERIES[@]}"; do
    grep -q "^Supersedes: $target\$" "patches/$q" || continue
    patch -p1 -R --dry-run -s --fuzz 0 -d "$SP" < "patches/$q" >/dev/null 2>&1 || $PY patches/_check_applied.py "patches/$q" "$SP" 2>/dev/null || continue
    printf '%s' "$q"; return 0
  done
  return 1
}
# The reverse dry-run is exact, but two patches touching the same file (the DFlash2 pair)
# can no longer be reversed individually once both are applied; then look for their content.
for name in "${SERIES[@]}"; do
  p="patches/$name"
  if [ "$name" = "dflash2-backport.patch" ]; then
    ok "dflash2-backport.patch retired (DFlash2 is native in vLLM 0.28.0)"
    continue
  fi
  if patch -p1 -R --dry-run -s --fuzz 0 -d "$SP" < "$p" >/dev/null 2>&1; then ok "$name applied"
  elif $PY patches/_check_applied.py "$p" "$SP" 2>/dev/null; then ok "$name applied (content check; hunks overlap another patch)"
  elif s=$(superseded_by "$name"); then ok "$name applied (superseded by $s, which is applied)"
  elif patch -p1 -N --dry-run -s --fuzz 0 -d "$SP" < "$p" >/dev/null 2>&1; then fail "$name NOT applied (patch -p1 -d $SP < $p)"
  else fail "$name neither applied nor applicable — vLLM version mismatch?"; fi
done
# Behavioural, not textual: the name must be in the live registry, so a comment or docstring cannot satisfy it
# (a text grep here would; found on the native 3090, 2026-09-13). Negative control: a made-up name exits 1 in the same image.
$PY -c "import vllm.envs as e, sys; sys.exit(0 if 'VLLM_MARLIN_INT8_INCLUDE_RE' in e.environment_variables else 1)" 2>/dev/null && ok "int8 layer-select env vars registered in envs.py (live registry)" || fail "envs.py does not register VLLM_MARLIN_INT8_INCLUDE_RE"

echo "== KVarN (optional, kvarn/)"
if [ -f "$SP/v1/attention/backends/kvarn_attn.py" ]; then
  if patch -p1 -R --dry-run -s --fuzz 0 -d "$SP" < kvarn/kvarn-0.28.0.patch >/dev/null 2>&1; then
    $PY -c "from vllm.v1.attention.backends.registry import AttentionBackendEnum; AttentionBackendEnum.KVARN.get_class()" 2>/dev/null && ok "KVarN backend importable, patch applied (KV=kvarn / CTX=huge available)" || fail "KVarN files present but backend does not import"
  else fail "KVarN modules present but kvarn-0.28.0.patch not applied (bash kvarn/install.sh)"; fi
  if $PY patches/_check_applied.py kvarn/kvarn-v2-runner-0.28.0.patch "$SP" >/dev/null 2>&1; then
    ok "kvarn-v2-runner-0.28.0.patch applied (SPEC=dflash2 + CTX=huge available)"
  else warn "kvarn-v2-runner-0.28.0.patch not applied (re-run bash kvarn/install.sh for DFlash2 at 240k)"; fi
else warn "KVarN not installed (optional; bash kvarn/install.sh for 262k context)"; fi

if [ $INSTALL = 0 ]; then
echo "== model at $MODEL"
if [ ! -f "$MODEL/config.json" ]; then fail "model not found (README Setup: hf download)"; else
$PY - "$MODEL" <<'EOF'
import json, os, sys
d = sys.argv[1].rstrip("/") + "/"
c = json.load(open(d + "config.json"))
qc = c.get("quantization_config", {})
groups = qc.get("config_groups", {})
ign = set(qc.get("ignore", []))
idx = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
F = 0
def ok(m): print("  PASS ", m)
def fail(m):
    global F
    print("  FAIL ", m); F += 1
# Zero points. One check, in two halves; it replaces the head-group block #158 added
# and keeps that block's rule and its advice.
#
# Head groups (lm_head, embed_tokens, mtp) must be symmetric, whether or not their
# weight_zero_point tensors were written. prepare/ writes those tensors symmetric --
# quant_heads_stream.py builds the head groups from group_0 and forces
# symmetric=true / zp_dtype=null on them -- and the paths that read them take no zero
# point: the quantized embedding lookup (CompressedTensorsEmbeddingWNA16Int) registers
# weight_packed, weight_scale and weight_shape and nothing else, and
# build_draft_vocab.py copies the lm_head's packed rows and scales, nothing else, into
# mtp.draft_lm_head. (The failure mode quant_heads_stream.py exists to normalize away,
# and the PR #139 field report.)
#
# Every packed module: vLLM builds the layer from the group it resolves to. An
# asymmetric group registers a weight_zero_point for the checkpoint to fill, and when
# the tensor is not there nothing says so -- the loader's missing-weight check is off
# for quantized models (model_loader/default_loader.py) -- so the layer serves on
# uninitialized zero points. A symmetric group registers none, and a weight_zero_point
# written for it is refused at load ("There is no module or parameter named ...").
# An asymmetric *body* group is legitimate: an AWQ body carries real zero points --
# philbert440/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ, this repo's own worked
# example, is symmetric=false on group_0 and serves fine, and
# patches/marlin-int8-asym-zp.patch runs such bodies on the INT8_ACT=int8 path too.
# The group is resolved the way vLLM's find_matched_target does it: modules in
# "ignore" are skipped, then the first target in config order that names the module
# (exactly, or re: with re.match), else a "Linear" class target -- which never covers
# the embedding or an LM head, since those are not Linear layers in vLLM.
#
# An absent "symmetric" key means symmetric: QuantizationArgs declares
# symmetric: bool = True (compressed_tensors quant_args.py), so a foreign
# export that omits it must not be read as asymmetric.
import re
HEAD_TARGETS = {"re:.*lm_head$", "re:.*embed_tokens$", r"re:^mtp\..*"}
def is_head_group(g):
    return bool(set(g.get("targets") or []) & HEAD_TARGETS)
def is_pack_quantized(g):
    # a group may leave format null and inherit quantization_config["format"]
    return (g.get("format") or qc.get("format")) == "pack-quantized"
def is_asym(g):
    return g.get("weights") is not None and g["weights"].get("symmetric", True) is not True
def names(mod, t):
    return bool(re.match(t[3:], mod)) if t.startswith("re:") else t == mod
def group_of(mod):
    if any(names(mod, t) for t in ign):
        return None
    for name, g in groups.items():
        if any(names(mod, t) for t in g.get("targets") or []):
            return name
    if not mod.endswith(("embed_tokens", "lm_head")):
        for name, g in groups.items():
            if "Linear" in (g.get("targets") or []):
                return name
    return None
packed = [k[:-len(".weight_packed")] for k in idx if k.endswith(".weight_packed")]
of = {m: group_of(m) for m in packed}
asym_head = [name for name, g in groups.items()
             if is_pack_quantized(g) and is_head_group(g) and is_asym(g)]
for name in asym_head:
    tgt = groups[name].get("targets")
    if any(of[m] == name and m + ".weight_zero_point" in idx for m in packed):
        fail(f"head group {name} ({tgt}) declares asymmetric weights and its zero points are written, but the head paths read none: the quantized embedding lookup has no weight_zero_point (vLLM refuses it at load) and build_draft_vocab.py copies none into the draft head. Requantize with prepare/quant_heads_stream.py")
    else:
        fail(f"head group {name} ({tgt}) declares asymmetric weights (zero-point): prepare/ writes those tensors symmetric, so vLLM will look for weight_zero_point tensors that were never written. Requantize with prepare/quant_heads_stream.py")
if not asym_head:
    ok(f"no head group declares zero points ({len(groups)} groups; an asymmetric body group is expected for AWQ exports)")
# the head groups failed above are not re-reported module by module
checked = [m for m in packed if of[m] is not None and of[m] not in asym_head]
zp_bad = [m for m in checked if is_asym(groups[of[m]]) != (m + ".weight_zero_point" in idx)]
for m in zp_bad[:5]:
    if is_asym(groups[of[m]]):
        fail(f"{m}: group {of[m]} declares asymmetric weights but the index has no {m}.weight_zero_point, so vLLM would serve the layer on uninitialized zero points (its missing-weight check is off for quantized models). The export's config and tensors disagree: re-export it (prepare/quant_heads_stream.py rewrites only the head groups)")
    else:
        fail(f"{m}: group {of[m]} is symmetric but the index carries {m}.weight_zero_point, which vLLM refuses at load. The export's config and tensors disagree: re-export it (prepare/quant_heads_stream.py rewrites only the head groups)")
if len(zp_bad) > 5:
    fail(f"... {len(zp_bad)} packed modules whose zero points disagree with their group in total")
if not zp_bad:
    asym_groups = sorted({of[m] for m in checked if is_asym(groups[of[m]])})
    ok(f"zero points match their group on all {len(checked)} packed modules"
       + (" outside those head groups" if asym_head else "")
       + (f" (asymmetric: {', '.join(asym_groups)}; INT8_ACT=int8 runs it through patches/marlin-int8-asym-zp.patch)" if asym_groups else " (all symmetric)"))
# lm_head requantized to int8 (prepare/quant_lm_head.py), or int4-GPTQ as the
# drafter/ pipeline writes it (the shipped ...-AutoRound-fast layout). The width
# is whatever config declares; what must hold is the packed geometry it implies,
# so a config claiming int4 over int8 tensors (or vice versa) still fails here.
lm_g = next((g for g in groups.values() if g.get("targets") == ["re:.*lm_head$"]), None)
lm_bits = lm_g["weights"]["num_bits"] if lm_g else None
if "lm_head.weight_packed" not in idx or lm_bits not in (4, 8):
    fail(f"lm_head not requantized to int4/int8 (prepare/quant_lm_head.py; lm_head group says num_bits={lm_bits})")
else:
    tc = c.get("text_config", c)
    V, K = tc.get("vocab_size"), tc.get("hidden_size")
    sh = {}
    with open(d + idx["lm_head.weight_packed"], "rb") as f:
        import struct
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    for k in ("lm_head.weight_packed", "lm_head.weight_scale"):
        if k in hdr: sh[k] = tuple(hdr[k]["shape"])
    want = {"lm_head.weight_packed": (V, K * lm_bits // 32), "lm_head.weight_scale": (V, K // 128)}
    if sh == want:
        ok(f"lm_head requantized to int{lm_bits}, packed geometry matches the declared width (prepare/quant_lm_head.py / fast variant)")
    else:
        fail(f"lm_head declares int{lm_bits} but packed geometry {sh} != implied {want}")
if any(k.endswith("embed_tokens.weight_packed") for k in idx) and any(g["targets"] == ["re:.*embed_tokens$"] for g in groups.values()): ok("embed_tokens requantized to int8 (prepare/quant_embed.py)")
else: fail("embed_tokens not requantized: run prepare/quant_embed.py")
if "mtp.layers.0.mlp.down_proj.weight_packed" in idx and "mtp.layers.0.mlp.down_proj" not in ign: ok("MTP draft module quantized (prepare/quant_mtp.py)")
else: print("  WARN  MTP module still bf16 (prepare/quant_mtp.py) — single-user mode is slower without it")
if "mtp.draft_lm_head.weight_packed" in idx and os.path.exists(d + "mtp_draft_vocab_ids.pt"): ok("40k-token draft head present (prepare/build_draft_vocab.py)")
else: print("  WARN  draft head missing (prepare/build_draft_vocab.py --ids prepare/draft_vocab_ids.json) — single-user mode drafts with the full lm_head")
missing = [f for f in set(idx.values()) if not os.path.exists(d + f)]
if missing: fail(f"safetensors shards missing: {missing}")
else: ok(f"{len(set(idx.values()))} safetensors shards present")
# duplicate tensors across shards: vLLM reads every key of every shard it opens,
# not just the index-mapped ones, so a tensor that lives in more than its mapped
# shard gets loaded twice — the second load wins, or a shape mismatch aborts the
# boot far from the cause. This happens for real when a model dir is assembled by
# hardlinking shards from the dir it was built from and one of them still carries
# superseded tensors (e.g. an int8 MTP module surviving inside a hardlinked
import struct
def keys_of(f):
    with open(d + f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return set(k for k in json.loads(fh.read(n)) if k != "__metadata__")
holders = {}
for f in (f for f in os.listdir(d) if f.endswith(".safetensors") and ".bak" not in f):
    for k in keys_of(f):
        holders.setdefault(k, []).append(f)
colliding = [(name, hs) for name, hs in
             ((name, holders.get(name, [])) for name in idx) if hs != [idx[name]]]
if colliding:
    for name, hs in colliding:
        fail(f"tensor {name} present in {hs}, index maps {idx[name]}")
else:
    ok(f"every index tensor lives in exactly its mapped shard (no duplicates across {len(set(idx.values()))} mapped + stray files)")
sys.exit(1 if F else 0)
EOF
[ $? -ne 0 ] && FAILS=$((FAILS+1))
fi

echo "== single-user fast variant (optional)"
if [ -d "$HERE/models/Qwen3.8-27B-W4A16-AutoRound-fast" ]; then ok "fast variant present (int4-GPTQ lm_head/MTP, own-output draft vocab)"; else warn "no models/Qwen3.8-27B-W4A16-AutoRound-fast (venv/bin/python prepare/fetch_fast_variant.py; single-user mode is ~15% slower without it)"; fi

if [ $INSTALL = 0 ]; then
# A served model dir with no tokenizer.json is not an error to transformers: it hands
# back a Qwen2Tokenizer with a 1-token vocabulary that encodes everything to []. The
# server then dies far downstream on "ReasoningConfig: failed to tokenize reasoning
# strings", which names neither the dir nor the tokenizer. Encode <think> here instead.
echo "== tokenizers (every dir we serve --model from)"
$PY - "$MODEL" "$HERE/models/Qwen3.8-27B-W4A16-AutoRound-fast" <<'EOF'
import os, sys
F = 0
for d in sys.argv[1:]:
    if not os.path.isfile(os.path.join(d, "config.json")):
        continue
    name = os.path.basename(d.rstrip("/"))
    try:
        from transformers import AutoTokenizer
        ids = AutoTokenizer.from_pretrained(d).encode("<think>", add_special_tokens=False)
    except Exception as e:
        print(f"  FAIL  {name}: tokenizer will not load ({type(e).__name__}: {str(e)[:70]})"); F += 1; continue
    if ids:
        print(f"  PASS  {name}: tokenizer loads, <think> -> {ids}")
    else:
        print(f"  FAIL  {name}: no usable tokenizer in the dir (encodes everything to []). "
              f"Copy tokenizer.json and tokenizer_config.json in from the base model dir, "
              f"or re-run the download. Serving this dir fails with "
              f"'ReasoningConfig: failed to tokenize reasoning strings'.")
        F += 1
sys.exit(1 if F else 0)
EOF
[ $? -ne 0 ] && FAILS=$((FAILS+1))
fi
echo "== single-user DFlash2 drafter (optional, SPEC=dflash2)"
if [ -f "$HERE/models/Qwen3.8-27B-DFlash2-W4A16/config.json" ]; then
  $PY -c "import json,sys; c=json.load(open('$HERE/models/Qwen3.8-27B-DFlash2-W4A16/config.json')); assert c['architectures']==['DFlash2DraftModel'] and c['quantization_config']['quant_method']=='compressed-tensors'" 2>/dev/null && ok "DFlash2 drafter present, W4A16 (models/Qwen3.8-27B-DFlash2-W4A16)" || fail "models/Qwen3.8-27B-DFlash2-W4A16 is not a quantized DFlash2DraftModel checkpoint"
  [ -f "$SP/model_executor/models/qwen3_dflash2.py" ] || fail "DFlash2 drafter present but vLLM 0.28.0 native DFlash2 support is missing"
elif [ -f "$HERE/models/Qwen3.8-27B-DFlash2/config.json" ]; then warn "only the bf16 DFlash2 drafter is present (3.85 GB; venv/bin/python prepare/fetch_dflash2.py for the 1 GB W4A16 one)"
else warn "no DFlash2 drafter (venv/bin/python prepare/fetch_dflash2.py; SPEC=dflash2 single-user mode needs it)"; fi

echo "== keys / units"
# A key is optional: with neither api_key.txt nor VLLM_API_KEY the launchers export
# nothing and vLLM serves unauthenticated, which is a fine way to run this locally.
# Worth a WARN rather than silence only because both launchers bind 0.0.0.0.
[ -s api_key.txt ] || [ -n "${VLLM_API_KEY:-}" ] && ok "API key configured (api_key.txt or VLLM_API_KEY)" \
  || warn "no API key — the server will accept any request, and it listens on 0.0.0.0. Fine behind a firewall; otherwise: openssl rand -hex 24 > api_key.txt"
if [ -f /.dockerenv ]; then :; elif systemctl --user is-active qwen-serving >/dev/null 2>&1; then ok "systemd user unit qwen-serving active"; else warn "qwen-serving unit not active (fine if you launch the scripts by hand)"; fi
fi  # INSTALL

if [ $NOSRV = 0 ]; then
  echo "== live server (127.0.0.1:${PORT:-18020})"
  PORT=${PORT:-18020}
  if curl -sf -o /dev/null http://127.0.0.1:$PORT/health; then
    ok "/health 200"
    KEY=${VLLM_API_KEY:-$(cat api_key.txt 2>/dev/null)}
    R=$(curl -s http://127.0.0.1:$PORT/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hvad er hovedstaden i Danmark? Svar med ét ord."}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}')
    echo "$R" | grep -qi "københavn\|copenhagen" && ok "chat completion answers ('$(echo "$R" | $PY -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>/dev/null)')" || fail "chat completion wrong/failed: $(echo "$R" | head -c 200)"
    LOG=$HERE/qwen.log
    if [ -f "$LOG" ]; then
      grep -oE "Using [A-Z_]+ attention backend" "$LOG" | tail -1 | sed 's/^/  INFO  /'
      grep -oE "GPU KV cache size: [0-9,]+ tokens" "$LOG" | tail -1 | sed 's/^/  INFO  /'
      grep -oE "Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" "$LOG" | tail -1 | sed 's/^/  INFO  /'
      grep -q "MarlinLinearKernel" "$LOG" && ok "Marlin kernels in use" || true
      grep -oE "capping max_num_seqs [0-9]+ -> [0-9]+" "$LOG" | tail -1 | sed 's/^/  INFO  KVarN /'
    fi
  else warn "no server on :$PORT (start batch/start_qwen.sh or single-user/start_qwen.sh, or pass --no-server)"; fi
fi
echo
[ $FAILS = 0 ] && echo "verify: OK ($FAILS failures)" || echo "verify: $FAILS FAILURE(S)"
exit $([ $FAILS = 0 ] && echo 0 || echo 1)
