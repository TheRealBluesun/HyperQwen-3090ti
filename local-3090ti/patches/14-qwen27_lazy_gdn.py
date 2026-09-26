"""qwen27-dev: lazy GDN state commit for spec decode (VLLM_QWEN27_LAZY_GDN=1).

A spec-decode step of M tokens normally stores the recurrent state after every token (M x 1.57 MB per GDN layer)
so the next step can start from whichever one was accepted. A lazy step stores only the state it started from
(the "base", in the request's first spec slot) plus a 133 KB log of its tokens' raw k, v, a, b; the next step
replays the accepted ones from the base (qwen27_gdn_rec.cu). Every state it computes is bit-identical to
what the full step would have stored.

Everything outside that kernel still expects the full layout (the align-mode block copies before and after a
step, non-spec decode, prefill), so:
  * a step runs lazy only when no block boundary lies near it (the align copies only fire at boundaries),
  * a request whose previous step was lazy and that will be read by anything else first is "materialized":
    its lazy step is replayed and every per-token state stored where the full step would have put it.
Runner side: prepare() (before preprocess_mamba) decides per request and publishes per-row mode/log ids;
the GDN metadata builder copies them (spec rows only) into persistent buffers the layers read (graph safe).
"""
import itertools
import os

import numpy as np
import torch

ENABLED = os.environ.get("VLLM_QWEN27_LAZY_GDN", "0") == "1"
FORCE_FULL = os.environ.get("VLLM_QWEN27_LAZY_GDN_FORCE_FULL", "0") == "1"   # new kernel, full layout (A/B)
FULL_EVERY = int(os.environ.get("VLLM_QWEN27_LAZY_GDN_FULL_EVERY", "0"))       # test: force periodic full steps
MAT_EVERY = int(os.environ.get("VLLM_QWEN27_LAZY_GDN_MAT_EVERY", "0"))         # test: force periodic materialize


def check_config(vllm_config):
    """Turn the feature off when the verify block can exceed 8 tokens: the per-slot log holds 8 tokens per
    region and the replay kernel assumes it, so a longer block would break acceptance (measured: DFLASH_TOKENS=15
    accepted only the first draft)."""
    global ENABLED
    spec = getattr(vllm_config, "speculative_config", None)
    k = getattr(spec, "num_speculative_tokens", 0) or 0
    if ENABLED and k + 1 > 8:
        import logging
        logging.getLogger("vllm").warning(
            "qwen27 lazy GDN: disabled, the verify block is %d tokens (supports <= 8)", k + 1)
        ENABLED = False
    return ENABLED
MARGIN_LO, MARGIN_HI = 8, 16          # V1 (exact positions)
V2_MARGIN_LO, V2_MARGIN_HI = 32, 24   # V2: num_computed_tokens_np is an optimistic mirror (async scheduling)
MAXB = 256
WMAX = 16

_SRC = os.path.dirname(os.path.abspath(__file__))
_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        _EXT = load(name="qwen27_gdn_lazy", sources=[os.path.join(_SRC, "qwen27_gdn_rec.cu")],
                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_86,code=sm_86", "-std=c++17"])
    return _EXT


# ---- device-side per-spec-row buffers (persistent: FULL cudagraphs bake their addresses)
_dev = {}


def dev_bufs(device):
    key = str(device)
    if key not in _dev:
        _dev[key] = (torch.zeros(MAXB, dtype=torch.int32, device=device), torch.zeros(MAXB, dtype=torch.int32, device=device))
    return _dev[key]


def alloc_layer_log(layer, nlog, device=None):
    """Per-layer log: [nlog][2 regions][8 tokens][k | v | a | b] bf16 + header [nlog][2 + WMAX] int32."""
    tok = layer.num_k_heads // layer.tp_size * layer.head_k_dim + layer.num_v_heads // layer.tp_size * layer.head_v_dim \
        + 2 * (layer.num_v_heads // layer.tp_size)
    layer._lazy_log = torch.zeros(nlog, 2, 8, tok, dtype=torch.bfloat16, device=device)
    layer._lazy_hdr = torch.zeros(nlog, 2 + WMAX, dtype=torch.int32, device=device)


# ---- runner-side bookkeeping
class _Req:
    __slots__ = ("log_id", "parity", "prev_lazy")

    def __init__(self, log_id):
        self.log_id, self.parity, self.prev_lazy = log_id, 0, False


_reqs: dict = {}
_free: list | None = None
_rows = None          # (mode np.int32[num_reqs], log np.int32[num_reqs]) for the step being prepared, or None
_block_size = None
_layers = None
_nsteps = 0
stats = dict(steps=0, lazy_rows=0, full_rows=0, materialized=0)


def _drop(rid):
    r = _reqs.pop(rid, None)
    if r is not None:
        _free.append(r.log_id)


def reset_rows():
    """Dummy / profile / capture runs: no lazy work (their metadata points at null slots anyway)."""
    global _rows
    _rows = None


def _gdn_layers(runner):
    global _layers
    if _layers is None:
        _layers = [m for m in runner.compilation_config.static_forward_context.values() if hasattr(m, "_lazy_log")]
    return _layers


_nprep = [0]


def prepare(runner, scheduler_output, num_reqs):
    global _free, _rows, _block_size, _nsteps
    if _free is None:
        nlog = runner.scheduler_config.max_num_seqs + 2
        _free = list(range(nlog))
    if _block_size is None:
        from vllm.v1.worker import mamba_utils
        _block_size = {s.block_size for s in mamba_utils.get_mamba_groups(runner.kv_cache_config)}.pop()
    bs = _block_size
    resumed = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    for rid in itertools.chain(scheduler_output.finished_req_ids, scheduler_output.preempted_req_ids or (), resumed):
        _drop(rid)
    _nsteps += 1
    spec = scheduler_output.scheduled_spec_decode_tokens
    mode = np.zeros(num_reqs, dtype=np.int32)
    logs = np.zeros(num_reqs, dtype=np.int32)
    mat_ids, mat_par = [], []
    for i, rid in enumerate(runner.input_batch.req_ids[:num_reqs]):
        r = _reqs.get(rid)
        nd = len(spec.get(rid, ()))
        m = scheduler_output.num_scheduled_tokens[rid]
        pos = runner.requests[rid].num_computed_tokens
        kernel = nd > 0 and m == nd + 1 and m <= 8
        prev_idx = runner.mamba_state_idx.get(rid)
        copies = prev_idx is not None and prev_idx != (pos + m + bs - 1) // bs - 1   # preprocess_mamba will copy
        if r is not None and r.prev_lazy and (not kernel or copies or (MAT_EVERY and _nsteps % MAT_EVERY == 0)):
            mat_ids.append(r.log_id); mat_par.append(r.parity); r.prev_lazy = False
        lazy = (kernel and not FORCE_FULL and pos >= MARGIN_LO
                and (pos - MARGIN_LO) // bs == (pos + m + MARGIN_HI) // bs
                and not (FULL_EVERY and _nsteps % FULL_EVERY == 0))
        if lazy and r is None:
            if _free:
                r = _reqs[rid] = _Req(_free.pop())
            else:
                lazy = False
        if r is not None:
            mode[i] = (1 if (r.prev_lazy and kernel) else 0) | (2 if lazy else 0) | (4 if r.parity else 0)
            logs[i] = r.log_id
            if lazy:
                r.parity ^= 1
            r.prev_lazy = lazy
        stats["lazy_rows" if lazy else "full_rows"] += 1
    stats["steps"] += 1
    if stats["steps"] % 500 == 1:
        with open(f"/tmp/qwen27_lazy_gdn.{os.getpid()}.log", "a") as f:
            f.write(f"{stats}\n")
    if mat_ids:
        stats["materialized"] += len(mat_ids)
        dev = runner.device
        ids = torch.tensor(mat_ids, dtype=torch.int32, pin_memory=True).to(dev, non_blocking=True)
        par = torch.tensor(mat_par, dtype=torch.int32, pin_memory=True).to(dev, non_blocking=True)
        e = ext()
        for L in _gdn_layers(runner):
            e.gdn_materialize(L.A_log, L.dt_bias, L.kv_cache[1], L._lazy_log, L._lazy_hdr, ids, par,
                              L.num_k_heads // L.tp_size)
    _rows = (mode, logs)


_npub = [0, 0]


def publish(spec_mask_cpu, num_reqs, device):
    """GDN metadata build: spec rows' mode/log ids -> persistent device buffers (every builder of the step
    writes the same values). Rows past the spec rows keep stale values; their token counts are 0."""
    mbuf, lbuf = dev_bufs(device)
    if _rows is None or len(_rows[0]) < num_reqs:
        mbuf.zero_()
        return
    mask = spec_mask_cpu.numpy()[:num_reqs] if spec_mask_cpu is not None else np.zeros(num_reqs, dtype=bool)
    md, lg = _rows[0][:num_reqs][mask], _rows[1][:num_reqs][mask]
    n = len(md)
    if n == 0:
        mbuf.zero_()
        return
    both = torch.from_numpy(np.concatenate([md, lg])).pin_memory().to(device, non_blocking=True)
    mbuf[:n].copy_(both[:n], non_blocking=True)
    lbuf[:n].copy_(both[n:], non_blocking=True)


# ---- V2 model runner (vllm/v1/worker/gpu): state keyed by the persistent request-state slot (idx_mapping),
#      which doubles as the log id. add_request() resets a slot for every new / resumed request.
_v2_prev = None      # np.bool_[nslots]
_v2_par = None       # np.int8[nslots]


def reset_slot(idx):
    if _v2_prev is not None:
        _v2_prev[idx] = False
        _v2_par[idx] = 0


def prepare_v2(model_state, input_batch, kv_cache_config):
    global _rows, _v2_prev, _v2_par, _nsteps, _block_size
    if _v2_prev is None:
        _v2_prev = np.zeros(model_state.max_num_reqs, dtype=np.bool_)
        _v2_par = np.zeros(model_state.max_num_reqs, dtype=np.int8)
    if _block_size is None:
        from vllm.v1.worker import mamba_utils
        _block_size = {s.block_size for s in mamba_utils.get_mamba_groups(kv_cache_config)}.pop()
    bs = _block_size
    _nsteps += 1
    n = input_batch.num_reqs
    slots = input_batch.idx_mapping_np[:n]
    sched = input_batch.num_scheduled_tokens[:n]
    nd = input_batch.num_draft_tokens_per_req[:n] if input_batch.num_draft_tokens_per_req is not None else np.zeros(n, np.int64)
    pos = input_batch.num_computed_tokens_np[:n]
    kernel = (nd > 0) & (sched == nd + 1) & (sched <= 8)          # same rows GDN treats as spec decode
    lazy = kernel & (pos >= V2_MARGIN_LO) & ((pos - V2_MARGIN_LO) // bs == (pos + sched + V2_MARGIN_HI) // bs)
    L0 = _gdn_layers_v2(model_state)
    lazy &= slots < (L0[0]._lazy_log.shape[0] if L0 else 0)          # a log per slot
    if FORCE_FULL or (FULL_EVERY and _nsteps % FULL_EVERY == 0):
        lazy[:] = False
    prev = _v2_prev[slots]
    mat = prev & (~kernel | (MAT_EVERY > 0 and _nsteps % MAT_EVERY == 0))
    if mat.any():
        ids = slots[mat].astype(np.int32)
        par = _v2_par[ids].astype(np.int32)
        stats["materialized"] += len(ids)
        dev = model_state.device
        ids_t = torch.from_numpy(ids).pin_memory().to(dev, non_blocking=True)
        par_t = torch.from_numpy(par).pin_memory().to(dev, non_blocking=True)
        e = ext()
        for L in _gdn_layers_v2(model_state):
            e.gdn_materialize(L.A_log, L.dt_bias, L.kv_cache[1], L._lazy_log, L._lazy_hdr, ids_t, par_t,
                              L.num_k_heads // L.tp_size)
        prev = prev & ~mat
    mode = ((prev & kernel).astype(np.int32) | (lazy.astype(np.int32) << 1) | (_v2_par[slots].astype(np.int32) << 2))
    _v2_par[slots] ^= lazy.astype(np.int8)
    _v2_prev[slots] = lazy
    stats["steps"] += 1
    stats["lazy_rows"] += int(lazy.sum()); stats["full_rows"] += int((~lazy).sum())
    if stats["steps"] % 20000 in (1, 2001):
        with open(f"/tmp/qwen27_lazy_gdn.{os.getpid()}.log", "a") as f:
            f.write(f"{stats}\n")
    _rows = (mode.astype(np.int32), slots.astype(np.int32))


def _gdn_layers_v2(model_state):
    global _layers
    if _layers is None:
        _layers = [m for m in model_state.vllm_config.compilation_config.static_forward_context.values()
                   if hasattr(m, "_lazy_log")]
    return _layers
