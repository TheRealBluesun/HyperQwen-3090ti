// Spec-decode gated delta rule (drop-in for vLLM's fused_sigmoid_gating_delta_rule_update on the spec path,
// use_qk_l2norm_in_kernel=True, inplace_final_state=True, IS_KDA=False), plus "lazy state commit".
//
//   per request n (tokens bos..eos, T <= 8), value head hv, state row vi (a V index; K is contiguous):
//     h0 = state[idx[n][nacc-1]]  (zero output + no writes when that slot is <= 0, like FLA)
//     per token t: g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b), q,k l2-normalized, q *= scale
//                  h *= exp(g); v' = (v - h.k) * beta; h += v' k^T; o = h.q; o -> bf16; state[idx[n][t]] = fp16(h)
//
// Lazy commit (per-request mode bits): instead of writing T states (12.6 MB per layer at T = 8), a lazy step
// writes fp16(h0) to idx[n][0] (the "base") and logs its tokens' raw k, v, a, b (133 KB). The next step replays
// its first nacc logged tokens from the base -- the same fp32 arithmetic the full step would have run, so the
// state it reaches is bit-identical to what the full step would have stored in idx[n][nacc-1].
//   mode bit 0 (PREV_LAZY): h0 = replay(state[idx[n][0]], log[parity], nacc tokens) instead of state[idx[n][nacc-1]]
//   mode bit 1 (THIS_LAZY): write base + log[parity ^ 1] + header, no per-token states
//   mode bit 2: parity
// materialize() turns a lazy step's leftovers back into the full layout (every per-token state in its slot) for
// any consumer other than this kernel (block-boundary copies, non-spec paths).
// Layout: CTA = RB rows of one head of one request; 16 lanes per row (8 k each). Every input of the CTA is
// loaded in one prologue (the FLA kernel's per-token dependent loads are what make it slow).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace gr {
constexpr int DH = 128, TMAX = 8, WMAX = 16;
constexpr int LPR = 8;                                   // lanes per row (16 k each)
constexpr int NWARP = 4, THREADS = NWARP * 32, RB = NWARP * (32 / LPR);   // 16 rows per CTA
constexpr int PREV_LAZY = 1, THIS_LAZY = 2, PARITY = 4;

struct Log {                         // per-layer log buffer, indexed by a per-request log id
  __nv_bfloat16* buf;                // [nlog][2][TMAX][tok]: tok = k[HK*DH] | v[HV*DH] | a[HV] | b[HV]
  int* hdr;                          // [nlog][2 + WMAX]: T, W, slot ids of the lazy step
  int tok;                           // elements per logged token
};

struct P {
  const __nv_bfloat16 *q, *k, *v;    // token-strided views: q[t * sq + h * DH + d]
  long long sq, sk, sv;
  const __nv_bfloat16 *a, *b;        // [T][HV], token strides sa / sb
  long long sa, sb;
  const float* A_log; const __nv_bfloat16* dt_bias;
  __half* state; long long sstride;  // [slots][HV][DH][DH]
  const int* sidx; int sidx_w;       // [N][sidx_w]
  const int* nacc; const int* cu;    // [N], [N + 1]
  __nv_bfloat16* o;                  // [T][HV][DH]
  int HK, HV; float scale;
  const int* mode; const int* log_id;   // [N] (nullptr: plain mode)
  Log lg;
};

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }

// Normalize one token's 128-vector across a warp (4 elements per lane), exactly as the main path does.
__device__ __forceinline__ void l2n_warp(const __nv_bfloat16* src, float mul, float* dst, int lane) {
  const uint2 raw = *reinterpret_cast<const uint2*>(src + lane * 4);
  const __nv_bfloat16* b4 = reinterpret_cast<const __nv_bfloat16*>(&raw);
  float x[4], s = 0.f;
#pragma unroll
  for (int e = 0; e < 4; e++) { x[e] = bf(b4[e]); s += x[e] * x[e]; }
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  const float r = rsqrtf(s + 1e-6f) * mul;
#pragma unroll
  for (int e = 0; e < 4; e++) dst[lane * 4 + e] = x[e] * r;
}
__device__ __forceinline__ void gate(float a, float b, const P& p, int hv, float* eg, float* bt) {
  const float x = a + bf(p.dt_bias[hv]);
  const float sp = x <= 20.f ? logf(1.f + expf(x)) : x;   // same form as FLA
  *eg = expf(-expf(p.A_log[hv]) * sp);
  *bt = 1.f / (1.f + expf(-b));
}
// the recurrence without the output (replay) / with it
__device__ __forceinline__ void ld16(float* x, const float* src, int j) {
#pragma unroll
  for (int c = 0; c < 4; c++) *reinterpret_cast<float4*>(x + 4 * c) = *reinterpret_cast<const float4*>(src + 4 * j + 32 * c);
}
__device__ __forceinline__ float red8(float x) {
  x += __shfl_xor_sync(0xffffffffu, x, 4); x += __shfl_xor_sync(0xffffffffu, x, 2); x += __shfl_xor_sync(0xffffffffu, x, 1);
  return x;
}
__device__ __forceinline__ void step_h(float* h, float e_g, float be, float v, const float* kv) {
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
#pragma unroll
  for (int e = 0; e < 16; e += 4) {
    h[e] *= e_g; h[e + 1] *= e_g; h[e + 2] *= e_g; h[e + 3] *= e_g;
    d0 += h[e] * kv[e]; d1 += h[e + 1] * kv[e + 1]; d2 += h[e + 2] * kv[e + 2]; d3 += h[e + 3] * kv[e + 3];
  }
  const float d = red8((d0 + d1) + (d2 + d3));
  const float vv = (v - d) * be;
#pragma unroll
  for (int e = 0; e < 16; e++) h[e] += vv * kv[e];
}
__device__ __forceinline__ float out_h(const float* h, const float* qv) {
  float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
#pragma unroll
  for (int e = 0; e < 16; e += 4) { o0 += h[e] * qv[e]; o1 += h[e + 1] * qv[e + 1]; o2 += h[e + 2] * qv[e + 2]; o3 += h[e + 3] * qv[e + 3]; }
  return red8((o0 + o1) + (o2 + o3));
}
__device__ __forceinline__ void load_h(float* h, const __half* row, int j) {
#pragma unroll
  for (int c = 0; c < 4; c++) {
    const uint2 r = *reinterpret_cast<const uint2*>(row + 4 * j + 32 * c);
    const __half2* a = reinterpret_cast<const __half2*>(&r);
    const float2 f0 = __half22float2(a[0]), f1 = __half22float2(a[1]);
    h[4 * c] = f0.x; h[4 * c + 1] = f0.y; h[4 * c + 2] = f1.x; h[4 * c + 3] = f1.y;
  }
}
__device__ __forceinline__ void store_h(const float* h, __half* row, int j) {
#pragma unroll
  for (int c = 0; c < 4; c++) {
    uint2 r; __half2* a = reinterpret_cast<__half2*>(&r);
    a[0] = __floats2half2_rn(h[4 * c], h[4 * c + 1]); a[1] = __floats2half2_rn(h[4 * c + 2], h[4 * c + 3]);
    *reinterpret_cast<uint2*>(row + 4 * j + 32 * c) = r;
  }
}
__device__ __forceinline__ void round_h(float* h) {
#pragma unroll
  for (int e = 0; e < 16; e += 2) { const float2 f = __half22float2(__floats2half2_rn(h[e], h[e + 1])); h[e] = f.x; h[e + 1] = f.y; }
}

// Replay set: normalized k, the CTA's v rows, gates of R logged tokens (log region `reg`), loaded into smem.
struct Rep { float kn[TMAX][DH]; float eg[TMAX], bt[TMAX]; };
__device__ __forceinline__ void load_rep(const P& p, const __nv_bfloat16* reg, int R, int kh, int hv, Rep& s, int warp, int lane) {
  const int HKD = p.HK * DH, HVD = p.HV * DH;
  for (int t = warp; t < R; t += NWARP) l2n_warp(reg + (long long)t * p.lg.tok + kh * DH, 1.f, s.kn[t], lane);
  if (threadIdx.x >= THREADS - 32 && lane < R) {
    const __nv_bfloat16* e = reg + (long long)lane * p.lg.tok + HKD + HVD;
    gate(bf(e[hv]), bf(e[p.HV + hv]), p, hv, &s.eg[lane], &s.bt[lane]);
  }
}
// v of one row for up to TMAX tokens, into registers
__device__ __forceinline__ void load_v(float* vr, const __nv_bfloat16* base, long long stride, int T) {
#pragma unroll
  for (int t = 0; t < TMAX; t++) vr[t] = t < T ? bf(base[t * stride]) : 0.f;
}

__global__ void __launch_bounds__(THREADS) rec_kernel(P p) {
  const int rb = blockIdx.x, hv = blockIdx.y, n = blockIdx.z;
  const int bos = p.cu[n], T = p.cu[n + 1] - bos;
  if (T <= 0) return;
  const int kh = hv / (p.HV / p.HK);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, j = lane % LPR;
  const int vi = rb * RB + warp * (32 / LPR) + lane / LPR;
  __shared__ __align__(16) float qn[TMAX][DH], kn[TMAX][DH];
  __shared__ float eg[TMAX], bt[TMAX];
  __shared__ __align__(16) Rep rep;
  const int mode = p.mode ? p.mode[n] : 0;
  const int* srow = p.sidx + n * p.sidx_w;
  const long long rowoff = ((long long)hv * DH + vi) * DH;
  const int HKD = p.HK * DH, HVD = p.HV * DH;

  const int nacc = p.nacc[n];
  int R = 0, slot_in;
  float rv[TMAX];
  if (mode & PREV_LAZY) {
    slot_in = srow[0];
    R = nacc;
    const __nv_bfloat16* rreg = p.lg.buf + ((long long)p.log_id[n] * 2 + ((mode & PARITY) ? 1 : 0)) * TMAX * p.lg.tok;
    load_rep(p, rreg, R, kh, hv, rep, warp, lane);
    load_v(rv, rreg + HKD + hv * DH + vi, p.lg.tok, R);
  } else {
    slot_in = (nacc >= 1 && nacc <= p.sidx_w) ? srow[nacc - 1] : 0;
  }
  for (int t = warp; t < T; t += NWARP) {
    l2n_warp(p.q + (bos + t) * p.sq + kh * DH, p.scale, qn[t], lane);
    l2n_warp(p.k + (bos + t) * p.sk + kh * DH, 1.f, kn[t], lane);
  }
  if (threadIdx.x >= THREADS - 32 && lane < T)
    gate(bf(p.a[(bos + lane) * p.sa + hv]), bf(p.b[(bos + lane) * p.sb + hv]), p, hv, &eg[lane], &bt[lane]);
  float vr[TMAX];
  load_v(vr, p.v + bos * p.sv + hv * DH + vi, p.sv, T);
  float h[16];
  if (slot_in > 0) load_h(h, p.state + slot_in * p.sstride + rowoff, j);
  if ((mode & THIS_LAZY) && slot_in > 0) {   // null-slot rows (padding, dummy runs) touch nothing
    __nv_bfloat16* wreg = p.lg.buf + ((long long)p.log_id[n] * 2 + ((mode & PARITY) ? 0 : 1)) * TMAX * p.lg.tok;
    if (j == 0) for (int t = 0; t < T; t++) wreg[(long long)t * p.lg.tok + HKD + hv * DH + vi] = p.v[(bos + t) * p.sv + hv * DH + vi];
    if (rb == 0) {
      if (threadIdx.x < T) {
        const int t = threadIdx.x;
        wreg[(long long)t * p.lg.tok + HKD + HVD + hv] = p.a[(bos + t) * p.sa + hv];
        wreg[(long long)t * p.lg.tok + HKD + HVD + p.HV + hv] = p.b[(bos + t) * p.sb + hv];
      }
      if (hv % (p.HV / p.HK) == 0) {
        for (int i = threadIdx.x; i < T * DH; i += THREADS) {
          const int t = i / DH, d = i % DH;
          wreg[(long long)t * p.lg.tok + kh * DH + d] = p.k[(bos + t) * p.sk + kh * DH + d];
        }
      }
      if (hv == 0 && threadIdx.x <= p.sidx_w + 1) {
        int* hd = p.lg.hdr + p.log_id[n] * (2 + WMAX);
        hd[threadIdx.x] = threadIdx.x == 0 ? T : threadIdx.x == 1 ? p.sidx_w : srow[threadIdx.x - 2];
      }
    }
  }
  __syncthreads();

  __nv_bfloat16* op = p.o + ((long long)bos * p.HV + hv) * DH + vi;
  const long long ostride = (long long)p.HV * DH;
  if (slot_in <= 0) {
    if (j == 0) for (int t = 0; t < T; t++) op[t * ostride] = __float2bfloat16(0.f);
    return;
  }
#pragma unroll
  for (int t = 0; t < TMAX; t++) {
    if (t < R) { float kv[16]; ld16(kv, rep.kn[t], j); step_h(h, rep.eg[t], rep.bt[t], rv[t], kv); }
  }
  if (mode & PREV_LAZY) round_h(h);   // the full path stores/reloads the state as fp16 at this point
  if (mode & THIS_LAZY) store_h(h, p.state + srow[0] * p.sstride + rowoff, j);
  const bool full = !(mode & THIS_LAZY);
#pragma unroll
  for (int t = 0; t < TMAX; t++) {
    if (t < T) {
      float kv[16], qv[16];
      ld16(kv, kn[t], j); ld16(qv, qn[t], j);
      step_h(h, eg[t], bt[t], vr[t], kv);
      const float ov = out_h(h, qv);
      if (j == 0) op[t * ostride] = __float2bfloat16(ov);
      if (full) { const int so = srow[t]; if (so > 0) store_h(h, p.state + so * p.sstride + rowoff, j); }
    }
  }
}

// materialize: for each listed log id, replay the lazy step from its base and store every per-token state in
// the slots that step used (the full layout). reg_parity[i]: region the lazy step wrote.
__global__ void __launch_bounds__(THREADS) mat_kernel(P p, const int* ids, const int* reg_parity) {
  const int rb = blockIdx.x, hv = blockIdx.y, i0 = blockIdx.z;
  const int id = ids[i0];
  const int* hd = p.lg.hdr + id * (2 + WMAX);
  const int T = hd[0];
  const int* slots = hd + 2;
  const int kh = hv / (p.HV / p.HK);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, j = lane % LPR;
  const int vi = rb * RB + warp * (32 / LPR) + lane / LPR;
  __shared__ __align__(16) Rep rep;
  const __nv_bfloat16* reg = p.lg.buf + ((long long)id * 2 + reg_parity[i0]) * TMAX * p.lg.tok;
  load_rep(p, reg, T, kh, hv, rep, warp, lane);
  float rv[TMAX];
  load_v(rv, reg + p.HK * DH + hv * DH + vi, p.lg.tok, T);
  const long long rowoff = ((long long)hv * DH + vi) * DH;
  const int base = slots[0];
  float h[16];
  if (base > 0) load_h(h, p.state + base * p.sstride + rowoff, j);
  __syncthreads();
  if (base <= 0) return;
#pragma unroll
  for (int t = 0; t < TMAX; t++) {
    if (t < T) {
      float kv[16]; ld16(kv, rep.kn[t], j);
      step_h(h, rep.eg[t], rep.bt[t], rv[t], kv);
      if (slots[t] > 0) store_h(h, p.state + slots[t] * p.sstride + rowoff, j);
    }
  }
}
}  // namespace gr

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

static gr::P make_p(torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor state, int HK, int HV) {
  gr::P p{};
  // the slot dim may be padded (vLLM pads mamba pages); each slot's [HV][DH][DH] block must be dense
  TORCH_CHECK(state.scalar_type() == at::kHalf && state.dim() == 4 && state.size(-1) == gr::DH && state.size(-2) == gr::DH);
  TORCH_CHECK(state.stride(3) == 1 && state.stride(2) == gr::DH && state.stride(1) == gr::DH * gr::DH && state.stride(0) % 8 == 0);
  TORCH_CHECK(A_log.scalar_type() == at::kFloat && dt_bias.scalar_type() == at::kBFloat16);
  p.A_log = A_log.data_ptr<float>(); p.dt_bias = reinterpret_cast<const __nv_bfloat16*>(dt_bias.data_ptr());
  p.state = reinterpret_cast<__half*>(state.data_ptr()); p.sstride = state.stride(0);
  p.HK = HK; p.HV = HV;
  return p;
}
static void set_log(gr::P& p, torch::Tensor logbuf, torch::Tensor loghdr) {
  TORCH_CHECK(logbuf.scalar_type() == at::kBFloat16 && logbuf.dim() == 4 && logbuf.size(1) == 2 && logbuf.size(2) == gr::TMAX);
  TORCH_CHECK(loghdr.scalar_type() == at::kInt && loghdr.size(1) == 2 + gr::WMAX);
  p.lg.buf = reinterpret_cast<__nv_bfloat16*>(logbuf.data_ptr()); p.lg.hdr = loghdr.data_ptr<int>(); p.lg.tok = logbuf.size(3);
  TORCH_CHECK(p.lg.tok == p.HK * gr::DH + p.HV * gr::DH + 2 * p.HV);
}

// q, k: [1, T, HK, DH] (token stride free), v: [1, T, HV, DH]; a, b: [T, HV]; state [slots, HV, DH, DH] fp16;
// sidx [N, W] int32; nacc [N] int32; cu [N + 1] int32; o [T, HV, DH] bf16 (written).
// mode/log_id [N] int32 and logbuf [nlog, 2, TMAX, tok] bf16 / loghdr [nlog, 2 + WMAX] int32: lazy commit (mode may be empty).
void gdn_rec(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b, torch::Tensor A_log,
             torch::Tensor dt_bias, torch::Tensor state, torch::Tensor sidx, torch::Tensor nacc, torch::Tensor cu,
             torch::Tensor o, double scale, torch::Tensor mode, torch::Tensor log_id, torch::Tensor logbuf, torch::Tensor loghdr) {
  TORCH_CHECK(q.size(-1) == gr::DH && v.size(-1) == gr::DH);
  TORCH_CHECK(q.stride(-1) == 1 && q.stride(-2) == gr::DH && k.stride(-1) == 1 && k.stride(-2) == gr::DH && v.stride(-1) == 1 && v.stride(-2) == gr::DH);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.stride(1) == 1 && b.stride(1) == 1 && sidx.is_contiguous() && o.is_contiguous());
  TORCH_CHECK(sidx.size(1) <= gr::WMAX);
  TORCH_CHECK(sidx.scalar_type() == at::kInt && nacc.scalar_type() == at::kInt && cu.scalar_type() == at::kInt);
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16 && q.scalar_type() == at::kBFloat16 && o.scalar_type() == at::kBFloat16);
  TORCH_CHECK(a.size(-1) == v.size(-2) && b.size(-1) == v.size(-2));
  const int N = cu.numel() - 1, HV = v.size(-2);
  gr::P p = make_p(A_log, dt_bias, state, q.size(-2), HV);
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()); p.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  p.sq = q.stride(1); p.sk = k.stride(1); p.sv = v.stride(1);
  p.a = reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()); p.b = reinterpret_cast<const __nv_bfloat16*>(b.data_ptr());
  p.sa = a.stride(0); p.sb = b.stride(0);
  p.sidx = sidx.data_ptr<int>(); p.sidx_w = sidx.size(1); p.nacc = nacc.data_ptr<int>(); p.cu = cu.data_ptr<int>();
  p.o = reinterpret_cast<__nv_bfloat16*>(o.data_ptr());
  p.scale = (float)scale;
  if (mode.numel()) {
    TORCH_CHECK(mode.numel() >= N && log_id.numel() >= N && mode.scalar_type() == at::kInt && log_id.scalar_type() == at::kInt);
    p.mode = mode.data_ptr<int>(); p.log_id = log_id.data_ptr<int>();
    set_log(p, logbuf, loghdr);
  }
  if (N == 0) return;
  dim3 grid(gr::DH / gr::RB, HV, N);
  gr::rec_kernel<<<grid, gr::THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(p);
}

// ids [K] int32 log ids to materialize, reg_parity [K] int32 (region the lazy step wrote).
void gdn_materialize(torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor state, torch::Tensor logbuf, torch::Tensor loghdr,
                     torch::Tensor ids, torch::Tensor reg_parity, int64_t HK) {
  const int K = ids.numel();
  if (K == 0) return;
  const int HV = state.size(1);
  gr::P p = make_p(A_log, dt_bias, state, (int)HK, HV);
  set_log(p, logbuf, loghdr);
  dim3 grid(gr::DH / gr::RB, HV, K);
  gr::mat_kernel<<<grid, gr::THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(p, ids.data_ptr<int>(), reg_parity.data_ptr<int>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_rec", &gdn_rec);
  m.def("gdn_materialize", &gdn_materialize);
}
