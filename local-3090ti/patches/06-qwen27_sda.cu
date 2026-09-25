// qwen27-dev: split-KV speculative-verify attention on the int8_per_token_head cache, sm86 mma.sync.
// Layout contract (same as the Triton _spec_attn_partial it replaces):
//   q [T, Hq, D] bf16; cache views [num_blocks, BS, Hkv, D] int8 whose head rows are D + 4 bytes
//   (one float32 scale inline after the data); block_table [reqs, max_blocks]; seqused_k [reqs]
//   (kv length incl. the new tokens); cu_q [reqs + 1]. Query token i of a request sits at kv
//   position seqused - q_len + i. Partials: ((req*Hq + head)*qmax + i)*nseg + seg.
// One CTA = (request, query-row tile, kv head, KV segment), 4 warps. Per 32-key tile:
//   QK^T: warp w takes keys [8w, 8w+8) for all rows (int8 K -> bf16 in registers);
//   online softmax over the tile through shared memory (2 threads per row);
//   PV: warp w takes head dims [w*D/4, (w+1)*D/4), so its fp32 accumulator stays small.
// KV tiles arrive by 16-byte cp.async, double-buffered: every token row is 16-byte aligned, a head
// row is only 4-byte aligned, so each key copies an aligned (D/16+1)*16-byte window and reads
// from offset `shift` (0/4/8/12) inside it.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define TK 32

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, int n) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
}
__device__ __forceinline__ void mma_bf16(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
  __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<unsigned*>(&r);
}
__device__ __forceinline__ unsigned i8x2_bf16x2(unsigned short v) {
  return pack_bf16((float)(signed char)(v & 0xff), (float)(signed char)(v >> 8));
}


__device__ __forceinline__ void mma_f16(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// 4 int8 (little-endian in w) -> two fp16x2 {x0,x1}, {x2,x3}, exactly: 0x64XX is 1024 + XX in fp16, so
// (x + 128) placed in the low byte gives 1152 + x; one half2 subtract removes the offset.
__device__ __forceinline__ void i8x4_f16x4(unsigned w, unsigned& lo, unsigned& hi) {
  const unsigned u = w ^ 0x80808080u;
  lo = __byte_perm(u, 0x64646464u, 0x5140);
  hi = __byte_perm(u, 0x64646464u, 0x7362);
  const __half2 bias = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));  // 1152
  __half2 l = __hsub2(*reinterpret_cast<__half2*>(&lo), bias), h = __hsub2(*reinterpret_cast<__half2*>(&hi), bias);
  lo = *reinterpret_cast<unsigned*>(&l); hi = *reinterpret_cast<unsigned*>(&h);
}
__device__ __forceinline__ unsigned pack_f16(float lo, float hi) {
  __half2 r = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<unsigned*>(&r);
}

__device__ __forceinline__ void mma_s8(int* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pack_s8x4(int a, int b, int c, int d) {
  return (unsigned)(a & 0xff) | ((unsigned)(b & 0xff) << 8) | ((unsigned)(c & 0xff) << 16) | ((unsigned)(d & 0xff) << 24);
}

struct Params {
  const __nv_bfloat16* q; long stride_qt, stride_qh;
  const unsigned char* k; const unsigned char* v; long stride_kb, stride_ks, stride_kh, stride_vb, stride_vs, stride_vh;
  const int* bt; long stride_bt; int BS;
  const int* seqused; const int* cu_q;
  float* part_o; float* part_m; float* part_l;
  float scale; int Hq, G, qmax, nseg, ntile, QT, window, causal, min_tiles;
};

template <int D, int MT>
__global__ void __launch_bounds__(128) sda_partial_kernel(const Params p) {
  constexpr int ROWS = MT * 16;
  constexpr int QSTR = D + 8;            // bf16 elements per Q row in smem
  constexpr int KVCH = D / 16 + 1;       // 16-byte chunks per key window
  constexpr int KVSTR = KVCH * 16;       // bytes per key row in smem
  constexpr int SSTR = TK + 1;           // floats per S row
  constexpr int PSTR = TK + 8;           // bf16 per P row
  constexpr int DW = D / 4, NT = DW / 8; // PV dims per warp, n8 tiles per warp
  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem);
  unsigned char* KVs = smem + ROWS * QSTR * 2;
  float* Ss = reinterpret_cast<float*>(KVs + 4 * TK * KVSTR);
  __nv_bfloat16* Ps = reinterpret_cast<__nv_bfloat16*>(Ss + ROWS * SSTR);
  float* alpha_s = reinterpret_cast<float*>(Ps + ROWS * PSTR);

  const int req = blockIdx.x / p.ntile, qtile = blockIdx.x % p.ntile;
  const int kvh = blockIdx.y, seg = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
  const int q_start = p.cu_q[req], q_len = p.cu_q[req + 1] - q_start;
  const int kv_len = p.seqused[req];
  const int G = p.G, QT = p.QT;
  if (qtile * QT >= q_len) return;

  // Q tile -> smem (rows r = i*G + g for query token qtile*QT + i, head kvh*G + g); zero invalid rows.
  for (int idx = tid; idx < ROWS * (D / 8); idx += 128) {
    int r = idx / (D / 8), c = (idx % (D / 8)) * 8;
    int qi = qtile * QT + r / G;
    uint4 val = make_uint4(0, 0, 0, 0);
    if (r < QT * G && qi < q_len)
      val = *reinterpret_cast<const uint4*>(p.q + (long)(q_start + qi) * p.stride_qt + (long)(kvh * G + r % G) * p.stride_qh + c);
    *reinterpret_cast<uint4*>(Qs + r * QSTR + c) = val;
  }

  // this segment's tiles (sliding window: start at the first tile any row can see)
  const int tiles_total = (kv_len + TK - 1) / TK;
  int t_lo = 0;
  if (p.window > 0) { int lo = kv_len - q_len - p.window + 1; t_lo = lo > 0 ? lo / TK : 0; }
  // Use only as many segments as the context needs (>= p.min_tiles tiles each): every live segment
  // costs a D-float partial per row, written here and re-read by the combine. Idle CTAs exit.
  const int nseg_eff = min(p.nseg, max(1, (tiles_total - t_lo + p.min_tiles - 1) / p.min_tiles));
  if (seg >= nseg_eff) return;
  const int per = (tiles_total - t_lo + nseg_eff - 1) / nseg_eff;
  const int t0 = t_lo + seg * per, t1 = min(t0 + per, tiles_total);

  // Head rows are 4-byte aligned (K and V interleave per head: [.., Hkv, 2, D + 4]); token and block
  // strides are multiples of 16, so each head's offset inside its 16-byte window is fixed per CTA.
  const unsigned char* khead = p.k + (long)kvh * p.stride_kh;
  const unsigned char* vhead = p.v + (long)kvh * p.stride_vh;
  const int kshift = (int)((uintptr_t)khead & 15), vshift = (int)((uintptr_t)vhead & 15);
  khead -= kshift; vhead -= vshift;
  const int kbytes = kshift + D + 4, vbytes = vshift + D + 4;   // exact bytes per window (no overread)
  auto load_tile = [&](int t, int st) {
    const int pos0 = t * TK;
    const long blk = p.bt[req * p.stride_bt + pos0 / p.BS];
    const int slot0 = pos0 % p.BS;
    const unsigned char* kb = khead + blk * p.stride_kb + (long)slot0 * p.stride_ks;
    const unsigned char* vb = vhead + blk * p.stride_vb + (long)slot0 * p.stride_vs;
    unsigned char* ks = KVs + (st * 2) * TK * KVSTR;
    unsigned char* vs = KVs + (st * 2 + 1) * TK * KVSTR;
    for (int i = tid; i < TK * KVCH; i += 128) {
      int key = i / KVCH, ch = i % KVCH;
      const bool ok = pos0 + key < kv_len;
      const int kn = ok ? min(16, max(0, kbytes - ch * 16)) : 0;
      const int vn = ok ? min(16, max(0, vbytes - ch * 16)) : 0;
      cp_async16(ks + key * KVSTR + ch * 16, kb + (long)key * p.stride_ks + ch * 16, kn);
      cp_async16(vs + key * KVSTR + ch * 16, vb + (long)key * p.stride_vs + ch * 16, vn);
    }
  };

  float acc[MT][NT][4];
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int b = 0; b < NT; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;
  float m_r = -INFINITY, l_r = 0.f;          // softmax state of row tid >> 1 (2 threads per row)
  const int sr = tid >> 1, shalf = (tid & 1) * (TK / 2);

  if (t0 < t1) load_tile(t0, 0);
  cp_async_commit();
  for (int t = t0; t < t1; ++t) {
    const int st = (t - t0) & 1;
    // Tile t is the only copy in flight: wait for it, then barrier -- which also guarantees every
    // warp is done with tile t-1 (its V in stage st^1) -- and only then start filling stage st^1.
    cp_async_wait<0>();
    __syncthreads();
    if (t + 1 < t1) load_tile(t + 1, st ^ 1);
    cp_async_commit();
    const unsigned char* ks = KVs + (st * 2) * TK * KVSTR + kshift;
    const unsigned char* vs = KVs + (st * 2 + 1) * TK * KVSTR + vshift;

    // ---- S = Q K^T for keys [8*warp, 8*warp + 8)
    float sacc[MT][4];
#pragma unroll
    for (int a = 0; a < MT; ++a) sacc[a][0] = sacc[a][1] = sacc[a][2] = sacc[a][3] = 0.f;
    const unsigned char* krow = ks + (warp * 8 + gid) * KVSTR;
#pragma unroll 4
    for (int kk = 0; kk < D / 16; ++kk) {
      unsigned b0 = i8x2_bf16x2(*reinterpret_cast<const unsigned short*>(krow + kk * 16 + tig * 2));
      unsigned b1 = i8x2_bf16x2(*reinterpret_cast<const unsigned short*>(krow + kk * 16 + 8 + tig * 2));
#pragma unroll
      for (int a = 0; a < MT; ++a) {
        unsigned a0, a1, a2, a3;
        ldmatrix_x4(a0, a1, a2, a3, Qs + (a * 16 + (lane & 15)) * QSTR + kk * 16 + (lane >> 4) * 8);
        mma_bf16(sacc[a], a0, a1, a2, a3, b0, b1);
      }
    }
    {
      const int key0 = warp * 8 + tig * 2;
      const float ksc0 = *reinterpret_cast<const float*>(ks + key0 * KVSTR + D) * p.scale;
      const float ksc1 = *reinterpret_cast<const float*>(ks + (key0 + 1) * KVSTR + D) * p.scale;
      const int pos0 = t * TK + key0;
#pragma unroll
      for (int a = 0; a < MT; ++a)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = a * 16 + gid + h * 8;
          const int qi = qtile * QT + r / G;
          const bool rv = (r < QT * G) && (qi < q_len);
          const int qpos = kv_len - q_len + qi;
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const int pos = pos0 + j;
            bool ok = rv && pos < kv_len;
            if (p.causal) ok = ok && pos <= qpos;
            if (p.window > 0) ok = ok && (qpos - pos < p.window) && (p.causal || pos - qpos < p.window);
            Ss[r * SSTR + key0 + j] = ok ? sacc[a][h * 2 + j] * (j ? ksc1 : ksc0) : -INFINITY;
          }
        }
    }
    __syncthreads();

    // ---- online softmax over this tile; P' = P * v_scale -> smem (bf16)
    if (sr < ROWS) {
      float mx = -INFINITY;
#pragma unroll
      for (int k = 0; k < TK / 2; ++k) mx = fmaxf(mx, Ss[sr * SSTR + shalf + k]);
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
      const float m_new = fmaxf(m_r, mx);
      const float m_safe = (m_new == -INFINITY) ? 0.f : m_new;
      const float alpha = (m_r == -INFINITY) ? 0.f : __expf(m_r - m_safe);
      float sum = 0.f;
#pragma unroll
      for (int k = 0; k < TK / 2; ++k) {
        const float pv = __expf(Ss[sr * SSTR + shalf + k] - m_safe);
        sum += pv;
        const float vsc = *reinterpret_cast<const float*>(vs + (shalf + k) * KVSTR + D);
        Ps[sr * PSTR + shalf + k] = __float2bfloat16(pv * vsc);
      }
      sum += __shfl_xor_sync(0xffffffff, sum, 1);
      l_r = l_r * alpha + sum;
      m_r = m_new;
      if ((tid & 1) == 0) alpha_s[sr] = alpha;
    }
    __syncthreads();

    // ---- O[:, dims of this warp] = alpha * O + P' V
#pragma unroll
    for (int a = 0; a < MT; ++a) {
      const float al = alpha_s[a * 16 + gid], ah = alpha_s[a * 16 + gid + 8];
#pragma unroll
      for (int n = 0; n < NT; ++n) { acc[a][n][0] *= al; acc[a][n][1] *= al; acc[a][n][2] *= ah; acc[a][n][3] *= ah; }
    }
    const unsigned char* vcol = vs + warp * DW + gid;
#pragma unroll
    for (int kk = 0; kk < TK / 16; ++kk) {
      unsigned af[MT][4];
#pragma unroll
      for (int a = 0; a < MT; ++a)
        ldmatrix_x4(af[a][0], af[a][1], af[a][2], af[a][3], Ps + (a * 16 + (lane & 15)) * PSTR + kk * 16 + (lane >> 4) * 8);
      const int k0 = kk * 16 + tig * 2;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const unsigned char* vc = vcol + n * 8;
        const unsigned b0 = pack_bf16((float)(signed char)vc[k0 * KVSTR], (float)(signed char)vc[(k0 + 1) * KVSTR]);
        const unsigned b1 = pack_bf16((float)(signed char)vc[(k0 + 8) * KVSTR], (float)(signed char)vc[(k0 + 9) * KVSTR]);
#pragma unroll
        for (int a = 0; a < MT; ++a) mma_bf16(acc[a][n], af[a][0], af[a][1], af[a][2], af[a][3], b0, b1);
      }
    }
  }
  cp_async_wait<0>();

  // ---- partials (unnormalised o, running max m, running sum l), same layout as the Triton kernel
  if (sr < ROWS && (tid & 1) == 0) {
    const int qi = qtile * QT + sr / G;
    if (sr < QT * G && qi < q_len) {
      const long pidx = ((long)(req * p.Hq + kvh * G + sr % G) * p.qmax + qi) * p.nseg + seg;
      p.part_m[pidx] = m_r;
      p.part_l[pidx] = l_r;
    }
  }
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = a * 16 + gid + h * 8;
      const int qi = qtile * QT + r / G;
      if (r < QT * G && qi < q_len) {
        const long pidx = ((long)(req * p.Hq + kvh * G + r % G) * p.qmax + qi) * p.nseg + seg;
        float* o = p.part_o + pidx * D + warp * DW + tig * 2;
#pragma unroll
        for (int n = 0; n < NT; ++n) *reinterpret_cast<float2*>(o + n * 8) = make_float2(acc[a][n][h * 2], acc[a][n][h * 2 + 1]);
      }
    }
}

template <int D>
__global__ void sda_combine_kernel(const float* part_o, const float* part_m, const float* part_l, __nv_bfloat16* out,
                                   const int* cu_q, long stride_ot, long stride_oh, int Hq, int qmax, int nseg,
                                   const int* seqused, int window, int min_tiles) {
  const int req = blockIdx.x, h = blockIdx.y, i = blockIdx.z, d = threadIdx.x;
  const int q_start = cu_q[req], q_len = cu_q[req + 1] - q_start;
  if (i >= q_len) return;
  const int nseg_all = nseg;
  {
    const int kv_len = seqused[req], tiles_total = (kv_len + TK - 1) / TK;
    int t_lo = 0;
    if (window > 0) { int lo = kv_len - q_len - window + 1; t_lo = lo > 0 ? lo / TK : 0; }
    nseg = min(nseg_all, max(1, (tiles_total - t_lo + min_tiles - 1) / min_tiles));
  }
  const long base = ((long)(req * Hq + h) * qmax + i) * nseg_all;
  float mmax = -INFINITY;
  for (int s = 0; s < nseg; ++s) mmax = fmaxf(mmax, part_m[base + s]);
  if (mmax == -INFINITY) mmax = 0.f;
  float lt = 0.f, o = 0.f;
  for (int s = 0; s < nseg; ++s) {
    const float w = __expf(part_m[base + s] - mmax);
    lt += part_l[base + s] * w;
    o += part_o[(base + s) * D + d] * w;
  }
  out[(long)(q_start + i) * stride_ot + (long)h * stride_oh + d] = __float2bfloat16(o / fmaxf(lt, 1e-30f));
}

template <int D, int MT>
static void launch_partial(const Params& p, int num_reqs, int hkv, cudaStream_t stream) {
  constexpr int ROWS = MT * 16;
  const int smem = ROWS * (D + 8) * 2 + 4 * TK * (D / 16 + 1) * 16 + ROWS * (TK + 1) * 4 + ROWS * (TK + 8) * 2 + ROWS * 4;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(sda_partial_kernel<D, MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
  dim3 grid(num_reqs * p.ntile, hkv, p.nseg);
  sda_partial_kernel<D, MT><<<grid, 128, smem, stream>>>(p);
}


// ---------------------------------------------------------------------------------------------
// v2: Q lives in registers (each warp holds its D/4 slice of every row as mma A fragments, loaded
// once); Q.K^T is split along D across the 4 warps and the partial scores are summed with
// shared-memory float atomics. ~45 KB of shared memory at D=256, so two CTAs fit on an SM.
template <int D, int MT>
__global__ void __launch_bounds__(128, 2) sda_partial_v2_kernel(const Params p) {
  constexpr int ROWS = MT * 16;
  constexpr int KVCH = D / 16 + 1, KVSTR = KVCH * 16;
  constexpr int SSTR = TK + 1, PSTR = TK + 8;
  constexpr int DW = D / 4, NT = DW / 8, KS = DW / 16;   // per-warp head-dim slice, its n8 tiles and k16 steps
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* KVs = smem;                                             // 2 stages x (K, V)
  float* Ss = reinterpret_cast<float*>(KVs + 4 * TK * KVSTR);            // 2 buffers (ping-pong)
  __nv_bfloat16* Ps = reinterpret_cast<__nv_bfloat16*>(Ss + 2 * ROWS * SSTR);
  float* alpha_s = reinterpret_cast<float*>(Ps + ROWS * PSTR);

  const int req = blockIdx.x / p.ntile, qtile = blockIdx.x % p.ntile;
  const int kvh = blockIdx.y, seg = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
  const int q_start = p.cu_q[req], q_len = p.cu_q[req + 1] - q_start;
  const int kv_len = p.seqused[req];
  const int G = p.G, QT = p.QT;
  if (qtile * QT >= q_len) return;

  // Q A-fragments for dims [warp*DW, warp*DW + DW): a0 (row gid, k 2tig..), a1 (row gid+8), a2 (row gid, k+8), a3 (row gid+8, k+8)
  unsigned qa[MT][KS][4];
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = a * 16 + gid + h * 8;
      const int qi = qtile * QT + r / G;
      const bool rv = (r < QT * G) && (qi < q_len);
      const __nv_bfloat16* qrow = p.q + (long)(q_start + qi) * p.stride_qt + (long)(kvh * G + r % G) * p.stride_qh + warp * DW;
#pragma unroll
      for (int kk = 0; kk < KS; ++kk) {
        qa[a][kk][h] = rv ? *reinterpret_cast<const unsigned*>(qrow + kk * 16 + tig * 2) : 0u;
        qa[a][kk][2 + h] = rv ? *reinterpret_cast<const unsigned*>(qrow + kk * 16 + 8 + tig * 2) : 0u;
      }
    }
  for (int i = tid; i < 2 * ROWS * SSTR; i += 128) Ss[i] = 0.f;

  const int tiles_total = (kv_len + TK - 1) / TK;
  int t_lo = 0;
  if (p.window > 0) { int lo = kv_len - q_len - p.window + 1; t_lo = lo > 0 ? lo / TK : 0; }
  // Use only as many segments as the context needs (>= p.min_tiles tiles each): every live segment
  // costs a D-float partial per row, written here and re-read by the combine. Idle CTAs exit.
  const int nseg_eff = min(p.nseg, max(1, (tiles_total - t_lo + p.min_tiles - 1) / p.min_tiles));
  if (seg >= nseg_eff) return;
  const int per = (tiles_total - t_lo + nseg_eff - 1) / nseg_eff;
  const int t0 = t_lo + seg * per, t1 = min(t0 + per, tiles_total);

  const unsigned char* khead = p.k + (long)kvh * p.stride_kh;
  const unsigned char* vhead = p.v + (long)kvh * p.stride_vh;
  const int kshift = (int)((uintptr_t)khead & 15), vshift = (int)((uintptr_t)vhead & 15);
  khead -= kshift; vhead -= vshift;
  const int kbytes = kshift + D + 4, vbytes = vshift + D + 4;
  auto load_tile = [&](int t, int st) {
    const int pos0 = t * TK;
    const long blk = p.bt[req * p.stride_bt + pos0 / p.BS];
    const int slot0 = pos0 % p.BS;
    const unsigned char* kb = khead + blk * p.stride_kb + (long)slot0 * p.stride_ks;
    const unsigned char* vb = vhead + blk * p.stride_vb + (long)slot0 * p.stride_vs;
    unsigned char* ks = KVs + (st * 2) * TK * KVSTR;
    unsigned char* vs = KVs + (st * 2 + 1) * TK * KVSTR;
    for (int i = tid; i < TK * KVCH; i += 128) {
      const int key = i / KVCH, ch = i % KVCH;
      const bool ok = pos0 + key < kv_len;
      const int kn = ok ? min(16, max(0, kbytes - ch * 16)) : 0;
      const int vn = ok ? min(16, max(0, vbytes - ch * 16)) : 0;
      cp_async16(ks + key * KVSTR + ch * 16, kb + (long)key * p.stride_ks + ch * 16, kn);
      cp_async16(vs + key * KVSTR + ch * 16, vb + (long)key * p.stride_vs + ch * 16, vn);
    }
  };

  float acc[MT][NT][4];
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int b = 0; b < NT; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;
  float m_r = -INFINITY, l_r = 0.f;
  const int sr = tid >> 1, shalf = (tid & 1) * (TK / 2);

  if (t0 < t1) load_tile(t0, 0);
  cp_async_commit();
  for (int t = t0; t < t1; ++t) {
    const int st = (t - t0) & 1;
    // Tile t is the only copy in flight: wait for it, then barrier -- which also guarantees every
    // warp is done with tile t-1 (its V in stage st^1) -- and only then start filling stage st^1.
    cp_async_wait<0>();
    __syncthreads();
    if (t + 1 < t1) load_tile(t + 1, st ^ 1);
    cp_async_commit();
    const unsigned char* ks = KVs + (st * 2) * TK * KVSTR + kshift;
    const unsigned char* vs = KVs + (st * 2 + 1) * TK * KVSTR + vshift;
    float* S = Ss + st * ROWS * SSTR;

    // ---- partial S over this warp's head-dim slice, all TK keys; atomically summed into S
#pragma unroll
    for (int n = 0; n < TK / 8; ++n) {
      float sacc[MT][4];
#pragma unroll
      for (int a = 0; a < MT; ++a) sacc[a][0] = sacc[a][1] = sacc[a][2] = sacc[a][3] = 0.f;
      const unsigned char* krow = ks + (n * 8 + gid) * KVSTR + warp * DW;
#pragma unroll
      for (int kk = 0; kk < KS; ++kk) {
        const unsigned b0 = i8x2_bf16x2(*reinterpret_cast<const unsigned short*>(krow + kk * 16 + tig * 2));
        const unsigned b1 = i8x2_bf16x2(*reinterpret_cast<const unsigned short*>(krow + kk * 16 + 8 + tig * 2));
#pragma unroll
        for (int a = 0; a < MT; ++a) mma_bf16(sacc[a], qa[a][kk][0], qa[a][kk][1], qa[a][kk][2], qa[a][kk][3], b0, b1);
      }
#pragma unroll
      for (int a = 0; a < MT; ++a)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int j = 0; j < 2; ++j)
            atomicAdd(&S[(a * 16 + gid + h * 8) * SSTR + n * 8 + tig * 2 + j], sacc[a][h * 2 + j]);
    }
    __syncthreads();

    // ---- scale, mask, online softmax (2 threads per row); zero the other S buffer for tile t+1
    if (sr < ROWS) {
      const int qi = qtile * QT + sr / G;
      const bool rv = (sr < QT * G) && (qi < q_len);
      const int qpos = kv_len - q_len + qi;
      float sv[TK / 2];
      float mx = -INFINITY;
#pragma unroll
      for (int k = 0; k < TK / 2; ++k) {
        const int key = shalf + k, pos = t * TK + key;
        bool ok = rv && pos < kv_len;
        if (p.causal) ok = ok && pos <= qpos;
        if (p.window > 0) ok = ok && (qpos - pos < p.window) && (p.causal || pos - qpos < p.window);
        sv[k] = ok ? S[sr * SSTR + key] * (*reinterpret_cast<const float*>(ks + key * KVSTR + D) * p.scale) : -INFINITY;
        mx = fmaxf(mx, sv[k]);
      }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
      const float m_new = fmaxf(m_r, mx);
      const float m_safe = (m_new == -INFINITY) ? 0.f : m_new;
      const float alpha = (m_r == -INFINITY) ? 0.f : __expf(m_r - m_safe);
      float sum = 0.f;
#pragma unroll
      for (int k = 0; k < TK / 2; ++k) {
        const float pv = __expf(sv[k] - m_safe);
        sum += pv;
        Ps[sr * PSTR + shalf + k] = __float2bfloat16(pv * *reinterpret_cast<const float*>(vs + (shalf + k) * KVSTR + D));
      }
      sum += __shfl_xor_sync(0xffffffff, sum, 1);
      l_r = l_r * alpha + sum;
      m_r = m_new;
      if ((tid & 1) == 0) alpha_s[sr] = alpha;
    }
    {
      float* So = Ss + (st ^ 1) * ROWS * SSTR;
      for (int i = tid; i < ROWS * SSTR; i += 128) So[i] = 0.f;
    }
    __syncthreads();

    // ---- O[:, this warp's dims] = alpha * O + P' V
#pragma unroll
    for (int a = 0; a < MT; ++a) {
      const float al = alpha_s[a * 16 + gid], ah = alpha_s[a * 16 + gid + 8];
#pragma unroll
      for (int n = 0; n < NT; ++n) { acc[a][n][0] *= al; acc[a][n][1] *= al; acc[a][n][2] *= ah; acc[a][n][3] *= ah; }
    }
    const unsigned char* vcol = vs + warp * DW + gid;
#pragma unroll
    for (int kk = 0; kk < TK / 16; ++kk) {
      unsigned af[MT][4];
#pragma unroll
      for (int a = 0; a < MT; ++a)
        ldmatrix_x4(af[a][0], af[a][1], af[a][2], af[a][3], Ps + (a * 16 + (lane & 15)) * PSTR + kk * 16 + (lane >> 4) * 8);
      const int k0 = kk * 16 + tig * 2;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const unsigned char* vc = vcol + n * 8;
        const unsigned b0 = pack_bf16((float)(signed char)vc[k0 * KVSTR], (float)(signed char)vc[(k0 + 1) * KVSTR]);
        const unsigned b1 = pack_bf16((float)(signed char)vc[(k0 + 8) * KVSTR], (float)(signed char)vc[(k0 + 9) * KVSTR]);
#pragma unroll
        for (int a = 0; a < MT; ++a) mma_bf16(acc[a][n], af[a][0], af[a][1], af[a][2], af[a][3], b0, b1);
      }
    }
  }
  cp_async_wait<0>();

  if (sr < ROWS && (tid & 1) == 0) {
    const int qi = qtile * QT + sr / G;
    if (sr < QT * G && qi < q_len) {
      const long pidx = ((long)(req * p.Hq + kvh * G + sr % G) * p.qmax + qi) * p.nseg + seg;
      p.part_m[pidx] = m_r;
      p.part_l[pidx] = l_r;
    }
  }
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = a * 16 + gid + h * 8;
      const int qi = qtile * QT + r / G;
      if (r < QT * G && qi < q_len) {
        const long pidx = ((long)(req * p.Hq + kvh * G + r % G) * p.qmax + qi) * p.nseg + seg;
        float* o = p.part_o + pidx * D + warp * DW + tig * 2;
#pragma unroll
        for (int n = 0; n < NT; ++n) *reinterpret_cast<float2*>(o + n * 8) = make_float2(acc[a][n][h * 2], acc[a][n][h * 2 + 1]);
      }
    }
}

template <int D, int MT>
static void launch_partial_v2(const Params& p, int num_reqs, int hkv, cudaStream_t stream) {
  constexpr int ROWS = MT * 16;
  const int smem = 4 * TK * (D / 16 + 1) * 16 + 2 * ROWS * (TK + 1) * 4 + ROWS * (TK + 8) * 2 + ROWS * 4;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(sda_partial_v2_kernel<D, MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
  dim3 grid(num_reqs * p.ntile, hkv, p.nseg);
  sda_partial_v2_kernel<D, MT><<<grid, 128, smem, stream>>>(p);
}

// v3: // v2: Q lives in registers (each warp holds its D/4 slice of every row as mma A fragments, loaded
// once); Q.K^T is split along D across the 4 warps and the partial scores are summed with
// shared-memory float atomics. ~45 KB of shared memory at D=256, so two CTAs fit on an SM.
template <int D, int MT>
__global__ void __launch_bounds__(128, 2) sda_partial_v3_kernel(const Params p) {
  constexpr int ROWS = MT * 16;
  constexpr int KVCH = D / 16 + 1, KVSTR = KVCH * 16;
  constexpr int SSTR = TK + 1, PSTR = TK + 8;
  constexpr int DW = D / 4, NT = DW / 8, KS = D / 16;    // PV: per-warp head-dim slice; QK: full-D k16 steps
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* KVs = smem;                                             // 2 stages x (K, V)
  float* Ss = reinterpret_cast<float*>(KVs + 4 * TK * KVSTR);
  __nv_bfloat16* Ps = reinterpret_cast<__nv_bfloat16*>(Ss + ROWS * SSTR);
  float* alpha_s = reinterpret_cast<float*>(Ps + ROWS * PSTR);

  const int req = blockIdx.x / p.ntile, qtile = blockIdx.x % p.ntile;
  const int kvh = blockIdx.y, seg = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
  const int q_start = p.cu_q[req], q_len = p.cu_q[req + 1] - q_start;
  const int kv_len = p.seqused[req];
  const int G = p.G, QT = p.QT;
  if (qtile * QT >= q_len) return;

  // Q A-fragments: warp w owns row tile w (rows w*16 .. w*16+15) over the full head dim.
  unsigned qa[KS][4];
  const int mw = warp < MT ? warp : 0;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = mw * 16 + gid + h * 8;
    const int qi = qtile * QT + r / G;
    const bool rv = warp < MT && (r < QT * G) && (qi < q_len);
    const __nv_bfloat16* qrow = p.q + (long)(q_start + qi) * p.stride_qt + (long)(kvh * G + r % G) * p.stride_qh;
#pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
      qa[kk][h] = rv ? *reinterpret_cast<const unsigned*>(qrow + kk * 16 + tig * 2) : 0u;
      qa[kk][2 + h] = rv ? *reinterpret_cast<const unsigned*>(qrow + kk * 16 + 8 + tig * 2) : 0u;
    }
  }

  const int tiles_total = (kv_len + TK - 1) / TK;
  int t_lo = 0;
  if (p.window > 0) { int lo = kv_len - q_len - p.window + 1; t_lo = lo > 0 ? lo / TK : 0; }
  // Use only as many segments as the context needs (>= p.min_tiles tiles each): every live segment
  // costs a D-float partial per row, written here and re-read by the combine. Idle CTAs exit.
  const int nseg_eff = min(p.nseg, max(1, (tiles_total - t_lo + p.min_tiles - 1) / p.min_tiles));
  if (seg >= nseg_eff) return;
  const int per = (tiles_total - t_lo + nseg_eff - 1) / nseg_eff;
  const int t0 = t_lo + seg * per, t1 = min(t0 + per, tiles_total);

  const unsigned char* khead = p.k + (long)kvh * p.stride_kh;
  const unsigned char* vhead = p.v + (long)kvh * p.stride_vh;
  const int kshift = (int)((uintptr_t)khead & 15), vshift = (int)((uintptr_t)vhead & 15);
  khead -= kshift; vhead -= vshift;
  const int kbytes = kshift + D + 4, vbytes = vshift + D + 4;
  auto load_tile = [&](int t, int st) {
    const int pos0 = t * TK;
    const long blk = p.bt[req * p.stride_bt + pos0 / p.BS];
    const int slot0 = pos0 % p.BS;
    const unsigned char* kb = khead + blk * p.stride_kb + (long)slot0 * p.stride_ks;
    const unsigned char* vb = vhead + blk * p.stride_vb + (long)slot0 * p.stride_vs;
    unsigned char* ks = KVs + (st * 2) * TK * KVSTR;
    unsigned char* vs = KVs + (st * 2 + 1) * TK * KVSTR;
    for (int i = tid; i < TK * KVCH; i += 128) {
      const int key = i / KVCH, ch = i % KVCH;
      const bool ok = pos0 + key < kv_len;
      const int kn = ok ? min(16, max(0, kbytes - ch * 16)) : 0;
      const int vn = ok ? min(16, max(0, vbytes - ch * 16)) : 0;
      cp_async16(ks + key * KVSTR + ch * 16, kb + (long)key * p.stride_ks + ch * 16, kn);
      cp_async16(vs + key * KVSTR + ch * 16, vb + (long)key * p.stride_vs + ch * 16, vn);
    }
  };

  float acc[MT][NT][4];
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int b = 0; b < NT; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;
  float m_r = -INFINITY, l_r = 0.f;
  const int sr = tid >> 1, shalf = (tid & 1) * (TK / 2);

  if (t0 < t1) load_tile(t0, 0);
  cp_async_commit();
  for (int t = t0; t < t1; ++t) {
    const int st = (t - t0) & 1;
    // Tile t is the only copy in flight: wait for it, then barrier -- which also guarantees every
    // warp is done with tile t-1 (its V in stage st^1) -- and only then start filling stage st^1.
    cp_async_wait<0>();
    __syncthreads();
    if (t + 1 < t1) load_tile(t + 1, st ^ 1);
    cp_async_commit();
    const unsigned char* ks = KVs + (st * 2) * TK * KVSTR + kshift;
    const unsigned char* vs = KVs + (st * 2 + 1) * TK * KVSTR + vshift;
    float* S = Ss;

    // ---- S rows of this warp's row tile, all TK keys, full head dim
    if (warp < MT) {
#pragma unroll
      for (int n = 0; n < TK / 8; ++n) {
        float sacc[4] = {0.f, 0.f, 0.f, 0.f};
        const unsigned char* krow = ks + (n * 8 + gid) * KVSTR;
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
          const unsigned b0 = i8x2_bf16x2(*reinterpret_cast<const unsigned short*>(krow + kk * 16 + tig * 2));
          const unsigned b1 = i8x2_bf16x2(*reinterpret_cast<const unsigned short*>(krow + kk * 16 + 8 + tig * 2));
          mma_bf16(sacc, qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b0, b1);
        }
        float* srow = S + (warp * 16 + gid) * SSTR + n * 8 + tig * 2;
        srow[0] = sacc[0]; srow[1] = sacc[1]; srow[8 * SSTR] = sacc[2]; srow[8 * SSTR + 1] = sacc[3];
      }
    }
    __syncthreads();

    // ---- scale, mask, online softmax (2 threads per row); zero the other S buffer for tile t+1
    if (sr < ROWS) {
      const int qi = qtile * QT + sr / G;
      const bool rv = (sr < QT * G) && (qi < q_len);
      const int qpos = kv_len - q_len + qi;
      float sv[TK / 2];
      float mx = -INFINITY;
#pragma unroll
      for (int k = 0; k < TK / 2; ++k) {
        const int key = shalf + k, pos = t * TK + key;
        bool ok = rv && pos < kv_len;
        if (p.causal) ok = ok && pos <= qpos;
        if (p.window > 0) ok = ok && (qpos - pos < p.window) && (p.causal || pos - qpos < p.window);
        sv[k] = ok ? S[sr * SSTR + key] * (*reinterpret_cast<const float*>(ks + key * KVSTR + D) * p.scale) : -INFINITY;
        mx = fmaxf(mx, sv[k]);
      }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
      const float m_new = fmaxf(m_r, mx);
      const float m_safe = (m_new == -INFINITY) ? 0.f : m_new;
      const float alpha = (m_r == -INFINITY) ? 0.f : __expf(m_r - m_safe);
      float sum = 0.f;
#pragma unroll
      for (int k = 0; k < TK / 2; ++k) {
        const float pv = __expf(sv[k] - m_safe);
        sum += pv;
        Ps[sr * PSTR + shalf + k] = __float2bfloat16(pv * *reinterpret_cast<const float*>(vs + (shalf + k) * KVSTR + D));
      }
      sum += __shfl_xor_sync(0xffffffff, sum, 1);
      l_r = l_r * alpha + sum;
      m_r = m_new;
      if ((tid & 1) == 0) alpha_s[sr] = alpha;
    }
    __syncthreads();

    // ---- O[:, this warp's dims] = alpha * O + P' V
#pragma unroll
    for (int a = 0; a < MT; ++a) {
      const float al = alpha_s[a * 16 + gid], ah = alpha_s[a * 16 + gid + 8];
#pragma unroll
      for (int n = 0; n < NT; ++n) { acc[a][n][0] *= al; acc[a][n][1] *= al; acc[a][n][2] *= ah; acc[a][n][3] *= ah; }
    }
    const unsigned char* vcol = vs + warp * DW + gid;
#pragma unroll
    for (int kk = 0; kk < TK / 16; ++kk) {
      unsigned af[MT][4];
#pragma unroll
      for (int a = 0; a < MT; ++a)
        ldmatrix_x4(af[a][0], af[a][1], af[a][2], af[a][3], Ps + (a * 16 + (lane & 15)) * PSTR + kk * 16 + (lane >> 4) * 8);
      const int k0 = kk * 16 + tig * 2;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const unsigned char* vc = vcol + n * 8;
        const unsigned b0 = pack_bf16((float)(signed char)vc[k0 * KVSTR], (float)(signed char)vc[(k0 + 1) * KVSTR]);
        const unsigned b1 = pack_bf16((float)(signed char)vc[(k0 + 8) * KVSTR], (float)(signed char)vc[(k0 + 9) * KVSTR]);
#pragma unroll
        for (int a = 0; a < MT; ++a) mma_bf16(acc[a][n], af[a][0], af[a][1], af[a][2], af[a][3], b0, b1);
      }
    }
  }
  cp_async_wait<0>();

  if (sr < ROWS && (tid & 1) == 0) {
    const int qi = qtile * QT + sr / G;
    if (sr < QT * G && qi < q_len) {
      const long pidx = ((long)(req * p.Hq + kvh * G + sr % G) * p.qmax + qi) * p.nseg + seg;
      p.part_m[pidx] = m_r;
      p.part_l[pidx] = l_r;
    }
  }
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = a * 16 + gid + h * 8;
      const int qi = qtile * QT + r / G;
      if (r < QT * G && qi < q_len) {
        const long pidx = ((long)(req * p.Hq + kvh * G + r % G) * p.qmax + qi) * p.nseg + seg;
        float* o = p.part_o + pidx * D + warp * DW + tig * 2;
#pragma unroll
        for (int n = 0; n < NT; ++n) *reinterpret_cast<float2*>(o + n * 8) = make_float2(acc[a][n][h * 2], acc[a][n][h * 2 + 1]);
      }
    }
}


template <int D, int MT>
static void launch_partial_v3(const Params& p, int num_reqs, int hkv, cudaStream_t stream) {
  constexpr int ROWS = MT * 16;
  const int smem = 4 * TK * (D / 16 + 1) * 16 + ROWS * (TK + 1) * 4 + ROWS * (TK + 8) * 2 + ROWS * 4;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(sda_partial_v3_kernel<D, MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
  dim3 grid(num_reqs * p.ntile, hkv, p.nseg);
  sda_partial_v3_kernel<D, MT><<<grid, 128, smem, stream>>>(p);
}

// v4 (fp16 Q.K^T, permuted head dim, 32-bit K fragment loads): // v2: Q lives in registers (each warp holds its D/4 slice of every row as mma A fragments, loaded
// once); Q.K^T is split along D across the 4 warps and the partial scores are summed with
// shared-memory float atomics. ~45 KB of shared memory at D=256, so two CTAs fit on an SM.
template <int D, int MT>
__global__ void __launch_bounds__(128, 2) sda_partial_v4_kernel(const Params p) {
  constexpr int ROWS = MT * 16;
  constexpr int KVCH = D / 16 + 1, KVSTR = KVCH * 16;
  constexpr int SSTR = TK + 1, PSTR = TK + 8;
  constexpr int DW = D / 4, NT = DW / 8, KS = D / 16;    // PV: per-warp head-dim slice; QK: full-D k16 steps
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* KVs = smem;                                             // 2 stages x (K, V)
  float* Ss = reinterpret_cast<float*>(KVs + 4 * TK * KVSTR);
  __nv_bfloat16* Ps = reinterpret_cast<__nv_bfloat16*>(Ss + ROWS * SSTR);
  float* alpha_s = reinterpret_cast<float*>(Ps + ROWS * PSTR);

  const int req = blockIdx.x / p.ntile, qtile = blockIdx.x % p.ntile;
  const int kvh = blockIdx.y, seg = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
  const int q_start = p.cu_q[req], q_len = p.cu_q[req + 1] - q_start;
  const int kv_len = p.seqused[req];
  const int G = p.G, QT = p.QT;
  if (qtile * QT >= q_len) return;

  // Q A-fragments: warp w owns row tile w (rows w*16 .. w*16+15) over the full head dim.
  unsigned qa[KS][4];
  const int mw = warp < MT ? warp : 0;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = mw * 16 + gid + h * 8;
    const int qi = qtile * QT + r / G;
    const bool rv = warp < MT && (r < QT * G) && (qi < q_len);
    const __nv_bfloat16* qrow = p.q + (long)(q_start + qi) * p.stride_qt + (long)(kvh * G + r % G) * p.stride_qh;
#pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
      // Q.K^T sums over the head dim, so any permutation applied to both Q and K is exact: thread
      // tig's mma k-slots {2t, 2t+1 | 2t+8, 2t+9} take dims {4t, 4t+1 | 4t+2, 4t+3} of each 16-block,
      // which makes its K fragment one contiguous 4-byte word.
      if (rv) {
        const uint2 w = *reinterpret_cast<const uint2*>(qrow + kk * 16 + tig * 4);
        const __nv_bfloat162 q01 = *reinterpret_cast<const __nv_bfloat162*>(&w.x), q23 = *reinterpret_cast<const __nv_bfloat162*>(&w.y);
        qa[kk][h] = pack_f16(__bfloat162float(q01.x), __bfloat162float(q01.y));
        qa[kk][2 + h] = pack_f16(__bfloat162float(q23.x), __bfloat162float(q23.y));
      } else {
        qa[kk][h] = 0u; qa[kk][2 + h] = 0u;
      }
    }
  }

  const int tiles_total = (kv_len + TK - 1) / TK;
  int t_lo = 0;
  if (p.window > 0) { int lo = kv_len - q_len - p.window + 1; t_lo = lo > 0 ? lo / TK : 0; }
  // Use only as many segments as the context needs (>= p.min_tiles tiles each): every live segment
  // costs a D-float partial per row, written here and re-read by the combine. Idle CTAs exit.
  const int nseg_eff = min(p.nseg, max(1, (tiles_total - t_lo + p.min_tiles - 1) / p.min_tiles));
  if (seg >= nseg_eff) return;
  const int per = (tiles_total - t_lo + nseg_eff - 1) / nseg_eff;
  const int t0 = t_lo + seg * per, t1 = min(t0 + per, tiles_total);

  const unsigned char* khead = p.k + (long)kvh * p.stride_kh;
  const unsigned char* vhead = p.v + (long)kvh * p.stride_vh;
  const int kshift = (int)((uintptr_t)khead & 15), vshift = (int)((uintptr_t)vhead & 15);
  khead -= kshift; vhead -= vshift;
  const int kbytes = kshift + D + 4, vbytes = vshift + D + 4;
  auto load_tile = [&](int t, int st) {
    const int pos0 = t * TK;
    const long blk = p.bt[req * p.stride_bt + pos0 / p.BS];
    const int slot0 = pos0 % p.BS;
    const unsigned char* kb = khead + blk * p.stride_kb + (long)slot0 * p.stride_ks;
    const unsigned char* vb = vhead + blk * p.stride_vb + (long)slot0 * p.stride_vs;
    unsigned char* ks = KVs + (st * 2) * TK * KVSTR;
    unsigned char* vs = KVs + (st * 2 + 1) * TK * KVSTR;
    for (int i = tid; i < TK * KVCH; i += 128) {
      const int key = i / KVCH, ch = i % KVCH;
      const bool ok = pos0 + key < kv_len;
      const int kn = ok ? min(16, max(0, kbytes - ch * 16)) : 0;
      const int vn = ok ? min(16, max(0, vbytes - ch * 16)) : 0;
      cp_async16(ks + key * KVSTR + ch * 16, kb + (long)key * p.stride_ks + ch * 16, kn);
      cp_async16(vs + key * KVSTR + ch * 16, vb + (long)key * p.stride_vs + ch * 16, vn);
    }
  };

  float acc[MT][NT][4];
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int b = 0; b < NT; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;
  float m_r = -INFINITY, l_r = 0.f;
  const int sr = tid >> 1, shalf = (tid & 1) * (TK / 2);

  if (t0 < t1) load_tile(t0, 0);
  cp_async_commit();
  for (int t = t0; t < t1; ++t) {
    const int st = (t - t0) & 1;
    // Tile t is the only copy in flight: wait for it, then barrier -- which also guarantees every
    // warp is done with tile t-1 (its V in stage st^1) -- and only then start filling stage st^1.
    cp_async_wait<0>();
    __syncthreads();
    if (t + 1 < t1) load_tile(t + 1, st ^ 1);
    cp_async_commit();
    const unsigned char* ks = KVs + (st * 2) * TK * KVSTR + kshift;
    const unsigned char* vs = KVs + (st * 2 + 1) * TK * KVSTR + vshift;
    float* S = Ss;

    // ---- S rows of this warp's row tile, all TK keys, full head dim
    if (warp < MT) {
      float sacc[TK / 8][4];
#pragma unroll
      for (int n = 0; n < TK / 8; ++n) sacc[n][0] = sacc[n][1] = sacc[n][2] = sacc[n][3] = 0.f;
      const unsigned char* kr = ks + gid * KVSTR + tig * 4;
#pragma unroll
      for (int kk = 0; kk < KS; ++kk) {
#pragma unroll
        for (int n = 0; n < TK / 8; ++n) {   // TK/8 independent accumulation chains
          unsigned b0, b1;
          i8x4_f16x4(*reinterpret_cast<const unsigned*>(kr + n * 8 * KVSTR + kk * 16), b0, b1);
          mma_f16(sacc[n], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b0, b1);
        }
      }
#pragma unroll
      for (int n = 0; n < TK / 8; ++n) {
        float* srow = S + (warp * 16 + gid) * SSTR + n * 8 + tig * 2;
        srow[0] = sacc[n][0]; srow[1] = sacc[n][1]; srow[8 * SSTR] = sacc[n][2]; srow[8 * SSTR + 1] = sacc[n][3];
      }
    }
    __syncthreads();

    // ---- scale, mask, online softmax (2 threads per row); zero the other S buffer for tile t+1
    if (sr < ROWS) {
      const int qi = qtile * QT + sr / G;
      const bool rv = (sr < QT * G) && (qi < q_len);
      const int qpos = kv_len - q_len + qi;
      float sv[TK / 2];
      float mx = -INFINITY;
#pragma unroll
      for (int k = 0; k < TK / 2; ++k) {
        const int key = shalf + k, pos = t * TK + key;
        bool ok = rv && pos < kv_len;
        if (p.causal) ok = ok && pos <= qpos;
        if (p.window > 0) ok = ok && (qpos - pos < p.window) && (p.causal || pos - qpos < p.window);
        sv[k] = ok ? S[sr * SSTR + key] * (*reinterpret_cast<const float*>(ks + key * KVSTR + D) * p.scale) : -INFINITY;
        mx = fmaxf(mx, sv[k]);
      }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
      const float m_new = fmaxf(m_r, mx);
      const float m_safe = (m_new == -INFINITY) ? 0.f : m_new;
      const float alpha = (m_r == -INFINITY) ? 0.f : __expf(m_r - m_safe);
      float sum = 0.f;
#pragma unroll
      for (int k = 0; k < TK / 2; ++k) {
        const float pv = __expf(sv[k] - m_safe);
        sum += pv;
        Ps[sr * PSTR + shalf + k] = __float2bfloat16(pv * *reinterpret_cast<const float*>(vs + (shalf + k) * KVSTR + D));
      }
      sum += __shfl_xor_sync(0xffffffff, sum, 1);
      l_r = l_r * alpha + sum;
      m_r = m_new;
      if ((tid & 1) == 0) alpha_s[sr] = alpha;
    }
    __syncthreads();

    // ---- O[:, this warp's dims] = alpha * O + P' V
#pragma unroll
    for (int a = 0; a < MT; ++a) {
      const float al = alpha_s[a * 16 + gid], ah = alpha_s[a * 16 + gid + 8];
#pragma unroll
      for (int n = 0; n < NT; ++n) { acc[a][n][0] *= al; acc[a][n][1] *= al; acc[a][n][2] *= ah; acc[a][n][3] *= ah; }
    }
    const unsigned char* vcol = vs + warp * DW + gid;
#pragma unroll
    for (int kk = 0; kk < TK / 16; ++kk) {
      unsigned af[MT][4];
#pragma unroll
      for (int a = 0; a < MT; ++a)
        ldmatrix_x4(af[a][0], af[a][1], af[a][2], af[a][3], Ps + (a * 16 + (lane & 15)) * PSTR + kk * 16 + (lane >> 4) * 8);
      const int k0 = kk * 16 + tig * 2;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const unsigned char* vc = vcol + n * 8;
        const unsigned b0 = pack_bf16((float)(signed char)vc[k0 * KVSTR], (float)(signed char)vc[(k0 + 1) * KVSTR]);
        const unsigned b1 = pack_bf16((float)(signed char)vc[(k0 + 8) * KVSTR], (float)(signed char)vc[(k0 + 9) * KVSTR]);
#pragma unroll
        for (int a = 0; a < MT; ++a) mma_bf16(acc[a][n], af[a][0], af[a][1], af[a][2], af[a][3], b0, b1);
      }
    }
  }
  cp_async_wait<0>();

  if (sr < ROWS && (tid & 1) == 0) {
    const int qi = qtile * QT + sr / G;
    if (sr < QT * G && qi < q_len) {
      const long pidx = ((long)(req * p.Hq + kvh * G + sr % G) * p.qmax + qi) * p.nseg + seg;
      p.part_m[pidx] = m_r;
      p.part_l[pidx] = l_r;
    }
  }
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = a * 16 + gid + h * 8;
      const int qi = qtile * QT + r / G;
      if (r < QT * G && qi < q_len) {
        const long pidx = ((long)(req * p.Hq + kvh * G + r % G) * p.qmax + qi) * p.nseg + seg;
        float* o = p.part_o + pidx * D + warp * DW + tig * 2;
#pragma unroll
        for (int n = 0; n < NT; ++n) *reinterpret_cast<float2*>(o + n * 8) = make_float2(acc[a][n][h * 2], acc[a][n][h * 2 + 1]);
      }
    }
}



template <int D, int MT>
static void launch_partial_v4(const Params& p, int num_reqs, int hkv, cudaStream_t stream) {
  constexpr int ROWS = MT * 16;
  const int smem = 4 * TK * (D / 16 + 1) * 16 + ROWS * (TK + 1) * 4 + ROWS * (TK + 8) * 2 + ROWS * 4;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(sda_partial_v4_kernel<D, MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
  dim3 grid(num_reqs * p.ntile, hkv, p.nseg);
  sda_partial_v4_kernel<D, MT><<<grid, 128, smem, stream>>>(p);
}

// v5 = v4 + rescale skip (__syncthreads_or), mask-free interior tiles, int8->fp32 via magic FADD in P.V: // v2: Q lives in registers (each warp holds its D/4 slice of every row as mma A fragments, loaded
// once); Q.K^T is split along D across the 4 warps and the partial scores are summed with
// shared-memory float atomics. ~45 KB of shared memory at D=256, so two CTAs fit on an SM.
template <int D, int MT>
__global__ void __launch_bounds__(128, 2) sda_partial_v5_kernel(const Params p) {
  constexpr int ROWS = MT * 16;
  constexpr int KVCH = D / 16 + 1, KVSTR = KVCH * 16;
  constexpr int SSTR = TK + 1, PSTR = TK + 8;
  constexpr int DW = D / 4, NT = DW / 8, KS = D / 16;    // PV: per-warp head-dim slice; QK: full-D k16 steps
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* KVs = smem;                                             // 2 stages x (K, V)
  float* Ss = reinterpret_cast<float*>(KVs + 4 * TK * KVSTR);
  __nv_bfloat16* Ps = reinterpret_cast<__nv_bfloat16*>(Ss + ROWS * SSTR);
  float* alpha_s = reinterpret_cast<float*>(Ps + ROWS * PSTR);

  const int req = blockIdx.x / p.ntile, qtile = blockIdx.x % p.ntile;
  const int kvh = blockIdx.y, seg = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
  const int q_start = p.cu_q[req], q_len = p.cu_q[req + 1] - q_start;
  const int kv_len = p.seqused[req];
  const int G = p.G, QT = p.QT;
  if (qtile * QT >= q_len) return;

  // Q A-fragments: warp w owns row tile w (rows w*16 .. w*16+15) over the full head dim.
  unsigned qa[KS][4];
  const int mw = warp < MT ? warp : 0;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = mw * 16 + gid + h * 8;
    const int qi = qtile * QT + r / G;
    const bool rv = warp < MT && (r < QT * G) && (qi < q_len);
    const __nv_bfloat16* qrow = p.q + (long)(q_start + qi) * p.stride_qt + (long)(kvh * G + r % G) * p.stride_qh;
#pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
      // Q.K^T sums over the head dim, so any permutation applied to both Q and K is exact: thread
      // tig's mma k-slots {2t, 2t+1 | 2t+8, 2t+9} take dims {4t, 4t+1 | 4t+2, 4t+3} of each 16-block,
      // which makes its K fragment one contiguous 4-byte word.
      if (rv) {
        const uint2 w = *reinterpret_cast<const uint2*>(qrow + kk * 16 + tig * 4);
        const __nv_bfloat162 q01 = *reinterpret_cast<const __nv_bfloat162*>(&w.x), q23 = *reinterpret_cast<const __nv_bfloat162*>(&w.y);
        qa[kk][h] = pack_f16(__bfloat162float(q01.x), __bfloat162float(q01.y));
        qa[kk][2 + h] = pack_f16(__bfloat162float(q23.x), __bfloat162float(q23.y));
      } else {
        qa[kk][h] = 0u; qa[kk][2 + h] = 0u;
      }
    }
  }

  const int tiles_total = (kv_len + TK - 1) / TK;
  int t_lo = 0;
  if (p.window > 0) { int lo = kv_len - q_len - p.window + 1; t_lo = lo > 0 ? lo / TK : 0; }
  // Use only as many segments as the context needs (>= p.min_tiles tiles each): every live segment
  // costs a D-float partial per row, written here and re-read by the combine. Idle CTAs exit.
  const int nseg_eff = min(p.nseg, max(1, (tiles_total - t_lo + p.min_tiles - 1) / p.min_tiles));
  if (seg >= nseg_eff) return;
  const int per = (tiles_total - t_lo + nseg_eff - 1) / nseg_eff;
  const int t0 = t_lo + seg * per, t1 = min(t0 + per, tiles_total);

  const unsigned char* khead = p.k + (long)kvh * p.stride_kh;
  const unsigned char* vhead = p.v + (long)kvh * p.stride_vh;
  const int kshift = (int)((uintptr_t)khead & 15), vshift = (int)((uintptr_t)vhead & 15);
  khead -= kshift; vhead -= vshift;
  const int kbytes = kshift + D + 4, vbytes = vshift + D + 4;
  auto load_tile = [&](int t, int st) {
    const int pos0 = t * TK;
    const long blk = p.bt[req * p.stride_bt + pos0 / p.BS];
    const int slot0 = pos0 % p.BS;
    const unsigned char* kb = khead + blk * p.stride_kb + (long)slot0 * p.stride_ks;
    const unsigned char* vb = vhead + blk * p.stride_vb + (long)slot0 * p.stride_vs;
    unsigned char* ks = KVs + (st * 2) * TK * KVSTR;
    unsigned char* vs = KVs + (st * 2 + 1) * TK * KVSTR;
    for (int i = tid; i < TK * KVCH; i += 128) {
      const int key = i / KVCH, ch = i % KVCH;
      const bool ok = pos0 + key < kv_len;
      const int kn = ok ? min(16, max(0, kbytes - ch * 16)) : 0;
      const int vn = ok ? min(16, max(0, vbytes - ch * 16)) : 0;
      cp_async16(ks + key * KVSTR + ch * 16, kb + (long)key * p.stride_ks + ch * 16, kn);
      cp_async16(vs + key * KVSTR + ch * 16, vb + (long)key * p.stride_vs + ch * 16, vn);
    }
  };

  float acc[MT][NT][4];
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int b = 0; b < NT; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;
  float m_r = -INFINITY, l_r = 0.f;
  const int sr = tid >> 1, shalf = (tid & 1) * (TK / 2);

  if (t0 < t1) load_tile(t0, 0);
  cp_async_commit();
  for (int t = t0; t < t1; ++t) {
    const int st = (t - t0) & 1;
    // Tile t is the only copy in flight: wait for it, then barrier -- which also guarantees every
    // warp is done with tile t-1 (its V in stage st^1) -- and only then start filling stage st^1.
    cp_async_wait<0>();
    __syncthreads();
    if (t + 1 < t1) load_tile(t + 1, st ^ 1);
    cp_async_commit();
    const unsigned char* ks = KVs + (st * 2) * TK * KVSTR + kshift;
    const unsigned char* vs = KVs + (st * 2 + 1) * TK * KVSTR + vshift;
    float* S = Ss;

    // ---- S rows of this warp's row tile, all TK keys, full head dim
    if (warp < MT) {
      float sacc[TK / 8][4];
#pragma unroll
      for (int n = 0; n < TK / 8; ++n) sacc[n][0] = sacc[n][1] = sacc[n][2] = sacc[n][3] = 0.f;
      const unsigned char* kr = ks + gid * KVSTR + tig * 4;
#pragma unroll
      for (int kk = 0; kk < KS; ++kk) {
#pragma unroll
        for (int n = 0; n < TK / 8; ++n) {   // TK/8 independent accumulation chains
          unsigned b0, b1;
          i8x4_f16x4(*reinterpret_cast<const unsigned*>(kr + n * 8 * KVSTR + kk * 16), b0, b1);
          mma_f16(sacc[n], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b0, b1);
        }
      }
#pragma unroll
      for (int n = 0; n < TK / 8; ++n) {
        float* srow = S + (warp * 16 + gid) * SSTR + n * 8 + tig * 2;
        srow[0] = sacc[n][0]; srow[1] = sacc[n][1]; srow[8 * SSTR] = sacc[n][2]; srow[8 * SSTR + 1] = sacc[n][3];
      }
    }
    __syncthreads();

    // ---- scale, mask, online softmax (2 threads per row)
    // Interior tiles (every key visible to every query row) skip the per-key mask.
    const int pos_t0 = t * TK;
    const bool full_tile = (p.causal ? (pos_t0 + TK - 1 <= kv_len - q_len) : (pos_t0 + TK <= kv_len)) &&
                           (p.window == 0 || pos_t0 >= kv_len - p.window);
    int changed = 0;
    if (sr < ROWS) {
      const int qi = qtile * QT + sr / G;
      const bool rv = (sr < QT * G) && (qi < q_len);
      const int qpos = kv_len - q_len + qi;
      float sv[TK / 2];
      float mx = -INFINITY;
      if (full_tile) {
#pragma unroll
        for (int k = 0; k < TK / 2; ++k) {
          const int key = shalf + k;
          sv[k] = rv ? S[sr * SSTR + key] * (*reinterpret_cast<const float*>(ks + key * KVSTR + D) * p.scale) : -INFINITY;
          mx = fmaxf(mx, sv[k]);
        }
      } else {
#pragma unroll
        for (int k = 0; k < TK / 2; ++k) {
          const int key = shalf + k, pos = pos_t0 + key;
          bool ok = rv && pos < kv_len;
          if (p.causal) ok = ok && pos <= qpos;
          if (p.window > 0) ok = ok && (qpos - pos < p.window) && (p.causal || pos - qpos < p.window);
          sv[k] = ok ? S[sr * SSTR + key] * (*reinterpret_cast<const float*>(ks + key * KVSTR + D) * p.scale) : -INFINITY;
          mx = fmaxf(mx, sv[k]);
        }
      }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
      const float m_new = fmaxf(m_r, mx);
      const float m_safe = (m_new == -INFINITY) ? 0.f : m_new;
      const float alpha = (m_r == -INFINITY) ? 0.f : __expf(m_r - m_safe);
      float sum = 0.f;
#pragma unroll
      for (int k = 0; k < TK / 2; ++k) {
        const float pv = __expf(sv[k] - m_safe);
        sum += pv;
        Ps[sr * PSTR + shalf + k] = __float2bfloat16(pv * *reinterpret_cast<const float*>(vs + (shalf + k) * KVSTR + D));
      }
      sum += __shfl_xor_sync(0xffffffff, sum, 1);
      l_r = l_r * alpha + sum;
      m_r = m_new;
      if ((tid & 1) == 0) alpha_s[sr] = alpha;
      changed = alpha != 1.f;
    }
    const int any_changed = __syncthreads_or(changed);   // the barrier the P.V phase needs anyway

    // ---- O[:, this warp's dims] = alpha * O + P' V (the rescale is skipped when no row max moved)
    if (any_changed)
#pragma unroll
    for (int a = 0; a < MT; ++a) {
      const float al = alpha_s[a * 16 + gid], ah = alpha_s[a * 16 + gid + 8];
#pragma unroll
      for (int n = 0; n < NT; ++n) { acc[a][n][0] *= al; acc[a][n][1] *= al; acc[a][n][2] *= ah; acc[a][n][3] *= ah; }
    }
    const unsigned char* vcol = vs + warp * DW + gid;
#pragma unroll
    for (int kk = 0; kk < TK / 16; ++kk) {
      unsigned af[MT][4];
#pragma unroll
      for (int a = 0; a < MT; ++a)
        ldmatrix_x4(af[a][0], af[a][1], af[a][2], af[a][3], Ps + (a * 16 + (lane & 15)) * PSTR + kk * 16 + (lane >> 4) * 8);
      const int k0 = kk * 16 + tig * 2;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const unsigned char* vc = vcol + n * 8;
        // int8 -> fp32 exactly without I2F: 0x4B000000 | (u ^ 0x80) is 2^23 + 128 + x as a float
        auto i8f = [](unsigned char u) { return __uint_as_float(0x4B000000u | (unsigned)(u ^ 0x80u)) - 8388736.f; };
        const unsigned b0 = pack_bf16(i8f(vc[k0 * KVSTR]), i8f(vc[(k0 + 1) * KVSTR]));
        const unsigned b1 = pack_bf16(i8f(vc[(k0 + 8) * KVSTR]), i8f(vc[(k0 + 9) * KVSTR]));
#pragma unroll
        for (int a = 0; a < MT; ++a) mma_bf16(acc[a][n], af[a][0], af[a][1], af[a][2], af[a][3], b0, b1);
      }
    }
  }
  cp_async_wait<0>();

  if (sr < ROWS && (tid & 1) == 0) {
    const int qi = qtile * QT + sr / G;
    if (sr < QT * G && qi < q_len) {
      const long pidx = ((long)(req * p.Hq + kvh * G + sr % G) * p.qmax + qi) * p.nseg + seg;
      p.part_m[pidx] = m_r;
      p.part_l[pidx] = l_r;
    }
  }
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = a * 16 + gid + h * 8;
      const int qi = qtile * QT + r / G;
      if (r < QT * G && qi < q_len) {
        const long pidx = ((long)(req * p.Hq + kvh * G + r % G) * p.qmax + qi) * p.nseg + seg;
        float* o = p.part_o + pidx * D + warp * DW + tig * 2;
#pragma unroll
        for (int n = 0; n < NT; ++n) *reinterpret_cast<float2*>(o + n * 8) = make_float2(acc[a][n][h * 2], acc[a][n][h * 2 + 1]);
      }
    }
}




template <int D, int MT>
static void launch_partial_v5(const Params& p, int num_reqs, int hkv, cudaStream_t stream) {
  constexpr int ROWS = MT * 16;
  const int smem = 4 * TK * (D / 16 + 1) * 16 + ROWS * (TK + 1) * 4 + ROWS * (TK + 8) * 2 + ROWS * 4;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(sda_partial_v5_kernel<D, MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
  dim3 grid(num_reqs * p.ntile, hkv, p.nseg);
  sda_partial_v5_kernel<D, MT><<<grid, 128, smem, stream>>>(p);
}

// v6 = v4 with the online softmax done in registers by the row-tile warps (4-lane shuffles): no fp32 S buffer, one barrier less per tile: // v2: Q lives in registers (each warp holds its D/4 slice of every row as mma A fragments, loaded
// once); Q.K^T is split along D across the 4 warps and the partial scores are summed with
// shared-memory float atomics. ~45 KB of shared memory at D=256, so two CTAs fit on an SM.
template <int D, int MT>
__global__ void __launch_bounds__(128, 2) sda_partial_v6_kernel(const Params p) {
  constexpr int ROWS = MT * 16;
  constexpr int KVCH = D / 16 + 1, KVSTR = KVCH * 16;
  constexpr int SSTR = TK + 1, PSTR = TK + 8;
  constexpr int DW = D / 4, NT = DW / 8, KS = D / 16;    // PV: per-warp head-dim slice; QK: full-D k16 steps
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* KVs = smem;                                             // 2 stages x (K, V)
  __nv_bfloat16* Ps = reinterpret_cast<__nv_bfloat16*>(KVs + 4 * TK * KVSTR);
  float* alpha_s = reinterpret_cast<float*>(Ps + ROWS * PSTR);

  const int req = blockIdx.x / p.ntile, qtile = blockIdx.x % p.ntile;
  const int kvh = blockIdx.y, seg = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
  const int q_start = p.cu_q[req], q_len = p.cu_q[req + 1] - q_start;
  const int kv_len = p.seqused[req];
  const int G = p.G, QT = p.QT;
  if (qtile * QT >= q_len) return;

  // Q A-fragments: warp w owns row tile w (rows w*16 .. w*16+15) over the full head dim.
  unsigned qa[KS][4];
  const int mw = warp < MT ? warp : 0;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = mw * 16 + gid + h * 8;
    const int qi = qtile * QT + r / G;
    const bool rv = warp < MT && (r < QT * G) && (qi < q_len);
    const __nv_bfloat16* qrow = p.q + (long)(q_start + qi) * p.stride_qt + (long)(kvh * G + r % G) * p.stride_qh;
#pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
      // Q.K^T sums over the head dim, so any permutation applied to both Q and K is exact: thread
      // tig's mma k-slots {2t, 2t+1 | 2t+8, 2t+9} take dims {4t, 4t+1 | 4t+2, 4t+3} of each 16-block,
      // which makes its K fragment one contiguous 4-byte word.
      if (rv) {
        const uint2 w = *reinterpret_cast<const uint2*>(qrow + kk * 16 + tig * 4);
        const __nv_bfloat162 q01 = *reinterpret_cast<const __nv_bfloat162*>(&w.x), q23 = *reinterpret_cast<const __nv_bfloat162*>(&w.y);
        qa[kk][h] = pack_f16(__bfloat162float(q01.x), __bfloat162float(q01.y));
        qa[kk][2 + h] = pack_f16(__bfloat162float(q23.x), __bfloat162float(q23.y));
      } else {
        qa[kk][h] = 0u; qa[kk][2 + h] = 0u;
      }
    }
  }

  const int tiles_total = (kv_len + TK - 1) / TK;
  int t_lo = 0;
  if (p.window > 0) { int lo = kv_len - q_len - p.window + 1; t_lo = lo > 0 ? lo / TK : 0; }
  // Use only as many segments as the context needs (>= p.min_tiles tiles each): every live segment
  // costs a D-float partial per row, written here and re-read by the combine. Idle CTAs exit.
  const int nseg_eff = min(p.nseg, max(1, (tiles_total - t_lo + p.min_tiles - 1) / p.min_tiles));
  if (seg >= nseg_eff) return;
  const int per = (tiles_total - t_lo + nseg_eff - 1) / nseg_eff;
  const int t0 = t_lo + seg * per, t1 = min(t0 + per, tiles_total);

  const unsigned char* khead = p.k + (long)kvh * p.stride_kh;
  const unsigned char* vhead = p.v + (long)kvh * p.stride_vh;
  const int kshift = (int)((uintptr_t)khead & 15), vshift = (int)((uintptr_t)vhead & 15);
  khead -= kshift; vhead -= vshift;
  const int kbytes = kshift + D + 4, vbytes = vshift + D + 4;
  auto load_tile = [&](int t, int st) {
    const int pos0 = t * TK;
    const long blk = p.bt[req * p.stride_bt + pos0 / p.BS];
    const int slot0 = pos0 % p.BS;
    const unsigned char* kb = khead + blk * p.stride_kb + (long)slot0 * p.stride_ks;
    const unsigned char* vb = vhead + blk * p.stride_vb + (long)slot0 * p.stride_vs;
    unsigned char* ks = KVs + (st * 2) * TK * KVSTR;
    unsigned char* vs = KVs + (st * 2 + 1) * TK * KVSTR;
    for (int i = tid; i < TK * KVCH; i += 128) {
      const int key = i / KVCH, ch = i % KVCH;
      const bool ok = pos0 + key < kv_len;
      const int kn = ok ? min(16, max(0, kbytes - ch * 16)) : 0;
      const int vn = ok ? min(16, max(0, vbytes - ch * 16)) : 0;
      cp_async16(ks + key * KVSTR + ch * 16, kb + (long)key * p.stride_ks + ch * 16, kn);
      cp_async16(vs + key * KVSTR + ch * 16, vb + (long)key * p.stride_vs + ch * 16, vn);
    }
  };

  float acc[MT][NT][4];
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int b = 0; b < NT; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;
  float m_lo = -INFINITY, m_hi = -INFINITY, l_lo = 0.f, l_hi = 0.f;   // rows mw*16+gid and +8
  const int r_lo = mw * 16 + gid, r_hi = r_lo + 8;
  const int qi_lo = qtile * QT + r_lo / G, qi_hi = qtile * QT + r_hi / G;
  const bool rv_lo = warp < MT && r_lo < QT * G && qi_lo < q_len;
  const bool rv_hi = warp < MT && r_hi < QT * G && qi_hi < q_len;
  const int qpos_lo = kv_len - q_len + qi_lo, qpos_hi = kv_len - q_len + qi_hi;

  if (t0 < t1) load_tile(t0, 0);
  cp_async_commit();
  for (int t = t0; t < t1; ++t) {
    const int st = (t - t0) & 1;
    // Tile t is the only copy in flight: wait for it, then barrier -- which also guarantees every
    // warp is done with tile t-1 (its V in stage st^1) -- and only then start filling stage st^1.
    cp_async_wait<0>();
    __syncthreads();
    if (t + 1 < t1) load_tile(t + 1, st ^ 1);
    cp_async_commit();
    const unsigned char* ks = KVs + (st * 2) * TK * KVSTR + kshift;
    const unsigned char* vs = KVs + (st * 2 + 1) * TK * KVSTR + vshift;

    // ---- S rows of this warp's row tile, all TK keys, full head dim
    if (warp < MT) {
      float sacc[TK / 8][4];
#pragma unroll
      for (int n = 0; n < TK / 8; ++n) sacc[n][0] = sacc[n][1] = sacc[n][2] = sacc[n][3] = 0.f;
      const unsigned char* kr = ks + gid * KVSTR + tig * 4;
#pragma unroll
      for (int kk = 0; kk < KS; ++kk) {
#pragma unroll
        for (int n = 0; n < TK / 8; ++n) {   // TK/8 independent accumulation chains
          unsigned b0, b1;
          i8x4_f16x4(*reinterpret_cast<const unsigned*>(kr + n * 8 * KVSTR + kk * 16), b0, b1);
          mma_f16(sacc[n], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b0, b1);
        }
      }
      // scale + mask in registers; this thread holds keys n*8 + tig*2 + {0,1} of rows r_lo / r_hi
      float mx_lo = -INFINITY, mx_hi = -INFINITY;
#pragma unroll
      for (int n = 0; n < TK / 8; ++n)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int key = n * 8 + tig * 2 + j, pos = t * TK + key;
          const float ksc = *reinterpret_cast<const float*>(ks + key * KVSTR + D) * p.scale;
          bool ok_lo = rv_lo && pos < kv_len, ok_hi = rv_hi && pos < kv_len;
          if (p.causal) { ok_lo = ok_lo && pos <= qpos_lo; ok_hi = ok_hi && pos <= qpos_hi; }
          if (p.window > 0) {
            ok_lo = ok_lo && (qpos_lo - pos < p.window) && (p.causal || pos - qpos_lo < p.window);
            ok_hi = ok_hi && (qpos_hi - pos < p.window) && (p.causal || pos - qpos_hi < p.window);
          }
          sacc[n][j] = ok_lo ? sacc[n][j] * ksc : -INFINITY;
          sacc[n][2 + j] = ok_hi ? sacc[n][2 + j] * ksc : -INFINITY;
          mx_lo = fmaxf(mx_lo, sacc[n][j]); mx_hi = fmaxf(mx_hi, sacc[n][2 + j]);
        }
      mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffff, mx_lo, 1)); mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffff, mx_lo, 2));
      mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffff, mx_hi, 1)); mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffff, mx_hi, 2));
      const float mn_lo = fmaxf(m_lo, mx_lo), mn_hi = fmaxf(m_hi, mx_hi);
      const float ms_lo = mn_lo == -INFINITY ? 0.f : mn_lo, ms_hi = mn_hi == -INFINITY ? 0.f : mn_hi;
      const float al_lo = m_lo == -INFINITY ? 0.f : __expf(m_lo - ms_lo), al_hi = m_hi == -INFINITY ? 0.f : __expf(m_hi - ms_hi);
      float sum_lo = 0.f, sum_hi = 0.f;
#pragma unroll
      for (int n = 0; n < TK / 8; ++n) {
        const int key = n * 8 + tig * 2;
        const float vs0 = *reinterpret_cast<const float*>(vs + key * KVSTR + D);
        const float vs1 = *reinterpret_cast<const float*>(vs + (key + 1) * KVSTR + D);
        const float p0 = __expf(sacc[n][0] - ms_lo), p1 = __expf(sacc[n][1] - ms_lo);
        const float p2 = __expf(sacc[n][2] - ms_hi), p3 = __expf(sacc[n][3] - ms_hi);
        sum_lo += p0 + p1; sum_hi += p2 + p3;
        *reinterpret_cast<__nv_bfloat162*>(Ps + r_lo * PSTR + key) = __floats2bfloat162_rn(p0 * vs0, p1 * vs1);
        *reinterpret_cast<__nv_bfloat162*>(Ps + r_hi * PSTR + key) = __floats2bfloat162_rn(p2 * vs0, p3 * vs1);
      }
      sum_lo += __shfl_xor_sync(0xffffffff, sum_lo, 1); sum_lo += __shfl_xor_sync(0xffffffff, sum_lo, 2);
      sum_hi += __shfl_xor_sync(0xffffffff, sum_hi, 1); sum_hi += __shfl_xor_sync(0xffffffff, sum_hi, 2);
      l_lo = l_lo * al_lo + sum_lo; l_hi = l_hi * al_hi + sum_hi;
      m_lo = mn_lo; m_hi = mn_hi;
      if (tig == 0) { alpha_s[r_lo] = al_lo; alpha_s[r_hi] = al_hi; }
    }
    __syncthreads();

    // ---- O[:, this warp's dims] = alpha * O + P' V
#pragma unroll
    for (int a = 0; a < MT; ++a) {
      const float al = alpha_s[a * 16 + gid], ah = alpha_s[a * 16 + gid + 8];
#pragma unroll
      for (int n = 0; n < NT; ++n) { acc[a][n][0] *= al; acc[a][n][1] *= al; acc[a][n][2] *= ah; acc[a][n][3] *= ah; }
    }
    const unsigned char* vcol = vs + warp * DW + gid;
#pragma unroll
    for (int kk = 0; kk < TK / 16; ++kk) {
      unsigned af[MT][4];
#pragma unroll
      for (int a = 0; a < MT; ++a)
        ldmatrix_x4(af[a][0], af[a][1], af[a][2], af[a][3], Ps + (a * 16 + (lane & 15)) * PSTR + kk * 16 + (lane >> 4) * 8);
      const int k0 = kk * 16 + tig * 2;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const unsigned char* vc = vcol + n * 8;
        const unsigned b0 = pack_bf16((float)(signed char)vc[k0 * KVSTR], (float)(signed char)vc[(k0 + 1) * KVSTR]);
        const unsigned b1 = pack_bf16((float)(signed char)vc[(k0 + 8) * KVSTR], (float)(signed char)vc[(k0 + 9) * KVSTR]);
#pragma unroll
        for (int a = 0; a < MT; ++a) mma_bf16(acc[a][n], af[a][0], af[a][1], af[a][2], af[a][3], b0, b1);
      }
    }
  }
  cp_async_wait<0>();

  if (warp < MT && tig == 0) {
    if (rv_lo) {
      const long pidx = ((long)(req * p.Hq + kvh * G + r_lo % G) * p.qmax + qi_lo) * p.nseg + seg;
      p.part_m[pidx] = m_lo; p.part_l[pidx] = l_lo;
    }
    if (rv_hi) {
      const long pidx = ((long)(req * p.Hq + kvh * G + r_hi % G) * p.qmax + qi_hi) * p.nseg + seg;
      p.part_m[pidx] = m_hi; p.part_l[pidx] = l_hi;
    }
  }
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = a * 16 + gid + h * 8;
      const int qi = qtile * QT + r / G;
      if (r < QT * G && qi < q_len) {
        const long pidx = ((long)(req * p.Hq + kvh * G + r % G) * p.qmax + qi) * p.nseg + seg;
        float* o = p.part_o + pidx * D + warp * DW + tig * 2;
#pragma unroll
        for (int n = 0; n < NT; ++n) *reinterpret_cast<float2*>(o + n * 8) = make_float2(acc[a][n][h * 2], acc[a][n][h * 2 + 1]);
      }
    }
}




template <int D, int MT>
static void launch_partial_v6(const Params& p, int num_reqs, int hkv, cudaStream_t stream) {
  constexpr int ROWS = MT * 16;
  const int smem = 4 * TK * (D / 16 + 1) * 16 + ROWS * (TK + 8) * 2 + ROWS * 4;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(sda_partial_v6_kernel<D, MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
  dim3 grid(num_reqs * p.ntile, hkv, p.nseg);
  sda_partial_v6_kernel<D, MT><<<grid, 128, smem, stream>>>(p);
}

// v7 = v6 with Q.K^T on int8 tensor cores: Q split per row into two int8 levels (q ~ s*q_hi + s/256*q_lo, ~15-bit), raw int8 K as B (no conversion), exact int32 accumulation: // v2: Q lives in registers (each warp holds its D/4 slice of every row as mma A fragments, loaded
// once); Q.K^T is split along D across the 4 warps and the partial scores are summed with
// shared-memory float atomics. ~45 KB of shared memory at D=256, so two CTAs fit on an SM.
template <int D, int MT>
__global__ void __launch_bounds__(128, 2) sda_partial_v7_kernel(const Params p) {
  constexpr int ROWS = MT * 16;
  constexpr int KVCH = D / 16 + 1, KVSTR = KVCH * 16;
  constexpr int SSTR = TK + 1, PSTR = TK + 8;
  constexpr int DW = D / 4, NT = DW / 8, KS = D / 16;    // PV: per-warp head-dim slice; QK: full-D k16 steps
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* KVs = smem;                                             // 2 stages x (K, V)
  __nv_bfloat16* Ps = reinterpret_cast<__nv_bfloat16*>(KVs + 4 * TK * KVSTR);
  float* alpha_s = reinterpret_cast<float*>(Ps + ROWS * PSTR);

  const int req = blockIdx.x / p.ntile, qtile = blockIdx.x % p.ntile;
  const int kvh = blockIdx.y, seg = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
  const int q_start = p.cu_q[req], q_len = p.cu_q[req + 1] - q_start;
  const int kv_len = p.seqused[req];
  const int G = p.G, QT = p.QT;
  if (qtile * QT >= q_len) return;

  // Q A-fragments: warp w owns row tile w (rows w*16 .. w*16+15) over the full head dim.
  constexpr int KC = D / 32;
  unsigned qh[KC][4], ql[KC][4];
  float qsc[2];   // per-row scale of rows gid / gid+8 of this warp's tile (q_hi units)
  {
    const int mw0 = warp < MT ? warp : 0;
    float qv[2][KC][8];
    float amax[2] = {0.f, 0.f};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = mw0 * 16 + gid + h * 8;
      const int qi = qtile * QT + r / G;
      const bool rv = warp < MT && (r < QT * G) && (qi < q_len);
      const __nv_bfloat16* qrow = p.q + (long)(q_start + qi) * p.stride_qt + (long)(kvh * G + r % G) * p.stride_qh;
#pragma unroll
      for (int c = 0; c < KC; ++c)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          float4 f = make_float4(0.f, 0.f, 0.f, 0.f);
          if (rv) {
            const uint2 w = *reinterpret_cast<const uint2*>(qrow + c * 32 + half * 16 + tig * 4);
            const __nv_bfloat162 a01 = *reinterpret_cast<const __nv_bfloat162*>(&w.x), a23 = *reinterpret_cast<const __nv_bfloat162*>(&w.y);
            f = make_float4(__bfloat162float(a01.x), __bfloat162float(a01.y), __bfloat162float(a23.x), __bfloat162float(a23.y));
          }
          qv[h][c][half * 4 + 0] = f.x; qv[h][c][half * 4 + 1] = f.y; qv[h][c][half * 4 + 2] = f.z; qv[h][c][half * 4 + 3] = f.w;
          amax[h] = fmaxf(amax[h], fmaxf(fmaxf(fabsf(f.x), fabsf(f.y)), fmaxf(fabsf(f.z), fabsf(f.w))));
        }
      amax[h] = fmaxf(amax[h], __shfl_xor_sync(0xffffffff, amax[h], 1));
      amax[h] = fmaxf(amax[h], __shfl_xor_sync(0xffffffff, amax[h], 2));
      qsc[h] = amax[h] > 0.f ? amax[h] / 127.f : 1.f;
    }
#pragma unroll
    for (int c = 0; c < KC; ++c)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        int hi[8], lo[8];
        const float inv = 1.f / qsc[h];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float x = qv[h][c][e] * inv;
          const float xh = rintf(x);
          hi[e] = (int)xh;
          lo[e] = max(-127, min(127, (int)rintf((x - xh) * 256.f)));
        }
        qh[c][h] = pack_s8x4(hi[0], hi[1], hi[2], hi[3]); qh[c][2 + h] = pack_s8x4(hi[4], hi[5], hi[6], hi[7]);
        ql[c][h] = pack_s8x4(lo[0], lo[1], lo[2], lo[3]); ql[c][2 + h] = pack_s8x4(lo[4], lo[5], lo[6], lo[7]);
      }
  }
  const int mw = warp < MT ? warp : 0;

  const int tiles_total = (kv_len + TK - 1) / TK;
  int t_lo = 0;
  if (p.window > 0) { int lo = kv_len - q_len - p.window + 1; t_lo = lo > 0 ? lo / TK : 0; }
  // Use only as many segments as the context needs (>= p.min_tiles tiles each): every live segment
  // costs a D-float partial per row, written here and re-read by the combine. Idle CTAs exit.
  const int nseg_eff = min(p.nseg, max(1, (tiles_total - t_lo + p.min_tiles - 1) / p.min_tiles));
  if (seg >= nseg_eff) return;
  const int per = (tiles_total - t_lo + nseg_eff - 1) / nseg_eff;
  const int t0 = t_lo + seg * per, t1 = min(t0 + per, tiles_total);

  const unsigned char* khead = p.k + (long)kvh * p.stride_kh;
  const unsigned char* vhead = p.v + (long)kvh * p.stride_vh;
  const int kshift = (int)((uintptr_t)khead & 15), vshift = (int)((uintptr_t)vhead & 15);
  khead -= kshift; vhead -= vshift;
  const int kbytes = kshift + D + 4, vbytes = vshift + D + 4;
  auto load_tile = [&](int t, int st) {
    const int pos0 = t * TK;
    const long blk = p.bt[req * p.stride_bt + pos0 / p.BS];
    const int slot0 = pos0 % p.BS;
    const unsigned char* kb = khead + blk * p.stride_kb + (long)slot0 * p.stride_ks;
    const unsigned char* vb = vhead + blk * p.stride_vb + (long)slot0 * p.stride_vs;
    unsigned char* ks = KVs + (st * 2) * TK * KVSTR;
    unsigned char* vs = KVs + (st * 2 + 1) * TK * KVSTR;
    for (int i = tid; i < TK * KVCH; i += 128) {
      const int key = i / KVCH, ch = i % KVCH;
      const bool ok = pos0 + key < kv_len;
      const int kn = ok ? min(16, max(0, kbytes - ch * 16)) : 0;
      const int vn = ok ? min(16, max(0, vbytes - ch * 16)) : 0;
      cp_async16(ks + key * KVSTR + ch * 16, kb + (long)key * p.stride_ks + ch * 16, kn);
      cp_async16(vs + key * KVSTR + ch * 16, vb + (long)key * p.stride_vs + ch * 16, vn);
    }
  };

  float acc[MT][NT][4];
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int b = 0; b < NT; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;
  float m_lo = -INFINITY, m_hi = -INFINITY, l_lo = 0.f, l_hi = 0.f;   // rows mw*16+gid and +8
  const int r_lo = mw * 16 + gid, r_hi = r_lo + 8;
  const int qi_lo = qtile * QT + r_lo / G, qi_hi = qtile * QT + r_hi / G;
  const bool rv_lo = warp < MT && r_lo < QT * G && qi_lo < q_len;
  const bool rv_hi = warp < MT && r_hi < QT * G && qi_hi < q_len;
  const int qpos_lo = kv_len - q_len + qi_lo, qpos_hi = kv_len - q_len + qi_hi;

  if (t0 < t1) load_tile(t0, 0);
  cp_async_commit();
  for (int t = t0; t < t1; ++t) {
    const int st = (t - t0) & 1;
    // Tile t is the only copy in flight: wait for it, then barrier -- which also guarantees every
    // warp is done with tile t-1 (its V in stage st^1) -- and only then start filling stage st^1.
    cp_async_wait<0>();
    __syncthreads();
    if (t + 1 < t1) load_tile(t + 1, st ^ 1);
    cp_async_commit();
    const unsigned char* ks = KVs + (st * 2) * TK * KVSTR + kshift;
    const unsigned char* vs = KVs + (st * 2 + 1) * TK * KVSTR + vshift;

    // ---- S rows of this warp's row tile, all TK keys, full head dim
    if (warp < MT) {
      int ah[TK / 8][4], al[TK / 8][4];
#pragma unroll
      for (int n = 0; n < TK / 8; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) { ah[n][e] = 0; al[n][e] = 0; }
      const unsigned char* kr = ks + gid * KVSTR + tig * 4;
#pragma unroll
      for (int c = 0; c < KC; ++c) {
#pragma unroll
        for (int n = 0; n < TK / 8; ++n) {
          const unsigned b0 = *reinterpret_cast<const unsigned*>(kr + n * 8 * KVSTR + c * 32);
          const unsigned b1 = *reinterpret_cast<const unsigned*>(kr + n * 8 * KVSTR + c * 32 + 16);
          mma_s8(ah[n], qh[c][0], qh[c][1], qh[c][2], qh[c][3], b0, b1);
          mma_s8(al[n], ql[c][0], ql[c][1], ql[c][2], ql[c][3], b0, b1);
        }
      }
      float sacc[TK / 8][4];
#pragma unroll
      for (int n = 0; n < TK / 8; ++n) {
        sacc[n][0] = ((float)ah[n][0] + (float)al[n][0] * (1.f / 256.f)) * qsc[0];
        sacc[n][1] = ((float)ah[n][1] + (float)al[n][1] * (1.f / 256.f)) * qsc[0];
        sacc[n][2] = ((float)ah[n][2] + (float)al[n][2] * (1.f / 256.f)) * qsc[1];
        sacc[n][3] = ((float)ah[n][3] + (float)al[n][3] * (1.f / 256.f)) * qsc[1];
      }
      // scale + mask in registers; this thread holds keys n*8 + tig*2 + {0,1} of rows r_lo / r_hi
      float mx_lo = -INFINITY, mx_hi = -INFINITY;
#pragma unroll
      for (int n = 0; n < TK / 8; ++n)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int key = n * 8 + tig * 2 + j, pos = t * TK + key;
          const float ksc = *reinterpret_cast<const float*>(ks + key * KVSTR + D) * p.scale;
          bool ok_lo = rv_lo && pos < kv_len, ok_hi = rv_hi && pos < kv_len;
          if (p.causal) { ok_lo = ok_lo && pos <= qpos_lo; ok_hi = ok_hi && pos <= qpos_hi; }
          if (p.window > 0) {
            ok_lo = ok_lo && (qpos_lo - pos < p.window) && (p.causal || pos - qpos_lo < p.window);
            ok_hi = ok_hi && (qpos_hi - pos < p.window) && (p.causal || pos - qpos_hi < p.window);
          }
          sacc[n][j] = ok_lo ? sacc[n][j] * ksc : -INFINITY;
          sacc[n][2 + j] = ok_hi ? sacc[n][2 + j] * ksc : -INFINITY;
          mx_lo = fmaxf(mx_lo, sacc[n][j]); mx_hi = fmaxf(mx_hi, sacc[n][2 + j]);
        }
      mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffff, mx_lo, 1)); mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffff, mx_lo, 2));
      mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffff, mx_hi, 1)); mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffff, mx_hi, 2));
      const float mn_lo = fmaxf(m_lo, mx_lo), mn_hi = fmaxf(m_hi, mx_hi);
      const float ms_lo = mn_lo == -INFINITY ? 0.f : mn_lo, ms_hi = mn_hi == -INFINITY ? 0.f : mn_hi;
      const float al_lo = m_lo == -INFINITY ? 0.f : __expf(m_lo - ms_lo), al_hi = m_hi == -INFINITY ? 0.f : __expf(m_hi - ms_hi);
      float sum_lo = 0.f, sum_hi = 0.f;
#pragma unroll
      for (int n = 0; n < TK / 8; ++n) {
        const int key = n * 8 + tig * 2;
        const float vs0 = *reinterpret_cast<const float*>(vs + key * KVSTR + D);
        const float vs1 = *reinterpret_cast<const float*>(vs + (key + 1) * KVSTR + D);
        const float p0 = __expf(sacc[n][0] - ms_lo), p1 = __expf(sacc[n][1] - ms_lo);
        const float p2 = __expf(sacc[n][2] - ms_hi), p3 = __expf(sacc[n][3] - ms_hi);
        sum_lo += p0 + p1; sum_hi += p2 + p3;
        *reinterpret_cast<__nv_bfloat162*>(Ps + r_lo * PSTR + key) = __floats2bfloat162_rn(p0 * vs0, p1 * vs1);
        *reinterpret_cast<__nv_bfloat162*>(Ps + r_hi * PSTR + key) = __floats2bfloat162_rn(p2 * vs0, p3 * vs1);
      }
      sum_lo += __shfl_xor_sync(0xffffffff, sum_lo, 1); sum_lo += __shfl_xor_sync(0xffffffff, sum_lo, 2);
      sum_hi += __shfl_xor_sync(0xffffffff, sum_hi, 1); sum_hi += __shfl_xor_sync(0xffffffff, sum_hi, 2);
      l_lo = l_lo * al_lo + sum_lo; l_hi = l_hi * al_hi + sum_hi;
      m_lo = mn_lo; m_hi = mn_hi;
      if (tig == 0) { alpha_s[r_lo] = al_lo; alpha_s[r_hi] = al_hi; }
    }
    __syncthreads();

    // ---- O[:, this warp's dims] = alpha * O + P' V
#pragma unroll
    for (int a = 0; a < MT; ++a) {
      const float al = alpha_s[a * 16 + gid], ah = alpha_s[a * 16 + gid + 8];
#pragma unroll
      for (int n = 0; n < NT; ++n) { acc[a][n][0] *= al; acc[a][n][1] *= al; acc[a][n][2] *= ah; acc[a][n][3] *= ah; }
    }
    const unsigned char* vcol = vs + warp * DW + gid;
#pragma unroll
    for (int kk = 0; kk < TK / 16; ++kk) {
      unsigned af[MT][4];
#pragma unroll
      for (int a = 0; a < MT; ++a)
        ldmatrix_x4(af[a][0], af[a][1], af[a][2], af[a][3], Ps + (a * 16 + (lane & 15)) * PSTR + kk * 16 + (lane >> 4) * 8);
      const int k0 = kk * 16 + tig * 2;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const unsigned char* vc = vcol + n * 8;
        const unsigned b0 = pack_bf16((float)(signed char)vc[k0 * KVSTR], (float)(signed char)vc[(k0 + 1) * KVSTR]);
        const unsigned b1 = pack_bf16((float)(signed char)vc[(k0 + 8) * KVSTR], (float)(signed char)vc[(k0 + 9) * KVSTR]);
#pragma unroll
        for (int a = 0; a < MT; ++a) mma_bf16(acc[a][n], af[a][0], af[a][1], af[a][2], af[a][3], b0, b1);
      }
    }
  }
  cp_async_wait<0>();

  if (warp < MT && tig == 0) {
    if (rv_lo) {
      const long pidx = ((long)(req * p.Hq + kvh * G + r_lo % G) * p.qmax + qi_lo) * p.nseg + seg;
      p.part_m[pidx] = m_lo; p.part_l[pidx] = l_lo;
    }
    if (rv_hi) {
      const long pidx = ((long)(req * p.Hq + kvh * G + r_hi % G) * p.qmax + qi_hi) * p.nseg + seg;
      p.part_m[pidx] = m_hi; p.part_l[pidx] = l_hi;
    }
  }
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = a * 16 + gid + h * 8;
      const int qi = qtile * QT + r / G;
      if (r < QT * G && qi < q_len) {
        const long pidx = ((long)(req * p.Hq + kvh * G + r % G) * p.qmax + qi) * p.nseg + seg;
        float* o = p.part_o + pidx * D + warp * DW + tig * 2;
#pragma unroll
        for (int n = 0; n < NT; ++n) *reinterpret_cast<float2*>(o + n * 8) = make_float2(acc[a][n][h * 2], acc[a][n][h * 2 + 1]);
      }
    }
}





template <int D, int MT>
static void launch_partial_v7(const Params& p, int num_reqs, int hkv, cudaStream_t stream) {
  constexpr int ROWS = MT * 16;
  const int smem = 4 * TK * (D / 16 + 1) * 16 + ROWS * (TK + 8) * 2 + ROWS * 4;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(sda_partial_v7_kernel<D, MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
  dim3 grid(num_reqs * p.ntile, hkv, p.nseg);
  sda_partial_v7_kernel<D, MT><<<grid, 128, smem, stream>>>(p);
}

// q [T, Hq, D] bf16; kc/vc int8 views [nb, BS, Hkv, D] (head rows D + 4 bytes, scale inline).
void sda_run(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor out, torch::Tensor cu_q,
             torch::Tensor seqused, torch::Tensor bt, torch::Tensor part_o, torch::Tensor part_m, torch::Tensor part_l,
             double scale, int64_t num_reqs, int64_t max_q, int64_t qmax, int64_t nseg, int64_t window, bool causal) {
  const int Hq = q.size(1), D = q.size(2), Hkv = kc.size(2), G = Hq / Hkv;
  TORCH_CHECK(D == 128 || D == 256, "sda: D must be 128 or 256");
  TORCH_CHECK(kc.stride(3) == 1 && vc.stride(3) == 1 && kc.stride(2) % 4 == 0 && vc.stride(2) % 4 == 0, "sda: head rows must be 4-byte aligned");
  TORCH_CHECK(kc.stride(1) % 16 == 0 && kc.stride(0) % 16 == 0 && vc.stride(1) % 16 == 0 && vc.stride(0) % 16 == 0, "sda: token rows must be 16-byte aligned");
  TORCH_CHECK(((uintptr_t)kc.data_ptr() % 4) == 0 && ((uintptr_t)vc.data_ptr() % 4) == 0, "sda: cache base must be 4-byte aligned");
  TORCH_CHECK(kc.size(1) % TK == 0, "sda: block size must be a multiple of 32");
  TORCH_CHECK(q.stride(2) == 1 && (q.stride(1) % 8) == 0 && (q.stride(0) % 8) == 0, "sda: q rows must be 16-byte aligned");
  int rows = (int)max_q * G;
  int MT = (rows + 15) / 16; if (MT > 4) MT = 4;
  int QT = MT * 16 / G;
  Params p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()); p.stride_qt = q.stride(0); p.stride_qh = q.stride(1);
  p.k = reinterpret_cast<const unsigned char*>(kc.data_ptr()); p.v = reinterpret_cast<const unsigned char*>(vc.data_ptr());
  p.stride_kb = kc.stride(0); p.stride_ks = kc.stride(1); p.stride_kh = kc.stride(2); p.stride_vb = vc.stride(0); p.stride_vs = vc.stride(1); p.stride_vh = vc.stride(2);
  p.bt = bt.data_ptr<int>(); p.stride_bt = bt.stride(0); p.BS = kc.size(1);
  p.seqused = seqused.data_ptr<int>(); p.cu_q = cu_q.data_ptr<int>();
  p.part_o = part_o.data_ptr<float>(); p.part_m = part_m.data_ptr<float>(); p.part_l = part_l.data_ptr<float>();
  p.scale = (float)scale; p.Hq = Hq; p.G = G; p.qmax = (int)qmax; p.nseg = (int)nseg; p.QT = QT;
  p.ntile = ((int)max_q + QT - 1) / QT; p.window = (int)window; p.causal = causal ? 1 : 0;
  { const char* mt = getenv("SDA_MIN_TILES"); p.min_tiles = mt ? atoi(mt) : 4; }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // v4 (Q in registers as fp16, 2 CTAs/SM; the caller uses 2x the segments) for both the target's
  // D=256 and the drafter's D=128 layers. SDA_VER=1|3 selects the older variants (tests).
  const char* ver = getenv("SDA_VER"); const int kv = ver ? atoi(ver) : 7;
#define L(DD, M) if (D == DD && MT == M) { if (kv == 7) launch_partial_v7<DD, M>(p, (int)num_reqs, Hkv, stream); else if (kv == 6) launch_partial_v6<DD, M>(p, (int)num_reqs, Hkv, stream); else if (kv == 5) launch_partial_v5<DD, M>(p, (int)num_reqs, Hkv, stream); else if (kv == 4) launch_partial_v4<DD, M>(p, (int)num_reqs, Hkv, stream); else if (kv == 3) launch_partial_v3<DD, M>(p, (int)num_reqs, Hkv, stream); else if (kv == 2) launch_partial_v2<DD, M>(p, (int)num_reqs, Hkv, stream); else launch_partial<DD, M>(p, (int)num_reqs, Hkv, stream); }
  L(256, 1) L(256, 2) L(256, 3) L(256, 4) L(128, 1) L(128, 2) L(128, 3) L(128, 4)
#undef L
  dim3 cgrid((int)num_reqs, Hq, (int)max_q);
  if (D == 256)
    sda_combine_kernel<256><<<cgrid, 256, 0, stream>>>(p.part_o, p.part_m, p.part_l, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), p.cu_q, out.stride(0), out.stride(1), Hq, (int)qmax, (int)nseg, p.seqused, p.window, p.min_tiles);
  else
    sda_combine_kernel<128><<<cgrid, 128, 0, stream>>>(p.part_o, p.part_m, p.part_l, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), p.cu_q, out.stride(0), out.stride(1), Hq, (int)qmax, (int)nseg, p.seqused, p.window, p.min_tiles);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &sda_run, "split-KV verify attention (int8 per-token-head, sm86)"); }
