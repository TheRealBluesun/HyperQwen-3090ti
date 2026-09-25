// thin pybind entry to marlin::marlin_mm (bf16 activations, uint4b8 weights, group scales, no act-order/zp)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "core/scalar_type.hpp"
namespace marlin {
void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias, void* a_s, void* b_s, void* g_s, void* zp,
               void* g_idx, void* perm, void* a_tmp, int prob_m, int prob_n, int prob_k, int lda, void* workspace,
               vllm::ScalarType const& a_type, vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
               vllm::ScalarType const& s_type, bool has_bias, bool has_act_order, bool is_k_full, bool has_zp, int num_groups,
               int group_size, int dev, cudaStream_t stream, int thread_k_init, int thread_n_init, int sms, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float);
}
void run(torch::Tensor a, torch::Tensor b_q_weight, torch::Tensor b_scales, torch::Tensor workspace, torch::Tensor c,
         torch::Tensor c_tmp, int64_t size_m, int64_t size_n, int64_t size_k, int64_t thread_k, int64_t thread_n, int64_t sms,
         bool use_fp32_reduce) {
  auto empty = torch::empty({0}, a.options());
  auto emptyf = torch::empty({0}, a.options().dtype(torch::kFloat));
  int num_groups = b_scales.size(0);
  int group_size = num_groups > 1 ? size_k / num_groups : -1;
  marlin::marlin_mm(a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(), c_tmp.data_ptr(), empty.data_ptr(), emptyf.data_ptr(),
                    b_scales.data_ptr(), emptyf.data_ptr(), empty.data_ptr(), empty.data_ptr(), empty.data_ptr(), empty.data_ptr(),
                    size_m, size_n, size_k, a.stride(0), workspace.data_ptr(), vllm::kBFloat16, vllm::kU4B8, vllm::kBFloat16,
                    vllm::kBFloat16, false, false, true, false, num_groups, group_size, a.get_device(),
                    at::cuda::getCurrentCUDAStream(), (int)thread_k, (int)thread_n, (int)sms, false, use_fp32_reduce, false);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
