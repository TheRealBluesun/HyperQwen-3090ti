# Marlin CTA-count table for M<=8 (patch 08)

At M=8 stock Marlin runs one CTA per SM and splits K across CTAs, so neighbouring CTAs share output
columns and pass partial sums through lock-ordered global reductions. For some shapes, fewer CTAs that
each own whole column tiles are faster even with SMs idle. This builds vLLM 0.29.0's own Marlin
(bf16 x uint4b8, group 128, M<=16 instantiations only; output identical up to the split-K summation
order) with the `sms` argument exposed, and routes the tabled shapes through it via HyperQwen's
`VLLM_MARLIN_TUNE=1` hook. Every other call goes to the stock `_C::marlin_gemm`.

Build (JIT on first use; the prep step once):

    curl -sL -o vllm-0.29.0.tar.gz https://github.com/vllm-project/vllm/archive/refs/tags/v0.29.0.tar.gz
    tar xzf vllm-0.29.0.tar.gz
    cp -r vllm-0.29.0/csrc/libtorch_stable/quantization/marlin gen && (cd gen && python generate_kernels.py 8.6)
    python mt_prep.py            # -> mt/: host.cu, kernels.cu, kernel_selector.h (+ headers)
    # install next to the vllm package (e.g. the dev PYTHONPATH overlay):
    #   marlin_tune_ext.py, marlin_best.py, marlin_tune_src/{mt/*, binding.cu, csrc/core/scalar_type.hpp}

The table (marlin_best.py) is (thread_k, thread_n, CTA count) per (N, K), from a sweep of the three
small-M tile configs x CTA counts 24..82 at M=8 (mt_table.py / the cfg sweep). RTX 3090 @ 350 W, per call:
gate_up 118.6 -> 113.3 us, GDN in_proj 57.3 -> 55.3, o_proj 24.6 -> 23.0, attn qkv 51.5 -> 50.1,
down 59.1 -> 58.1, drafter qkv 24.9 -> 23.3, drafter o_proj 18.3 -> 16.6. End to end: 24.13 -> 23.54
ms/step (memory-service replay), chat 23.73 -> 23.16 ms/step.
