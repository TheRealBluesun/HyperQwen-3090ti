# HyperQwen on an RTX 3090 Ti (300 W), Qwen3.8-27B, :8001

Local deployment and optimization work on top of upstream HyperQwen (syv-ai/HyperQwen).
Requirement: at least 128K context in every configuration.

- NOTES.md: setup, baseline comparison (vs buun-llama-cpp EXL3), traffic analysis, profile findings, gotchas.
- PROGRESS.md: every measured configuration, in order.
- deploy/: run_a.sh (launcher; SPEC=dflash2 CTX=long by default), the systemd user unit, the native
  install/prepare scripts, prof_a.sh (run under nsys with capture ranges).
- patches/: kernel changes on top of the HyperQwen-patched vLLM 0.29.0 (apply with patches/apply.sh
  after the upstream patch series). 01 = split-KV verify attention tuning, 02 = sm86 prefill-attention
  tuning. rejected-* were measured and not adopted.
- tools/: bench27.py (engine-neutral streamed benchmark; LONG=1 adds a ~109K test), capture27.py (nsys
  capture driver), bench_sda.py / bench_ua.py (kernel microbenchmarks).
- results/: raw bench27.py runs.
