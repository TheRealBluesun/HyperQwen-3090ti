#!/usr/bin/env bash
# Run :8001 (HyperQwen) under nsys; capture ranges are opened/closed via /start_profile and /stop_profile.
# Usage: prof_a.sh OUTDIR   (env: SPEC, CTX as for run_a.sh)
set -euo pipefail
OUT="$(realpath -m "$1")"; mkdir -p "$OUT"
export EXTRA_ARGS_ADD='--profiler-config {"profiler":"cuda"}'
exec ~/tools/nsight-systems/2025.5.2/target-linux-x64/nsys profile -t cuda,nvtx --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=repeat --trace-fork-before-exec=true \
  --sample=none --cpuctxsw=none --force-overwrite=true -o "$OUT/fn" ~/build/HyperQwen/local-3090ti/deploy/run.sh
