#!/usr/bin/env bash
# Launch the :8002 dev server under nsys (capture ranges via /start_profile) as transient unit hq-prof.
# Usage: dev_prof.sh OUTDIR [VAR=VALUE ...]. Stop with: kill -TERM <vllm serve pid in the unit's cgroup> (nsys then writes the report).
set -u
OUT=$(realpath -m "$1"); shift; mkdir -p "$OUT"
systemctl --user stop hyperqwen-b hq-dev hq-prof 2>/dev/null; systemctl --user reset-failed hq-prof 2>/dev/null
args=(--user --unit=hq-prof --setenv=GPU=1 --setenv=PORT=8002 --setenv=KV_MEM=${KV_MEM:-5398908108}
      --setenv=PYTHONPATH=$HOME/build/HyperQwen/dev/site "--setenv=EXTRA_ARGS_ADD=--profiler-config.profiler=cuda")
for kv in "$@"; do args+=(--setenv="$kv"); done
systemd-run "${args[@]}" $HOME/tools/nsight-systems/2025.5.2/target-linux-x64/nsys profile -t cuda,nvtx --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=repeat --trace-fork-before-exec=true --sample=none --cpuctxsw=none \
  --force-overwrite=true -o "$OUT/fn" $HOME/build/HyperQwen/local-3090ti/deploy/run.sh >/dev/null
for i in $(seq 1 120); do
  curl -sf http://$(hostname -I | awk '{print $1}'):8002/v1/models >/dev/null && { echo READY; exit 0; }
  systemctl --user is-active -q hq-prof || { echo FAILED; exit 1; }
  sleep 5
done
echo TIMEOUT; exit 1
