#!/usr/bin/env bash
# SIGTERM only the vllm serve process inside unit hq-prof, wait for nsys to finish writing.
CG=$(systemctl --user show -p ControlGroup --value hq-prof)
for p in $(cat /sys/fs/cgroup$CG/cgroup.procs); do
  tr "\0" " " < /proc/$p/cmdline 2>/dev/null | grep -q "bin/vllm serve" && { echo "stopping vllm pid $p ($(tr '\0' ' ' < /proc/$p/cmdline | grep -o -- '--port [0-9]*'))"; kill -TERM $p; break; }
done
for i in $(seq 1 120); do systemctl --user is-active -q hq-prof || break; sleep 5; done
echo "unit: $(systemctl --user is-active hq-prof)"
