#!/usr/bin/env bash
# (Re)start the :8002 dev server as transient unit hq-dev, using the dev copy of vllm (PYTHONPATH overlay).
# Usage: dev_up.sh [VAR=VALUE ...]   extra env for the launcher (e.g. DFLASH_TOKENS=11). Waits until ready.
set -u
systemctl --user stop hyperqwen-b hq-dev 2>/dev/null; systemctl --user reset-failed hq-dev 2>/dev/null
args=(--user --unit=hq-dev --setenv=GPU=1 --setenv=PORT=8002 --setenv=KV_MEM=${KV_MEM:-5398908108}
      --setenv=PYTHONPATH=$HOME/build/HyperQwen/dev/site)
for kv in "$@"; do args+=(--setenv="$kv"); done
systemd-run "${args[@]}" $HOME/build/HyperQwen/local-3090ti/deploy/run.sh >/dev/null
for i in $(seq 1 90); do
  curl -sf http://$(hostname -I | awk '{print $1}'):8002/v1/models >/dev/null && { echo READY; exit 0; }
  systemctl --user is-active -q hq-dev || { echo FAILED; journalctl --user -u hq-dev --no-pager | grep -E "Error|error" | tail -5 | cut -c1-300; exit 1; }
  sleep 5
done
echo TIMEOUT; exit 1
