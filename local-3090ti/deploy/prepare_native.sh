#!/usr/bin/env bash
set -euo pipefail
cd ~/build/HyperQwen
M=models/Qwen3.8-27B-W4A16-AutoRound
venv/bin/python prepare/quant_lm_head.py $M
venv/bin/python prepare/quant_embed.py   $M
venv/bin/python prepare/quant_mtp.py     $M
venv/bin/python prepare/build_draft_vocab.py $M --ids prepare/draft_vocab_ids.json
venv/bin/python prepare/fetch_fast_variant.py
venv/bin/python prepare/fetch_dflash2.py
VIRTUAL_ENV=$PWD/venv ~/.local/bin/uv pip install runai-model-streamer humanize
echo "== prepare done"
sed -e "s/#.*//" -e "s/^[[:space:]]*//;s/[[:space:]]*$//" -e "/^$/d" patches/series |
while IFS= read -r name; do
  case "$name" in dflash2-backport.patch) echo "skip $name"; continue ;; esac
  patch -p1 -N -d venv/lib/python3.12/site-packages/vllm < "patches/$name" > /dev/null || echo "PATCH FAILED: $name"
done
echo "== patches done"
bash verify.sh --no-server 2>&1 | tail -25
echo "== verify done"
