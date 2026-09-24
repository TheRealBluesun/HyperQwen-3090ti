#!/usr/bin/env bash
# HyperQwen single-user on GPU0 as the :8001 endpoint (replaces buun-27b-a; rollback:
# systemctl --user disable --now hyperqwen-a && systemctl --user enable --now buun-27b-a).
# CTX=long: with dflash2 = 128K context, int8 KV (Triton). SPEC=mtp CTX=long = 150K fp8 KV (needs the venv nvcc pinned to 13.0).
# SPEC=dflash2: DFlash2 block drafter, 7 drafts (int8 KV, 128K). Kernel patches: ~/qwen27-dev/patches (01 verify attn, 02 prefill attn).
export CUDA_VISIBLE_DEVICES=0 PORT=${PORT:-8001} HOST=${HOST:-$(ip -4 route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')} CTX=${CTX:-long} SPEC=${SPEC:-dflash2}
# Streamed weight loading (host has ~31 GB RAM and :8002 holds ~11 GB of it). Served under
# buun's name too, with thinking off by default like buun (clients can enable it per request).
export EXTRA_ARGS='--load-format=runai_streamer --model-loader-extra-config={"memory_limit":2542796800} --served-model-name Qwen3.8-27B qwen3.8-27b --default-chat-template-kwargs {"enable_thinking":false} '"${EXTRA_ARGS_ADD:-}"
cd ~/build/HyperQwen && exec bash single-user/start_qwen.sh "$@"
