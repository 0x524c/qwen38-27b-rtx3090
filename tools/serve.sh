#!/usr/bin/env bash
# Serve the profiles this box has measured, one at a time.
#
#   bash tools/serve.sh fast      # the default: W4A16 + DFlash2 speculation
#   bash tools/serve.sh wide      # W4A16, no speculation: for many sessions at once
#   bash tools/serve.sh nvfp4     # QUASAR NVFP4: needs its own flags, loses on quality
#   bash tools/serve.sh status    # what is running, and what it came up with
#
# `single`, `single-user`, `w4a16` and `batch` are accepted as aliases of fast/wide.
#
# The two modes are named for the win each one buys, not for the repo's own labels
# (`single-user` and `batch`), which describe the workload they were tuned for on the
# 24 GiB card they were written for. That reads wrong on a 32 GiB one: `fast` is the
# "single-user" profile and still serves 8 concurrent sequences here.
#
# `fast` is what this box settled on (2026-09-15, ordered by quality, speed, context,
# concurrency for a handful of coding-agent sessions): it reaches the same 262144
# context and costs 0.10% perplexity (8.1928 against 8.1847) for 2.4x the decode rate.
# The pool is the one axis `wide` keeps: four sessions would each need under ~140k
# tokens to all sit at full length under `fast`.
#
# The NVFP4 checkpoint has only ever been verified in batch mode (single-user needs
# the DFlash2 drafter and the MTP/draft-head artifacts that only the W4A16
# checkpoint has).
#
# Each checkpoint is served under its own model name as well as the generic
# qwen3.8-27b alias. That is what makes the choice in a harness honest: the loaded
# server answers only its own name, so picking the other one fails with "model does
# not exist" instead of quietly answering from the checkpoint you did not select.
#
# What each one costs, measured on this box (2026-09-15, 400 W, KVarN, 262144):
#
#   fast   W4A16 + DFlash2  PPL 8.1928  GSM8K 96.0%  C1 198 tok/s  pool 569,878 (2.17x)
#   wide   W4A16, no spec   PPL 8.1847  GSM8K 95.5%  C1  83 tok/s  pool 807,951 (3.08x)
#   nvfp4  QUASAR NVFP4     PPL 8.2854  GSM8K 94.0%  C1  55 tok/s  pool 397,128 (1.51x)
#
# The NVFP4 flags are not optional and are why this wrapper exists:
#   --gpu-util 0.85    its weights take 17.63 GiB on the GPU against ~16 for W4A16, and
#                      at 0.92 the boot dies in CUDA OOM; 0.85 is the highest that boots
#   --skip-verify      it has not been through prepare/, so verify.sh fails it on the
#                      requantized lm_head/embed — a reason to skip the gate, not to
#                      refuse the model
#   AUTOTUNE_SKIP_OPS  FlashInfer tunes 16 fp4_gemm tactics at boot, OOMs through them
#                      and falls back to the default tactic anyway: 13 minutes of boot
#                      for nothing
#   streamer           written by the profile generator on its own: the stock loader
#                      took this machine's 91 GiB plus its 45 GiB of swap on this
#                      checkpoint
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

NVFP4_DIR="/app/models/Qwen3.8-27B-QUASAR-NVFP4"

usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

SERVICE=batch
# Status has to work for whichever profile is up, and the two live in different
# compose services with different logs, so the service is read from what is running.
pool_line() {
  local svc
  svc=$(docker compose ps --services --status running 2>/dev/null | head -1)
  [ -n "$svc" ] || svc=$SERVICE
  docker compose logs "$svc" 2>&1 | grep -m1 "GPU KV cache size" || true
}

case "${1:-}" in
  status)
    docker compose ps --format '{{.Service}} {{.Status}}' 2>/dev/null | head -3
    p=$(pool_line)
    [ -n "$p" ] && printf '%s\n' "$p" || echo "no pool line: nothing is serving"
    exit 0 ;;
  fast|single|single-user)
    SERVICE=single
    FLAGS=(--mode single --ctx 262144 --model /app/models/Qwen3.8-27B-W4A16-AutoRound
           --extra-args "--served-model-name qwen3.8-27b-w4a16-vllm qwen3.8-27b"
           --extra-args "--chat-template /app/models/qwen3.8-hybrid.jinja") ;;
  wide|w4a16|batch)
    SERVICE=batch
    FLAGS=(--mode batch --kv kvarn
           --extra-args "--served-model-name qwen3.8-27b-w4a16-vllm qwen3.8-27b") ;;
  nvfp4)
    SERVICE=batch
    [ -d "$REPO/models/Qwen3.8-27B-QUASAR-NVFP4" ] || {
      echo "serve: models/Qwen3.8-27B-QUASAR-NVFP4 is not on disk (20.6 GB, not committed)." >&2
      echo "       /app/venv/bin/hf download QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 --local-dir $NVFP4_DIR" >&2
      exit 1; }
    FLAGS=(--mode batch --kv kvarn --gpu-util 0.85 --model "$NVFP4_DIR" --skip-verify
           --extra-args "--served-model-name qwen3.8-27b-nvfp4-vllm qwen3.8-27b"
           --env VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=fp4_gemm) ;;
  *) usage ;;
esac

echo "== stopping whatever of this project is running"
docker compose --profile batch --profile single down --remove-orphans >/dev/null 2>&1 || true

echo "== writing the $1 profile"
bash tools/gpu-profile.sh "${FLAGS[@]}" 2>&1 | grep -vE "^\s*$" | tail -14

echo "== starting $SERVICE"
docker compose --profile "$SERVICE" up -d 2>&1 | tail -1

# Two waits, not one: the pool line says the KV cache was sized, and then CUDA graph
# capture still has to run before the port answers. Checking health right after the
# pool line reports a working server as dead.
echo "== waiting for the pool (up to ~8 min)"
pool=""
for _ in $(seq 1 96); do
  p=$(pool_line)
  if [ -n "$p" ]; then pool="$p"; break; fi
  sleep 5
done
if [ -n "$pool" ]; then printf '%s\n' "${pool##*| }"; else echo "== no pool line after 8 min"; fi

echo "== waiting for health (graph capture follows the pool line)"
code=""
for _ in $(seq 1 96); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:18020/health || true)
  [ "$code" = "200" ] && break
  sleep 5
done
echo "== health: ${code:-no answer}"
if [ "${code:-}" != "200" ]; then
  echo "== it did not come up; the last lines say why:"
  docker compose logs --tail 15 "$SERVICE" 2>&1 | grep -viE "^\s*$" | tail -8
  exit 1
fi

case "$1" in
  nvfp4) cat <<'EOF'
== serving NVFP4: expect PPL 8.2854, C1 ~55 tok/s, and 397,128 tokens of pool
   (1.51x at 262144, so a second full-length request does not fit).
EOF
  ;;
  fast|single|single-user) cat <<'EOF'
== serving fast (W4A16 + DFlash2, CTX=huge, KVarN): expect PPL 8.1928, 198 tok/s at C1,
   163 at C4, 92 at C8, and 569,878 tokens of pool (2.17x at 262144). Speculation
   accepts ~3.4 tokens per verify step at C1 and still 3.0 at C4, so the rate holds
   across the 2-4 sessions this profile is sized for.
EOF
  ;;
  wide|w4a16|batch) cat <<'EOF'
== serving wide (W4A16 + int8 heads, no speculation): expect PPL 8.1847, 83 tok/s at
   C1, ~72 at C4, 76 at C8, 2,188 tok/s at C64, and 807,951 tokens of pool (3.08x at
   262144). Better quality by 0.10% and twice the pool; 2.4x slower at the concurrency
   a handful of sessions actually use. Built for fan-out, not for a few sessions.
EOF
  ;;
esac
