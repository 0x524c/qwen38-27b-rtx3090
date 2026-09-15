#!/usr/bin/env bash
# Serve one of the two checkpoints this box has measured, one at a time.
#
#   bash tools/serve.sh w4a16     # W4A16 + int8 heads: the better quality of the two
#   bash tools/serve.sh nvfp4     # QUASAR NVFP4: needs its own flags, and loses on quality
#   bash tools/serve.sh status    # what is running, and what it came up with
#
# Both are served in batch mode so the comparison is only about the checkpoint: batch
# is the mode that fits several harness sessions at once, and it is the only mode the
# NVFP4 checkpoint has been verified in (single-user mode needs the DFlash2 drafter
# and the MTP/draft-head artifacts that only the W4A16 checkpoint has).
#
# Each checkpoint is served under its own model name as well as the generic
# qwen3.8-27b alias. That is what makes the choice in a harness honest: the loaded
# server answers only its own name, so picking the other one fails with "model does
# not exist" instead of quietly answering from the checkpoint you did not select.
#
# What each one costs, measured on this box (2026-09-15, 500 W, KVarN, 262144):
#
#   W4A16   PPL 8.1840  GSM8K 94.5-95.5%  C1 85.9  pool 796,213 (3.04x)  util 0.92
#   NVFP4   PPL 8.2854  GSM8K 94.0%       C1 55.3  pool 397,128 (1.51x)  util 0.85
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

pool_line() { docker compose logs batch 2>&1 | grep -m1 "GPU KV cache size" || true; }

case "${1:-}" in
  status)
    docker compose ps --format '{{.Service}} {{.Status}}' 2>/dev/null | head -3
    p=$(pool_line)
    [ -n "$p" ] && printf '%s\n' "$p" || echo "no pool line: nothing is serving"
    exit 0 ;;
  w4a16)
    FLAGS=(--mode batch --kv kvarn
           --extra-args "--served-model-name qwen3.8-27b-w4a16-vllm qwen3.8-27b") ;;
  nvfp4)
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

echo "== starting batch"
docker compose --profile batch up -d 2>&1 | tail -1

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
  docker compose logs --tail 15 batch 2>&1 | grep -viE "^\s*$" | tail -8
  exit 1
fi

if [ "$1" = nvfp4 ]; then
  cat <<'EOF'
== serving NVFP4: expect PPL 8.2854, C1 ~55 tok/s, and 397,128 tokens of pool
   (1.51x at 262144, so a second full-length request does not fit).
EOF
else
  cat <<'EOF'
== serving W4A16 + int8 heads: expect PPL 8.1840, C1 ~86 tok/s, and 796,213 tokens
   of pool (3.04x at 262144). This is the better-quality of the two, and this batch
   mode is its slower mode: single-user (SPEC=dflash2, CTX=huge) reads 213 tok/s at
   C1 with 569,878 tokens of pool — `bash tools/gpu-profile.sh --mode single` and
   `docker compose --profile single up -d`.
EOF
fi
