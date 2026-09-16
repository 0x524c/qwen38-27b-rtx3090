#!/usr/bin/env bash
# Serve one of the checkpoints this box has measured, one at a time.
#
#   qwen38 start [profile]     # default: fast
#   qwen38 stop
#   qwen38 restart [profile]
#   qwen38 status              # what is running, and what it came up with
#   qwen38 list                # the profiles, the model ids they serve, and their cost
#   qwen38 logs [-f]
#
#   --id NAME     serve under this name instead of the profile's own (the generic
#                 qwen3.8-27b alias is served either way)
#   --model DIR   serve this checkpoint directory instead of the profile's default
#   --force       restart even when the requested profile is the one already running
#
# `bash tools/serve.sh fast` still works (a bare profile name means start), and the
# repo's own labels are aliases: single, single-user (= fast) and w4a16, batch (= wide).
#
# Profiles:
#   fast     W4A16 + DFlash2 speculation    262144 ctx, pool 569,878 (2.17x)  109.5 tok/s
#   wide     W4A16, no speculation          262144 ctx, pool 807,951 (3.08x)   75.2 tok/s
#   quality  INT6-Mixed + its own MTP head  262144 ctx, pool 333,637 (1.27x)   97.8 tok/s
#   nvfp4    QUASAR NVFP4                   262144 ctx, pool 397,128 (1.51x)   ~55 tok/s
#
# The two modes are named for the win each one buys, not for the repo's own labels
# (`single-user` and `batch`), which describe the workload they were tuned for on the
# 24 GiB card they were written for. That reads wrong on a 32 GiB one: `fast` is the
# "single-user" profile and still serves 8 concurrent sequences here.
#
# Which to run, in one line each:
#   fast     1-8 sessions, short or moderate context: 2.4x the decode rate of batch
#   wide     fan-out, or 4+ sessions carrying long context: at 4x60k it is 2.7x faster
#   quality  one heavy task where quality beats latency: +3.6% PPL, uses the whole card
#   nvfp4    only for its long-context prefill (+58%); loses on quality, speed and pool
#
# Each checkpoint is served under its own model name as well as the generic
# qwen3.8-27b alias. That is what makes the choice in a harness honest: the loaded
# server answers only its own name, so picking the other one fails with "model does
# not exist" instead of quietly answering from the checkpoint you did not select.
#
# What each one costs, measured on this box (2026-09-15, 500 W, KVarN, 262144):
#
#   fast   W4A16 + DFlash2  PPL 8.1928  GSM8K 96.0%  109.5 tok/s  569,878 (2.17x)
#   wide   W4A16, no spec   PPL 8.1847  GSM8K 95.5%   75.2 tok/s  807,951 (3.08x)
#   quality INT6 + MTP      PPL 7.8915  GSM8K 95.5%   97.8 tok/s  333,637 (1.27x)
#   nvfp4  QUASAR NVFP4     PPL 8.2854  GSM8K 94.0%   ~55  tok/s  397,128 (1.51x)
#
# The nvfp4 flags are not optional, and are why this wrapper exists:
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

# readlink -f first: this file is installed as a symlink in ~/bin, and BASH_SOURCE then
# points at the link, not at the repo copy.
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
cd "$REPO"

NVFP4_DIR="/app/models/Qwen3.8-27B-QUASAR-NVFP4"
INT6_DIR="/app/models/Qwen3.8-27B-INT6-Mixed-AutoRound"
W4A16_DIR="/app/models/Qwen3.8-27B-W4A16-AutoRound"
# The humming kernels (5/6/7-bit weights, which is what the INT6 checkpoint is made of)
# JIT-compile through a native helper that loads libnvrtc.so.13 by path; libnvrtc then
# opens libnvrtc-builtins.so.13.0 BY NAME, so that directory has to be on the loader
# path or the boot dies with "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE".
VLLM_LDP="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/app/venv/lib/python3.12/site-packages/nvidia/cu13/lib"
PORT="${PORT:-18020}"

usage() { sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2; }

# Status has to work for whichever profile is up, and the two live in different
# compose services with different logs, so the service is read from what is running.
running_service() { docker compose ps --services --status running 2>/dev/null | head -1; }
pool_line() {
  local svc; svc=$(running_service)
  [ -n "$svc" ] || svc=$SERVICE
  docker compose logs "$svc" 2>&1 | grep -m1 "GPU KV cache size" || true
}
served_ids() {
  local key; key=$(bash "$REPO/tools/print-api-key.sh" 2>/dev/null) || return 0
  curl -s --max-time 8 -H "Authorization: Bearer $key" "http://127.0.0.1:$PORT/v1/models" \
    | python3 -c 'import json,sys; print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null || true
}

CMD=""; PROFILE=""; ID_OVERRIDE=""; MODEL_OVERRIDE=""; FOLLOW=""; FORCE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --id)        ID_OVERRIDE=${2:-}; shift 2 ;;
    --model)     MODEL_OVERRIDE=${2:-}; shift 2 ;;
    -f|--follow) FOLLOW=1; shift ;;
    --force)     FORCE=1; shift ;;
    -h|--help)   usage ;;
    -*)          echo "serve: unknown flag $1" >&2; usage ;;
    *)  if [ -z "$CMD" ]; then CMD=$1
        elif [ -z "$PROFILE" ]; then PROFILE=$1
        else echo "serve: too many arguments (extra: $1)" >&2; usage; fi
        shift ;;
  esac
done
# A bare profile name means start: `qwen38 fast` and `bash tools/serve.sh fast` both work.
case "${CMD:-}" in
  fast|single|single-user|wide|w4a16|batch|quality|int6|nvfp4)
    [ -z "$PROFILE" ] && { PROFILE=$CMD; CMD=start; } ;;
esac
[ -z "$CMD" ] && usage

SERVICE=batch
PRIMARY_ID=""; PROFILE_MODEL="";
case "$PROFILE" in
  fast|single|single-user)
    SERVICE=single
    PRIMARY_ID=qwen3.8-27b-w4a16-vllm
    # Pinned on purpose. The single-user launcher prefers models/...-fast (int4-GPTQ
    # lm_head + MTP) whenever MODEL arrives empty, and that variant costs 0.6% of
    # perplexity against the base (8.2424 against 8.1928): leaving this empty would
    # silently serve the worse checkpoint under the fast profile's name.
    PROFILE_MODEL="$W4A16_DIR"
    FLAGS=(--mode single --ctx 262144
           --extra-args "--chat-template /app/models/qwen3.8-hybrid.jinja") ;;
  wide|w4a16|batch)
    SERVICE=batch
    PRIMARY_ID=qwen3.8-27b-w4a16-vllm
    PROFILE_MODEL="$W4A16_DIR"     # same trap as fast: pin what this profile means
    FLAGS=(--mode batch --kv kvarn) ;;
  quality|int6)
    SERVICE=single
    PRIMARY_ID=qwen3.8-27b-int6-quality-vllm
    PROFILE_MODEL="$INT6_DIR"
    [ -d "$REPO/models/Qwen3.8-27B-INT6-Mixed-AutoRound" ] || {
      echo "serve: models/Qwen3.8-27B-INT6-Mixed-AutoRound is not on disk (25.4 GB, not committed)." >&2
      echo "       /app/venv/bin/hf download Minachist/Qwen3.8-27B-INT6-Mixed-AutoRound --local-dir $INT6_DIR" >&2
      exit 1; }
    # 262144 at 0.98 utilization is the whole card: 22.88 GiB of weights + 4 GiB pinned pool
    # + the profiled peak leave ~350 MiB free. It is what buys context parity with fast on a
    # checkpoint whose weights are 7 GiB heavier, and it means nothing else fits on the GPU
    # while this profile is up. Drop --gpu-util-exact and --ctx to 196608 for headroom.
    FLAGS=(--mode single --spec mtp --kv huge --ctx 262144 --kv-mem-gib 4 --gpu-util-exact 0.98
           --no-vision --skip-verify --env "LD_LIBRARY_PATH=$VLLM_LDP"
           --extra-args "--chat-template /app/models/qwen3.8-hybrid.jinja") ;;
  nvfp4)
    SERVICE=batch
    PRIMARY_ID=qwen3.8-27b-nvfp4-vllm
    PROFILE_MODEL="$NVFP4_DIR"
    [ -d "$REPO/models/Qwen3.8-27B-QUASAR-NVFP4" ] || {
      echo "serve: models/Qwen3.8-27B-QUASAR-NVFP4 is not on disk (20.6 GB, not committed)." >&2
      echo "       /app/venv/bin/hf download QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 --local-dir $NVFP4_DIR" >&2
      exit 1; }
    FLAGS=(--mode batch --kv kvarn --gpu-util 0.85 --skip-verify
           --env VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=fp4_gemm) ;;
  "") : ;;   # stop/status/list/logs need no profile
  *) echo "serve: unknown profile '$PROFILE' (fast, wide, quality, nvfp4)" >&2; exit 2 ;;
esac

case "$CMD" in
  status)
    docker compose ps --format '{{.Service}} {{.Status}}' 2>/dev/null | head -3
    p=$(pool_line)
    [ -n "$p" ] && printf '%s\n' "$p" || echo "no pool line: nothing is serving"
    ids=$(served_ids)
    [ -n "$ids" ] && echo "ids: $ids"
    exit 0 ;;
  list)
    printf '%-9s %-32s %-9s %s\n' PROFILE MODEL_ID "GPU" "MEASURED (500 W)"
    printf '%-9s %-32s %-9s %s\n' ------- -------------------------------- --------- -----------------
    printf '%-9s %-32s %-9s %s\n' fast    qwen3.8-27b-w4a16-vllm          single    "PPL 8.1928  GSM8K 96.0%  109.5 tok/s  pool 569,878 (2.17x)  262144 ctx"
    printf '%-9s %-32s %-9s %s\n' wide    qwen3.8-27b-w4a16-vllm          batch     "PPL 8.1847  GSM8K 95.5%   75.2 tok/s  pool 807,951 (3.08x)  262144 ctx"
    printf '%-9s %-32s %-9s %s\n' quality qwen3.8-27b-int6-quality-vllm   single    "PPL 7.8915  GSM8K 95.5%   97.8 tok/s  pool 333,637 (1.27x)  262144 ctx"
    printf '%-9s %-32s %-9s %s\n' nvfp4   qwen3.8-27b-nvfp4-vllm          batch     "PPL 8.2854  GSM8K 94.0%   55.3 tok/s  pool 397,128 (1.51x)  262144 ctx"
    echo
    echo "aliases: single, single-user -> fast   ·   w4a16, batch -> wide   ·   int6 -> quality"
    now=$(running_service)
    [ -n "$now" ] && echo "running: $now  (ids: $(served_ids))" || echo "running: nothing"
    exit 0 ;;
  logs)
    svc=$(running_service); [ -n "$svc" ] || { echo "serve: nothing is running" >&2; exit 1; }
    if [ -n "$FOLLOW" ]; then docker compose logs -f "$svc"; else docker compose logs --tail 40 "$svc"; fi
    exit 0 ;;
  stop)
    # Starting what is already running is a no-op: the restart costs six minutes of loading
# and ~30 s of downtime, and the only reason to want it anyway is a config change that the
# profile file no longer reflects, which --force covers. The signature is read from the
# container's own environment (compose passes env_file entries in as process environment),
# so this never touches the profile file.
container_env() { docker compose exec -T "$1" printenv "$2" 2>/dev/null || true; }
served_name() { printf '%s' "$1" | sed -n 's/.*--served-model-name \([^ ]*\).*/\1/p'; }
if [ -z "$FORCE" ]; then
  run_svc=$(running_service)
  if [ -n "$run_svc" ]; then
    run_model=$(container_env "$run_svc" MODEL)
    run_id=$(served_name "$(container_env "$run_svc" EXTRA_ARGS)")
    if [ "$run_svc" = "$SERVICE" ] && [ "$run_model" = "$EFFECTIVE_MODEL" ] && [ "$run_id" = "$EFFECTIVE_ID" ]; then
      echo "== already serving $EFFECTIVE_ID from $run_model on $run_svc: nothing to do (--force restarts it)"
      exit 0
    fi
  fi
fi

echo "== stopping whatever of this project is running"
    docker compose --profile batch --profile single down --remove-orphans 2>&1 | sed 's/^/   /'
    exit 0 ;;
  start|restart) ;;
  *) usage ;;
esac

EFFECTIVE_ID=${ID_OVERRIDE:-$PRIMARY_ID}
FLAGS+=(--extra-args "--served-model-name $EFFECTIVE_ID qwen3.8-27b")
EFFECTIVE_MODEL=${MODEL_OVERRIDE:-$PROFILE_MODEL}
[ -n "$EFFECTIVE_MODEL" ] && FLAGS+=(--model "$EFFECTIVE_MODEL")

echo "== stopping whatever of this project is running"
docker compose --profile batch --profile single down --remove-orphans >/dev/null 2>&1 || true

echo "== writing the $PROFILE profile"
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
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/health" || true)
  [ "$code" = "200" ] && break
  sleep 5
done
echo "== health: ${code:-no answer}"
if [ "${code:-}" != "200" ]; then
  echo "== it did not come up; the last lines say why:"
  docker compose logs --tail 15 "$SERVICE" 2>&1 | grep -viE "^\s*$" | tail -8
  exit 1
fi
echo "== served as: $EFFECTIVE_ID, qwen3.8-27b"

case "$PROFILE" in
  nvfp4) cat <<'EOF'
== serving nvfp4 (QUASAR NVFP4): expect PPL 8.2854, C1 ~55 tok/s, and 397,128 tokens of
   pool (1.51x at 262144). Its only win is prefill at very long context (+58%); it loses
   on quality, speed and pool against every other profile here.
EOF
  ;;
  fast|single|single-user) cat <<'EOF'
== serving fast (W4A16 + DFlash2, CTX=huge, KVarN): expect PPL 8.1928, 210 tok/s at C1
   on real prompts, 109.5 on the six-task agent mix, and 569,878 tokens of pool (2.17x at
   262144). Speculation accepts ~3.4 tokens per verify step. Eight concurrent sequences;
   past a sum of ~570k live tokens the pool binds before the slots do.
EOF
  ;;
  wide|w4a16|batch) cat <<'EOF'
== serving wide (W4A16 + int8 heads, no speculation): expect PPL 8.1847, 84.5 tok/s at
   C1, 75.2 on the agent mix, 2,188 tok/s aggregate at C64, and 807,951 tokens of pool
   (3.08x at 262144). 64 slots: the profile for fan-out, and the one that wins at 4
   sessions carrying long context (70.3 against fast's 25.9 tok/s per stream at 4x60k).
EOF
  ;;
  quality|int6) cat <<'EOF'
== serving quality (INT6-Mixed + its own INT5 MTP head, KVarN, 262144): expect PPL 7.8915
   against fast's 8.1928 (-3.6%, code -7.0%) and 97.8 tok/s on the six-task agent mix
   against 109.5 for fast. Pool 333,637 tokens (1.27x) at 0.98 utilization, so ~5 sessions
   at 60k, not a swarm, and nothing else fits on the card. Vision off and verify skipped:
   the checkpoint's lm_head/embed are 16-bit by design, so verify.sh's requantization
   checks do not apply.
EOF
  ;;
esac
