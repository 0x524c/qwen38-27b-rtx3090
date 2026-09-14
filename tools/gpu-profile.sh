#!/usr/bin/env bash
# Hardware profile generator: writes the .env the compose file reads, sized for
# the GPU actually installed.
#
# Why this exists. The shipped defaults in single-user/start_qwen.sh and
# batch/start_qwen.sh are calibrated for the reference card every number in the
# README was measured on: a 24 GiB RTX 3090 (sm86) at 250 W, alone on the box.
# Two of those calibrations are wrong on any other card, and both fail silently:
#
#   - The KV pool is pinned by BYTES (KV_MEM), not by utilization, because the
#     V2 runner's profiled activation peak swings about a gigabyte between
#     starts and pinning is what makes the pool deterministic. The pinned value
#     is a 24 GiB constant, so a larger card strands its extra VRAM: a 32 GiB
#     card keeps the same ~5.2 GiB pool and the same context ceiling.
#   - --gpu-memory-utilization is a fraction of TOTAL memory, not of free
#     memory, so on a card that already holds someone else's allocation (a
#     training run, a compositor, a second server) vLLM asks for memory that is
#     not there and fails the startup free-memory check, at 0.93 as much as at
#     0.98.
#
# Both are env-var knobs, so this script resolves them from nvidia-smi instead
# of editing a launcher. On a 24 GiB card it reproduces the shipped profile.
#
#   bash tools/gpu-profile.sh                  # write .env
#   bash tools/gpu-profile.sh --print          # print, write nothing
#   bash tools/gpu-profile.sh --mode batch     # throughput profile
#   bash tools/gpu-profile.sh --explain        # the numbers and the arithmetic
#   bash tools/gpu-profile.sh --no-key         # leave the API key unset
#   bash tools/gpu-profile.sh --reserve-gib 10 # keep 10 GiB for other workloads
#   bash tools/gpu-profile.sh --table          # reserve vs context, no write
#   bash tools/gpu-profile.sh --max-len 33000  # force a ceiling already boot-tested
#
# It only reads (nvidia-smi, docker inspect). It never starts, stops or restarts
# anything, and it never overwrites an .env it did not write without --force.

set -u

MODE=single
PRINT=0
EXPLAIN=0
FORCE=0
NO_KEY=0
FORCE_MAX_LEN=0
SHOW_TABLE=0
RESERVE_GIB=""                             # empty = derive it from the card
while [ $# -gt 0 ]; do
  case "$1" in
    --print)       PRINT=1 ;;
    --explain)     EXPLAIN=1; SHOW_TABLE=1 ;;
    --table)       SHOW_TABLE=1 ;;
    --force)       FORCE=1 ;;
    --no-key)      NO_KEY=1 ;;
    --mode)        MODE=${2:-}; shift ;;
    --reserve-gib) RESERVE_GIB=${2:-0}; shift ;;
    --max-len)     FORCE_MAX_LEN=${2:-0}; shift ;;
    -h|--help) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
case "$MODE" in single|batch) ;; *) echo "--mode takes single or batch" >&2; exit 2 ;; esac
if [ -n "$RESERVE_GIB" ]; then
  case "$RESERVE_GIB" in ''|*[!0-9]*) echo "--reserve-gib takes a whole number of GiB" >&2; exit 2 ;; esac
fi
case "$FORCE_MAX_LEN" in ''|*[!0-9]*) echo "--max-len takes a token count" >&2; exit 2 ;; esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

# ---- reference card: the constants the shipped profiles were calibrated on ---
# Every README table and every entry in docs/gotchas.md was measured against
# these. They are not tunable here; a 24 GiB card must keep getting them.
REF_VRAM_MIB=24576
REF_PIN_FAST=5583457484          # CTX=fast and CTX=long pinned pool, bytes
REF_PIN_HUGE=5261334938          # CTX=huge (KVarN) pinned pool, bytes
REF_BUDGET_FAST_MIB=22856        # 0.93 x 24576, the single-user default
REF_MARGIN_MIB=1126              # what the shipped pin leaves for transients
# Pool bytes per token, measured rather than derived: these are the token counts
# the two shipped 24 GiB profiles actually report for the pool they pin, with
# prefix caching on (the state pages and the pool's block granularity are inside
# the number, which is why it is not simply the 78 KB/token the group fix
# documents).
#   CTX=fast  bf16  5,583,457,484 B -> 57,669 tokens
#   CTX=long  fp8   5,583,457,484 B -> 136,429 tokens
#   CTX=huge  KVarN 5,261,334,938 B -> 268,169 tokens
# The effective cost per token rises a little with pool size, so extrapolating
# upward on these under-predicts the token count, which is the safe direction.
BPT_BF16=96818
BPT_FP8=40925
BPT_KVARN=19620
# A ceiling is only accepted if the tokens fit in this much of the pool. The
# model above is fitted at ~5 GiB; this is what pays for the extrapolation.
POOL_USE_PCT=98
# Weights + drafter + CUDA graphs + the profiled activation peak, i.e. the
# reference budget minus the pool and the margin it left. Constant across cards
# for a given profile: none of those four terms grows with the card.
NON_KV_MIB=$(( REF_BUDGET_FAST_MIB - REF_PIN_FAST / 1048576 - REF_MARGIN_MIB ))
DESKTOP_MIB=512                  # compositor/X on a desktop card; 0 headless
MARGIN_MIB=1229                  # 1.2 GiB of transient headroom, on every card
# Ceilings the model supports, largest that fits wins. The small end matters:
# on a card shared with a training run the pool can be a few GiB, and a profile
# that refuses to boot is worth less than a shorter context that does.
MODEL_MAX_LEN=262144
CANDIDATES="8192 16384 32768 40960 49152 57344 65536 81920 98304 114688 131072 163840 196608 229376 262144"

# ---- probe ------------------------------------------------------------------
num() { nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -cd '0-9'; }
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^ *//;s/ *$//')
TOTAL=$(num memory.total); FREE=$(num memory.free)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
if [ -z "${TOTAL:-}" ] || [ -z "${FREE:-}" ]; then
  echo "gpu-profile: nvidia-smi gave no total/free VRAM; leaving the shipped" >&2
  echo "             24 GiB defaults alone (nothing written)." >&2
  exit 1
fi
# sm_120 for compute capability 12.0: the tag, not the version.
CC_TAG="${CC%%.*}${CC##*.}"
[ -n "$NAME" ] || NAME="unknown GPU"
OTHER=$(( TOTAL - FREE ))

IMAGE=ghcr.io/syv-ai/qwen38-27b-rtx3090:latest
# The budget has to be computed in the units vLLM checks, and those are not
# nvidia-smi's. On this box nvidia-smi reports 32,607 MiB total while the CUDA
# context vLLM sizes against sees 31.45 GiB, and the container's own context
# takes another ~500 MiB off the free side before vLLM's check runs. vLLM's
# check is literally `free_at_init >= util x total`, so guessing those offsets
# is what makes an otherwise sane profile fail at startup with
# "Free memory on device cuda:0 (28.19/31.45 GiB) is less than desired GPU
# memory utilization". Ask the same runtime instead: the probe below reports
# exactly what the serving container will see.
TOTAL_CUDA=""; FREE_CUDA=""
if command -v docker >/dev/null 2>&1; then
  CUDA_PROBE='venv/bin/python -c "import torch; f, t = torch.cuda.mem_get_info(); print(t//1048576, f//1048576)"'
  read -r TOTAL_CUDA FREE_CUDA <<<"$(timeout 240 docker run --rm --gpus all --entrypoint \
      bash "$IMAGE" -c "$CUDA_PROBE" 2>/dev/null | tail -1)" || true
fi
CUDA_UNITS=1
if ! [ "${TOTAL_CUDA:-}" -gt 0 ] 2>/dev/null || ! [ "${FREE_CUDA:-}" -gt 0 ] 2>/dev/null; then
  # No docker, or the probe failed: fall back to nvidia-smi with the offsets
  # measured above, and say so rather than pretending they are exact.
  CUDA_UNITS=0
  TOTAL_CUDA=$(( TOTAL - 448 )); FREE_CUDA=$(( FREE - 512 ))
fi

# Costs nothing to read and catches a real trap on this box: the compose file
# uses pull_policy: missing, so a stale local :latest is used as-is and the old
# stack is served with nothing in the logs saying so.
IMAGE_VLLM=""
if command -v docker >/dev/null 2>&1; then
  IMAGE_VLLM=$(timeout 60 docker run --rm --entrypoint cat \
      "$IMAGE" /app/docker/requirements.txt 2>/dev/null \
    | sed -n 's/^vllm==//p' | head -1)
fi

# ---- budget -----------------------------------------------------------------
# What may be handed to the pool. The clamp is unconditional and that is the
# whole point: --gpu-memory-utilization is a fraction of TOTAL memory, and vLLM
# refuses to start unless util x total fits in FREE memory -- a single foreign
# process holding 972 MiB is enough to fail the shipped 0.93 (measured on this
# box: "Free memory on device cuda:0 (28.19/31.45 GiB) on startup is less than
# desired GPU memory utilization (0.93, 29.25 GiB)").
RESERVE=$DESKTOP_MIB
if [ "$(( TOTAL - FREE ))" -gt 2048 ]; then
  SHARED=1
  RESERVE=0                               # the other process is already off the top
fi
WANT_UTIL=0.93
[ "$MODE" = batch ] && WANT_UTIL=0.972
SLACK_MIB=256                             # keeps the check true after rounding
# --reserve-gib N is "keep N GiB available for other workloads". Expressed
# against the card rather than against the moment: with a training run holding
# 8.9 GiB and a reserve of 10, the reserve already covers it and the server gets
# total - 10 GiB; with the card idle it gets the same, so the profile does not
# change depending on when it was generated. A workload larger than the reserve
# still wins, and then the fit below is what saves the boot.
#
# With no explicit reserve, the default is "assume whatever is on the card
# stays": a card that is already running someone else's job gets a profile that
# fits beside it, and an idle card gets the whole thing. Sizing to the current
# free memory without reserving is the one thing that does not work: a training
# run that grows from 2.3 GiB to 8.9 GiB between this script and the boot turned
# a valid profile into a failed startup three times in a row.
if [ -z "$RESERVE_GIB" ]; then
  if [ "$(( TOTAL - FREE ))" -le 1536 ]; then
    RESERVE_GIB=0
  else
    RESERVE_GIB=$(( (TOTAL_CUDA - FREE_CUDA + 1023) / 1024 ))
  fi
fi
RESERVE_MIB=$(( RESERVE_GIB * 1024 ))
FIT_MIB=$(( TOTAL_CUDA - RESERVE_MIB ))
[ "$FIT_MIB" -gt "$FREE_CUDA" ] && FIT_MIB=$FREE_CUDA
FIT_MIB=$(( FIT_MIB - SLACK_MIB ))
[ "$FIT_MIB" -lt 4096 ] && { echo "gpu-profile: only ${FIT_MIB} MiB usable on $NAME; free the card first." >&2; exit 1; }
UTIL=$(awk -v f="$FIT_MIB" -v t="$TOTAL_CUDA" 'BEGIN{u=f/t; if(u>0.98)u=0.98; u=int(u*1000)/1000; printf "%.3f", u}')
# The shipped default is the target; the free-memory fit caps it.
[ "$(awk -v a="$UTIL" -v b="$WANT_UTIL" 'BEGIN{print (a>b)?1:0}')" = 1 ] && UTIL=$WANT_UTIL
BUDGET_MIB=$(awk -v u="$UTIL" -v t="$TOTAL_CUDA" 'BEGIN{printf "%d", u*t}')
# vLLM's startup check is literally `util x total <= free`; rounding the fraction
# to three decimals is enough to break it (0.9007 up to 0.901 overshot a 29,366
# MiB free card by 12 MiB), so the budget is clamped to the fit as well.
[ "$BUDGET_MIB" -gt "$FIT_MIB" ] && BUDGET_MIB=$FIT_MIB
POOL_MIB=$(( BUDGET_MIB - NON_KV_MIB - MARGIN_MIB ))
[ "$POOL_MIB" -lt 2048 ] && { echo "gpu-profile: ${POOL_MIB} MiB left for the KV pool after weights/graphs/margin on $NAME; free the card." >&2; exit 1; }
POOL_BYTES=$(( POOL_MIB * 1048576 ))

# Largest context ceiling whose tokens fit the pool. This is vLLM's own boot
# check: a pool that cannot hold one request at --max-model-len makes the engine
# refuse to start.
pick_max_len() {  # <bytes per token>
  local bpt=$1 c best=""
  for c in $CANDIDATES; do
    if [ $(( c * bpt * 100 )) -le $(( POOL_BYTES * POOL_USE_PCT )) ] && [ "$c" -le "$MODEL_MAX_LEN" ]; then
      best=$c
    fi
  done
  printf '%s' "$best"
}

if [ "$MODE" = batch ]; then
  MAX_LEN=$(pick_max_len "$BPT_FP8")
  [ -z "$MAX_LEN" ] && { echo "gpu-profile: no context ceiling fits a ${POOL_MIB} MiB pool" >&2; exit 1; }
  PROFILE_BODY="KV=fp8
MAX_LEN=$MAX_LEN
GPU_UTIL=$UTIL
MAX_SEQS=64"
else
  # CTX picks which of the three single-user profiles .env activates; the pool
  # and the ceiling are written for all three below as a comment block so
  # switching CTX does not need this script re-run.
  CTX_DEFAULT=fast
  MAX_LEN_FAST=$(pick_max_len "$BPT_BF16")
  MAX_LEN_LONG=$(pick_max_len "$BPT_FP8")
  MAX_LEN_HUGE=$(pick_max_len "$BPT_KVARN")
  [ -z "$MAX_LEN_FAST" ] && { echo "gpu-profile: no context ceiling fits a ${POOL_MIB} MiB pool at bf16 KV" >&2; exit 1; }
  MAX_LEN=$MAX_LEN_FAST
  PROFILE_BODY="SPEC=dflash2
CTX=$CTX_DEFAULT
PREFIX_CACHE=1
MAX_LEN=$MAX_LEN
GPU_UTIL=$UTIL
MAX_SEQS=8
KV_MEM=$POOL_BYTES"
fi

reasons() {
  cat <<EOF
card            $NAME
compute cap     sm_${CC_TAG}
VRAM            ${TOTAL} MiB total (nvidia-smi), ${TOTAL_CUDA} MiB as the CUDA context sees it
free            ${FREE} MiB free (nvidia-smi), ${FREE_CUDA} MiB as the CUDA context sees it
held elsewhere  ${OTHER} MiB (nvidia-smi accounting)
gpu utilization $UTIL  (shipped default for this mode: $WANT_UTIL)
budget          ${BUDGET_MIB} MiB = $UTIL x ${TOTAL_CUDA}
non-KV          ${NON_KV_MIB} MiB (weights, drafter, CUDA graphs, activation peak)
margin          ${MARGIN_MIB} MiB transient + ${RESERVE} MiB desktop reserve + ${SLACK_MIB} MiB rounding slack
reserved        ${RESERVE_GIB} GiB kept for other workloads (--reserve-gib)
EOF
  if [ "$MODE" = batch ]; then
    echo "KV pool         sized by vLLM from the budget; batch mode does not pin it"
    echo "max-model-len   $MAX_LEN (fp8 KV, ${BPT_FP8} B/token)"
  else
    echo "KV pool         ${POOL_MIB} MiB pinned (24 GiB profile pins $(( REF_PIN_FAST / 1048576 )) MiB)"
    echo "pool cost       ${BPT_BF16} B/token bf16, ${BPT_FP8} fp8, ${BPT_KVARN} KVarN; ceiling takes ${POOL_USE_PCT}% of the pool"
    echo "max-model-len   fast(bf16) $MAX_LEN_FAST | long(fp8) $MAX_LEN_LONG | huge(KVarN) $MAX_LEN_HUGE"
  fi
  # On a shared card the context ceiling is a budget decision, so show the curve
  # instead of one point. These rows assume the card is otherwise free -- the
  # profile actually written is additionally capped by what is free right now.
  if [ "$SHOW_TABLE" = 1 ]; then
    echo
    printf '  %-9s %-9s %-11s %-11s %s\n' reserve pool fast long huge
    local r p budget mf ml mh
    for r in 0 2 4 6 8 10 12 16 20; do
      budget=$(( TOTAL_CUDA - r * 1024 - SLACK_MIB ))
      p=$(( budget - NON_KV_MIB - MARGIN_MIB ))
      if [ "$p" -lt 1024 ]; then
        printf '  %-9s %s\n' "${r}G" "no pool left"
        continue
      fi
      POOL_BYTES=$(( p * 1048576 ))
      mf=$(pick_max_len "$BPT_BF16"); ml=$(pick_max_len "$BPT_FP8"); mh=$(pick_max_len "$BPT_KVARN")
      printf '  %-9s %-9s %-11s %-11s %s\n' "${r}G" "${p}M" "${mf:-none}" "${ml:-none}" "${mh:-none}"
    done
    POOL_BYTES=$(( POOL_MIB * 1048576 ))
  fi
}

if [ "$EXPLAIN" = 1 ]; then
  reasons
  exit 0
fi

if [ -n "$IMAGE_VLLM" ] && [ "$IMAGE_VLLM" != "0.28.0" ]; then
  echo "gpu-profile: WARNING: the local ghcr.io/syv-ai/qwen38-27b-rtx3090:latest image" >&2
  echo "             contains vLLM $IMAGE_VLLM, not the 0.28.0 this repo pins. compose uses" >&2
  echo "             pull_policy: missing, so a stale local image is served as-is:" >&2
  echo "             docker pull ghcr.io/syv-ai/qwen38-27b-rtx3090:latest" >&2
fi
if [ "$SHARED" = 1 ] || [ "$(awk -v a="$UTIL" -v b="$WANT_UTIL" 'BEGIN{print (a<b)?1:0}')" = 1 ]; then
  echo "gpu-profile: ${OTHER} MiB of this card is used by another process, so the pool and" >&2
  echo "             the ceiling are sized to fit free memory (${POOL_MIB} MiB, util ${UTIL})" >&2
  echo "             rather than from the shipped defaults. Run this again with the card" >&2
  echo "             free for the full-card profile." >&2
fi
if [ "${CC_TAG:-}" = 120 ]; then
  echo "gpu-profile: note: sm_120 has no int8 tensor-core rate advantage to spend on the" >&2
  echo "             W4A8 path: int8 MMA measures 1.05x fp16 here (8192^3 matmul, torch" >&2
  echo "             _int_mm against fp16), against 2x on the sm86 cards every INT8_ACT" >&2
  echo "             figure in this repo was measured on. Leave INT8_ACT unset single-user;" >&2
  echo "             batch mode sets it by default, so override it there if prefill matters." >&2
fi

ENV_BODY="# Generated by tools/gpu-profile.sh for this machine. .env is gitignored.
# Sized for the card as it was when this ran: what was already using it is
# assumed to stay, and re-running is how you hand the pool back once it is idle.
$PROFILE_BODY"
# Keep the API key across re-runs: regenerating it on every profile refresh would
# silently lock out every client already pointed at this server.
KEEP_KEY=""
[ -e "$ENV_FILE" ] && KEEP_KEY=$(sed -n 's/^VLLM_API_KEY=//p' "$ENV_FILE" | head -1)
[ -z "$KEEP_KEY" ] && [ -s "$REPO_ROOT/api_key.txt" ] && KEEP_KEY=$(head -1 "$REPO_ROOT/api_key.txt")
# The server binds 0.0.0.0, so without a key every host on the LAN can use it.
# Generated here rather than left to the caller so a fresh install is not open;
# --no-key writes no key (local-only, the repo's documented default).
if [ "$NO_KEY" = 0 ] && [ -z "$KEEP_KEY" ]; then
  KEEP_KEY=$(openssl rand -hex 24 2>/dev/null) || KEEP_KEY=$(head -c 48 /dev/urandom | od -An -tx1 | tr -d ' \n')
fi
[ -n "$KEEP_KEY" ] && ENV_BODY="$ENV_BODY
VLLM_API_KEY=$KEEP_KEY"
[ "$MODE" = single ] && ENV_BODY="$ENV_BODY
#
# The other two single-user profiles on this card, applied by editing CTX=
# (pool ${POOL_MIB} MiB is shared by all three, only the ceiling moves):
#   CTX=long   fp8 KV, 3 drafts      -> MAX_LEN=${MAX_LEN_LONG:-none}
#   CTX=huge   KVarN 4/2-bit KV      -> MAX_LEN=${MAX_LEN_HUGE:-none}
# CTX=fast is the fastest of the three and the only one the split-KV verify
# attention applies to; long and huge exist to reach the context fast cannot."

if [ "$PRINT" = 1 ]; then
  # --print is for inspecting the profile, and the profile carries the API key:
  # printing a live secret into a terminal or a transcript is how it leaks.
  printf '%s\n' "$ENV_BODY" | sed 's/^VLLM_API_KEY=.*/VLLM_API_KEY=<preserved, not printed>/'
  { echo; reasons; } >&2
  exit 0
fi

if [ -e "$ENV_FILE" ] && [ "$FORCE" != 1 ] && ! grep -q "Generated by tools/gpu-profile.sh" "$ENV_FILE"; then
  echo "gpu-profile: $ENV_FILE exists and was not written by this script; not touching it." >&2
  echo "             Inspect it, then rerun with --force (a .env.gpu-profile.bak copy is kept)." >&2
  exit 2
fi
[ -e "$ENV_FILE" ] && cp "$ENV_FILE" "$ENV_FILE.gpu-profile.bak"
printf '%s\n' "$ENV_BODY" > "$ENV_FILE"
echo "gpu-profile: wrote $ENV_FILE"
reasons
if [ -e "$ENV_FILE.gpu-profile.bak" ]; then
  echo "backup          $ENV_FILE.gpu-profile.bak"
fi
