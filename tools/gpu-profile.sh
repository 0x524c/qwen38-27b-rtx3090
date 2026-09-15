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
#   bash tools/gpu-profile.sh --ctx 131072   # a smaller target than the model max
#   bash tools/gpu-profile.sh --kv fast       # force the bf16 profile if it fits
#
# It only reads (nvidia-smi, docker inspect). It never starts, stops or restarts
# anything, and it never overwrites an .env it did not write without --force.

set -u

MODE=single
PRINT=0
EXPLAIN=0
FORCE=0
NO_KEY=0
SHOW_TABLE=0
SEATS=""         # empty = the profile's own default
TARGET_LEN=""
KV_PREF=auto
TARGET_REACHED=1
RESERVE_GIB=""                             # empty = derive it from the card
while [ $# -gt 0 ]; do
  case "$1" in
    --print)       PRINT=1 ;;
    --explain)     EXPLAIN=1; SHOW_TABLE=1 ;;
    --table)       SHOW_TABLE=1 ;;
    --ctx)         TARGET_LEN=${2:-0}; shift ;;
    --kv)          KV_PREF=${2:-auto}; shift ;;
    --seats)       SEATS=${2:-0}; shift ;;
    --force)       FORCE=1 ;;
    --no-key)      NO_KEY=1 ;;
    --mode)        MODE=${2:-}; shift ;;
    --reserve-gib) RESERVE_GIB=${2:-0}; shift ;;
    -h|--help) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
case "$MODE" in single|batch) ;; *) echo "--mode takes single or batch" >&2; exit 2 ;; esac
if [ -n "$RESERVE_GIB" ]; then
  case "$RESERVE_GIB" in ''|*[!0-9]*) echo "--reserve-gib takes a whole number of GiB" >&2; exit 2 ;; esac
fi
case "$TARGET_LEN" in *[!0-9]*) echo "--ctx takes a token count" >&2; exit 2 ;; esac
# Single mode names its profiles for the CTX value it writes (fast/long/huge);
# batch mode names the cache itself (fp8/kvarn/int4pth). Accept both.
case "$KV_PREF" in auto|fast|long|huge|fp8|kvarn|int4pth) ;; *) echo "--kv takes auto, fast, long, huge, fp8, kvarn or int4pth" >&2; exit 2 ;; esac

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
#   CTX=long  int8_per_token_head, Triton  5,583,457,484 B -> 136,429 tokens
#   CTX=huge  KVarN 5,261,334,938 B -> 268,169 tokens
# The effective cost per token rises a little with pool size, so extrapolating
# upward on these under-predicts the token count, which is the safe direction.
BPT_BF16=96818
BPT_INT8PTH=40925
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
# No context ceiling by default: the model's own maximum. The repo ships
# 64k/150k/240k defaults because a 24 GiB card has to choose; this one does not
# have to, and a profile that stops short of the model's context is a profile the
# caller has to remember to raise.
[ -z "$TARGET_LEN" ] && TARGET_LEN=$MODEL_MAX_LEN

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
  probe=$(timeout 240 docker run --rm --gpus all --entrypoint \
      bash "$IMAGE" -c "$CUDA_PROBE" 2>/dev/null | tail -1)
  # Only two decimal integers count as an answer. The probe fails with a torch
  # error on stderr and nothing on stdout when another container holds the card,
  # and a read() of that used to leave the variables empty and the arithmetic
  # below to compute nonsense out of an empty string.
  case "$probe" in
    ''|*[!0-9\ ]*) ;;
    *\ *)
      case "${probe%% *}" in ''|*[!0-9]*) ;; *) TOTAL_CUDA=${probe%% *} ;; esac
      case "${probe##* }" in ''|*[!0-9]*) ;; *) FREE_CUDA=${probe##* } ;; esac
      ;;
  esac
fi
CUDA_UNITS=1
if [ -z "$TOTAL_CUDA" ] || [ -z "$FREE_CUDA" ]; then
  # No docker, or the probe did not answer: fall back to nvidia-smi with the
  # offsets measured above, and say so rather than pretending they are exact.
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
SHARED=0
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
# Two budgets, because the target matters more than the moment. FREE_NOW is what
# can be handed out without disturbing whatever is already on the card; FREE_CARD
# is what the card has when it is not shared. A profile that reaches a long
# context is written from FREE_CARD even when something else is resident, and the
# boot condition is reported instead of being pre-empted by a smaller profile.
FREE_NOW_MIB=$(( FREE_CUDA - RESERVE_MIB - SLACK_MIB ))
FREE_CARD_MIB=$(( TOTAL_CUDA - RESERVE_MIB - SLACK_MIB ))
if [ "$FREE_NOW_MIB" -lt 2048 ]; then
  echo "gpu-profile: only ${FREE_NOW_MIB} MiB usable on $NAME right now ($(( TOTAL - FREE )) MiB held elsewhere)." >&2
  echo "             If a server of this repo is running, stop it first:" >&2
  echo "             docker compose --profile single down" >&2
  exit 1
fi
# KVarN's fp16 tail pool and the paged KV cache draw from the same envelope:
# KVARN_POOL_MEM_FRAC (0.15 single-user, 0.25 batch) is the share of
# (util x total - weights) the staging pool takes, and the paged cache gets the
# rest (kvarn/.../kvarn/config.py, pool_budget_bytes). So a KVarN profile costs
# its pool divided by (1 - frac), and that is the number the budget has to hold.
KVARN_FRAC=${KVARN_POOL_MEM_FRAC:-15}
KVARN_FRAC=${KVARN_FRAC%%.*}
[ -z "$KVARN_FRAC" ] && KVARN_FRAC=15
POOL_NOW_MIB=$(( FREE_NOW_MIB - NON_KV_MIB - MARGIN_MIB ))
POOL_CARD_MIB=$(( FREE_CARD_MIB - NON_KV_MIB - MARGIN_MIB ))
# The shipped per-mode utilization is also a ceiling on the pool: sizing a pool
# past WANT_UTIL x total would ask vLLM for memory its own utilization check
# refuses, so the footprint budget is the smaller of free memory and that cap.
POOL_AT_WANT_MIB=$(awk -v w="$WANT_UTIL" -v t="$TOTAL_CUDA" 'BEGIN{printf "%d", w*t}')
POOL_AT_WANT_MIB=$(( POOL_AT_WANT_MIB - NON_KV_MIB - MARGIN_MIB ))
[ "$POOL_NOW_MIB" -gt "$POOL_AT_WANT_MIB" ] && POOL_NOW_MIB=$POOL_AT_WANT_MIB
[ "$POOL_CARD_MIB" -gt "$POOL_AT_WANT_MIB" ] && POOL_CARD_MIB=$POOL_AT_WANT_MIB

# What a pinned pool costs the budget under a given cache format. KVarN needs its
# staging share on top; the others are the pool and nothing else.
pool_footprint_mib() {  # <pool MiB> <format>
  if [ "$2" = huge ]; then
    printf '%s' $(( $1 * 100 / (100 - KVARN_FRAC) + 1 ))
  else
    printf '%s' "$1"
  fi
}

# The inverse: the largest pool a footprint budget allows under a format.
pool_max_mib() {  # <usable footprint MiB> <format>
  if [ "$2" = huge ]; then
    printf '%s' $(( $1 * (100 - KVARN_FRAC) / 100 ))
  else
    printf '%s' "$1"
  fi
}
[ "$POOL_CARD_MIB" -lt 2048 ] && { echo "gpu-profile: ${POOL_CARD_MIB} MiB left for the KV pool after weights/graphs/margin on $NAME." >&2; exit 1; }

# Pool a context ceiling needs, in MiB, from the measured cost per token. This is
# vLLM's own boot check: a pool that cannot hold one request at --max-model-len
# makes the engine refuse to start.
pool_need_mib() {  # <tokens> <bytes per token>
  local m=$(( $1 * $2 / 1048576 ))
  printf '%s' $(( m * 100 / POOL_USE_PCT + 1 ))
}

# The utilization that matches a pool. vLLM's check is `util x total <= free`, so
# the fraction is derived from what the pool needs instead of the other way round,
# and floored to three decimals because rounding up is enough to fail it.
util_for_pool() {  # <pool MiB>
  local need=$(( $1 + NON_KV_MIB + MARGIN_MIB + SLACK_MIB ))
  awk -v n="$need" -v t="$TOTAL_CUDA" -v w="$WANT_UTIL" \
    'BEGIN{u=n/t; if(u>w)u=w; u=int(u*1000)/1000; printf "%.3f", u}'
}

# Batch mode does not pin the pool, so the transient headroom has to be held back
# by the utilization itself: there is no pool number to shrink later. Deriving it
# from free memory alone puts the engine's own warmup over the edge -- at 0.966
# with MAX_SEQS=64 the first boot here died in "warming up sampler with 64 dummy
# requests" with 117 MiB free, which is the sharp floor docs/gotchas.md describes.
util_for_batch() {
  local avail=$(( FREE_CUDA - RESERVE_MIB - SLACK_MIB - MARGIN_MIB ))
  awk -v a="$avail" -v t="$TOTAL_CUDA" -v w="$WANT_UTIL" \
    'BEGIN{u=a/t; if(u>w)u=w; if(u<0.5)u=0.5; u=int(u*1000)/1000; printf "%.3f", u}'
}

# Largest context ceiling whose tokens fit a pool. This is vLLM's own boot
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
  # Batch mode does not pin the pool: vLLM sizes it from the utilization, so what
  # this can do is choose the cache format and the context ceiling that fit. KVarN
  # is the only format that reaches 262144 here, and batch is the mode for many
  # concurrent requests, where its lack of a speculator matters: no draft means no
  # per-request recurrent-state slots for drafts, and that -- not the KV tokens --
  # is what bounds how many requests a hybrid model keeps resident.
  case "$KV_PREF" in
    kvarn) BATCH_KV=kvarn; BATCH_BPT=$BPT_KVARN ;;
    int4pth) BATCH_KV=int4pth; BATCH_BPT=$BPT_INT8PTH ;;
    *) BATCH_KV=fp8; BATCH_BPT=$BPT_INT8PTH ;;
  esac
  POOL_MIB=$POOL_NOW_MIB
  POOL_BYTES=$(( POOL_NOW_MIB * 1048576 ))
  MAX_LEN=$(pick_max_len "$BATCH_BPT")
  [ -z "$MAX_LEN" ] && { echo "gpu-profile: no context ceiling fits a ${POOL_NOW_MIB} MiB pool" >&2; exit 1; }
  UTIL=$(util_for_batch)
  PROFILE_BODY="KV=$BATCH_KV
PREFIX_CACHE=1
MAX_LEN=$MAX_LEN
GPU_UTIL=$UTIL
MAX_SEQS=64"
else
  # Which single-user profile: the fastest one that reaches the target context.
  # bf16 KV (CTX=fast, FlashAttention, the split-KV verify kernel) is the fastest
  # and the only one a 262k target rules out: 24.2 GiB of pool for the tokens
  # alone. Past that, KVarN (CTX=huge) is the repo's path to 262k and needs about
  # half the pool of int8_per_token_head (CTX=long, the Triton backend), which is
  # what makes it the one that survives other work on the card; docs/long-context.md
  # also measures it faster than the Triton caches at 100k. Order is fastest
  # first among those that reach the target: fast, then huge, then long.
  CTX_DEFAULT=""
  for fmt in fast huge long; do
    case "$fmt" in
      fast) bpt=$BPT_BF16;; huge) bpt=$BPT_KVARN;; long) bpt=$BPT_INT8PTH;;
    esac
    pool_need=$(pool_need_mib "$TARGET_LEN" "$bpt")
    need=$(pool_footprint_mib "$pool_need" "$fmt")
    if [ "$need" -le "$POOL_CARD_MIB" ] && { [ "$KV_PREF" = auto ] || [ "$KV_PREF" = "$fmt" ]; }; then
      CTX_DEFAULT=$fmt; CTX_BPT=$bpt; TARGET_POOL_MIB=$pool_need; break
    fi
  done
  if [ -z "$CTX_DEFAULT" ]; then
    # Nothing reaches the target on this card. Fall back to the furthest any
    # format gets, and say the target was dropped.
    CTX_DEFAULT=huge; CTX_BPT=$BPT_KVARN
    POOL_MIB=$(pool_max_mib "$POOL_NOW_MIB" huge)
    POOL_BYTES=$(( POOL_MIB * 1048576 ))
    MAX_LEN=$(pick_max_len "$BPT_KVARN")
    TARGET_REACHED=0
  else
    TARGET_REACHED=1
    MAX_LEN=$TARGET_LEN
    # Pin the largest pool the card allows rather than a bare multiple of the
    # target. The pool is what both context and concurrency draw from, and a
    # seat is an admission, not a reservation (the launcher's own note), so a
    # pool sized for one long request is what caps a server that mostly sees
    # short ones -- 8 concurrent 40k agent calls need 8 x 40k x 19,620 B of
    # KVarN, which is more than a single 262144-token request does.
    POOL_MIB=$(pool_max_mib "$POOL_CARD_MIB" "$CTX_DEFAULT")
    cap_now=$(pool_max_mib "$POOL_NOW_MIB" "$CTX_DEFAULT")
    [ "$POOL_MIB" -gt "$cap_now" ] && POOL_MIB=$cap_now
    [ "$POOL_MIB" -lt "$TARGET_POOL_MIB" ] && POOL_MIB=$TARGET_POOL_MIB
  fi
  case "$CTX_DEFAULT" in
    # The shipped single-user defaults are single-stream figures (KVarN ships 2,
    # which is one long request plus one). A pool this size holds far more short
    # requests than that, and a server whose load is parallel agent calls hits the
    # admission limit long before it hits memory, so KVarN gets the same 8 as the
    # bf16 profile. --seats overrides.
    fast) MAX_SEQS_DEFAULT=8;; long) MAX_SEQS_DEFAULT=4;; huge) MAX_SEQS_DEFAULT=8;;
  esac
  MAX_SEQS=${SEATS:-$MAX_SEQS_DEFAULT}
  POOL_FOOTPRINT_MIB=$(pool_footprint_mib "$POOL_MIB" "$CTX_DEFAULT")
  UTIL=$(util_for_pool "$POOL_FOOTPRINT_MIB")
  POOL_BYTES=$(( POOL_MIB * 1048576 ))
  MAX_LEN_FAST=$(pick_max_len "$BPT_BF16")
  MAX_LEN_LONG=$(pick_max_len "$BPT_INT8PTH")
  MAX_LEN_HUGE=$(pick_max_len "$BPT_KVARN")
  PROFILE_BODY="SPEC=dflash2
CTX=$CTX_DEFAULT
PREFIX_CACHE=1
MAX_LEN=$MAX_LEN
GPU_UTIL=$UTIL
MAX_SEQS=$MAX_SEQS
KV_MEM=$POOL_BYTES"
fi

UTIL_MIB=$(awk -v u="$UTIL" -v t="$TOTAL_CUDA" 'BEGIN{printf "%d", u*t}')
# What has to be free for the engine to come up: everything the pool does not cover,
# plus the pool and, for KVarN, its staging share, plus the transient headroom.
# vLLM checks the utilization side of this itself at startup.
[ -n "${POOL_FOOTPRINT_MIB:-}" ] || POOL_FOOTPRINT_MIB=$POOL_MIB
NEED_FREE_MIB=$(( NON_KV_MIB + POOL_FOOTPRINT_MIB + MARGIN_MIB ))
BOOTS_NOW=0
[ "$NEED_FREE_MIB" -le "$FREE_CUDA" ] && BOOTS_NOW=1

reasons() {
  cat <<EOF
card            $NAME
compute cap     sm_${CC_TAG}
VRAM            ${TOTAL} MiB total (nvidia-smi), ${TOTAL_CUDA} MiB as the CUDA context sees it
free            ${FREE} MiB free (nvidia-smi), ${FREE_CUDA} MiB as the CUDA context sees it
held elsewhere  ${OTHER} MiB (nvidia-smi accounting)
gpu utilization $UTIL  (shipped default for this mode: $WANT_UTIL)
budget          ${UTIL_MIB} MiB = $UTIL x ${TOTAL_CUDA}
non-KV          ${NON_KV_MIB} MiB (weights, drafter, CUDA graphs, activation peak)
margin          ${MARGIN_MIB} MiB transient + ${RESERVE} MiB desktop reserve + ${SLACK_MIB} MiB rounding slack
reserved        ${RESERVE_GIB} GiB kept for other workloads (--reserve-gib)
target context  ${TARGET_LEN}$( [ "$TARGET_REACHED" = 0 ] && echo " (NOT reachable on this card; profile carries ${MAX_LEN})" )
free to boot    needs ${NEED_FREE_MIB} MiB, ${FREE_CUDA} MiB free now -> $( [ "$BOOTS_NOW" = 1 ] && echo "boots" || echo "WILL NOT BOOT until the card has room" )
EOF
  if [ "$MODE" = batch ]; then
    echo "KV pool         sized by vLLM from the budget (batch mode does not pin)"
    echo "max-model-len   $MAX_LEN (${BATCH_BPT:-$BPT_INT8PTH} B/token, KV=$BATCH_KV)"
  else
    echo "KV pool         ${POOL_MIB} MiB pinned (24 GiB profile pins $(( REF_PIN_FAST / 1048576 )) MiB)"
    [ "${POOL_FOOTPRINT_MIB:-$POOL_MIB}" != "$POOL_MIB" ] && \
      echo "                + $(( POOL_FOOTPRINT_MIB - POOL_MIB )) MiB KVarN staging at KVARN_POOL_MEM_FRAC=${KVARN_FRAC}%"
    echo "pool cost       ${BPT_BF16} B/token bf16, ${BPT_INT8PTH} int8_per_token_head, ${BPT_KVARN} KVarN; ceiling takes ${POOL_USE_PCT}% of the pool"
    echo "ceiling choices fast(bf16) ${MAX_LEN_FAST} | huge(KVarN) ${MAX_LEN_HUGE} | long(int8 pth) ${MAX_LEN_LONG} at this pool"
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
      mf=$(pick_max_len "$BPT_BF16"); ml=$(pick_max_len "$BPT_INT8PTH"); mh=$(pick_max_len "$BPT_KVARN")
      printf '  %-9s %-9s %-11s %-11s %s\n' "${r}G" "${p}M" "${mf:-none}" "${ml:-none}" "${mh:-none}"
    done
    POOL_BYTES=$(( POOL_MIB * 1048576 ))
  fi
}

[ "$CUDA_UNITS" = 0 ] && echo "gpu-profile: note: the CUDA-view probe did not answer, so these numbers come from\n             nvidia-smi plus the offsets measured on this box (~448 MiB of total and\n             ~512 MiB of free that the CUDA context does not report)." >&2
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
if [ "$TARGET_REACHED" = 0 ]; then
  echo "gpu-profile: WARNING: ${TARGET_LEN} tokens do not fit this card at any KV format:" >&2
  echo "             KVarN (the cheapest) needs $(( $(pool_need_mib "$TARGET_LEN" "$BPT_KVARN") )) MiB of pool and the card" >&2
  echo "             has ${POOL_CARD_MIB} MiB for one. The profile written carries ${MAX_LEN}." >&2
fi
if [ "$BOOTS_NOW" = 0 ]; then
  echo "gpu-profile: ${OTHER} MiB of this card is used by another process, so this profile will" >&2
  echo "             NOT boot yet: it needs ${NEED_FREE_MIB} MiB free and ${FREE_CUDA} MiB is free." >&2
  echo "             The profile is written for the card at ${TARGET_LEN} context anyway; free" >&2
  echo "             the card (or lower --ctx) and start the server." >&2
fi
if [ "$SHARED" = 1 ] && [ "$BOOTS_NOW" = 1 ]; then
  echo "gpu-profile: ${OTHER} MiB of this card is used by another process, and the profile" >&2
  echo "             still fits beside it (pool ${POOL_MIB} MiB, util ${UTIL}). The pool was" >&2
  echo "             capped at what is free now rather than at the card's own budget." >&2
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
[ "$MODE" = single ] && {
  ENV_BODY="$ENV_BODY
#
# All three single-user profiles on this pool of ${POOL_MIB} MiB (only the cache
# format, the ceiling and the slot count move -- edit CTX= to switch):
#   CTX=fast   bf16 KV on FlashAttention, 8 slots      -> ${MAX_LEN_FAST}
#   CTX=huge   KVarN 4/2-bit, 2 slots                  -> ${MAX_LEN_HUGE}
#   CTX=long   int8_per_token_head on Triton, 4 slots  -> ${MAX_LEN_LONG}
# active: ${CTX_DEFAULT}. fast is the fastest and the only one the split-KV verify
# kernel serves; huge and long are what reach the contexts bf16 cannot hold."
}

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
