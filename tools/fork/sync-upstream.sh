#!/usr/bin/env bash
# Reapply the 0x524c fork commits on top of the latest upstream main, and prove
# the patch set still applies.
#
# The fork keeps two long-lived refs:
#   main         - pristine mirror of upstream (syv-ai/qwen38-27b-rtx3090),
#                  always fast-forwarded, never carrying local commits
#   0x524c-patch - main plus the fork commits (this tooling, local fixes)
#
# What makes this repo different from an ordinary fork: most of its content is
# patches against a pinned vLLM (docker/requirements.txt, vllm==0.27.1). Two
# upstream changes can break the stack without touching a line we wrote - a new
# or renamed file in patches/, and a bump of that pin - so both are reported
# explicitly before the build gate runs.
#
#   bash tools/fork/sync-upstream.sh              # sync, then build (the gate)
#   bash tools/fork/sync-upstream.sh --no-build   # sync and report only
#
# It never pushes, never deletes, and stops on rebase conflicts with guidance.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UPSTREAM_URL="https://github.com/syv-ai/qwen38-27b-rtx3090"
UPSTREAM_BRANCH="main"
MIRROR_BRANCH="main"
PATCH_BRANCH="0x524c-patch"
BUILD=1
[ "${1:-}" = "--no-build" ] && BUILD=0

cd "$REPO_ROOT" || { echo "FAILED: cannot cd to $REPO_ROOT"; exit 1; }

step() { printf '\n== %s\n' "$1"; }
ok()   { printf 'ok: %s\n' "$1"; }
note() { printf '    %s\n' "$1"; }
fail() { printf 'FAILED: %s\n' "$1"; exit 1; }

pinned_vllm() {  # <ref> - the vllm== pin as of that ref
    git show "$1:docker/requirements.txt" 2>/dev/null \
        | sed -n 's/^vllm==\(.*\)$/\1/p' | head -1
}

step "preconditions"
git diff --quiet && git diff --cached --quiet \
    || fail "working tree has uncommitted changes; commit or stash them first"
ok "working tree clean"

step "upstream remote"
if git remote get-url upstream >/dev/null 2>&1; then
    ok "remote upstream already configured"
else
    git remote add upstream "$UPSTREAM_URL" || fail "adding upstream remote"
    ok "remote upstream added ($UPSTREAM_URL)"
fi

step "fetch upstream"
git fetch upstream || fail "git fetch upstream"
ok "fetched"

step "fast-forward $MIRROR_BRANCH"
current_branch="$(git rev-parse --abbrev-ref HEAD)"
before="$(git rev-parse "$MIRROR_BRANCH" 2>/dev/null)" || fail "no branch $MIRROR_BRANCH"
git checkout "$MIRROR_BRANCH" >/dev/null 2>&1 || fail "checkout $MIRROR_BRANCH"
behind="$(git rev-list --count "HEAD..upstream/$UPSTREAM_BRANCH")"
if [ "$behind" -eq 0 ]; then
    ok "$MIRROR_BRANCH already up to date"
else
    git merge --ff-only "upstream/$UPSTREAM_BRANCH" || fail \
        "$MIRROR_BRANCH does not fast-forward; it should never carry local commits"
    ok "$MIRROR_BRANCH advanced $behind commits to $(git rev-parse --short HEAD)"
fi
after="$(git rev-parse HEAD)"

step "what changed upstream that this stack depends on"
if [ "$before" = "$after" ]; then
    ok "nothing new"
else
    old_pin="$(pinned_vllm "$before")"; new_pin="$(pinned_vllm "$after")"
    if [ "$old_pin" != "$new_pin" ]; then
        printf 'ATTENTION: pinned vLLM moved %s -> %s\n' "${old_pin:-?}" "${new_pin:-?}"
        note "every patch in patches/ was written against the old pin. Read the"
        note "upstream commit that bumped it before trusting the build below."
    else
        ok "pinned vLLM unchanged (${new_pin:-unknown})"
    fi
    # A patch file added, renamed or removed changes what gets applied and in
    # which order: the launchers apply patches/*.patch in shell glob order, which
    # is why upstream renamed prefill-attn-int8 to triton-prefill-attn-int8 (it
    # has to land after spec-decode-attn).
    churn="$(git diff --name-status "$before" "$after" -- patches/ kvarn/ | grep -v '^M' || true)"
    if [ -n "$churn" ]; then
        printf 'patch set changed (added/removed/renamed):\n'
        printf '%s\n' "$churn" | sed 's/^/    /'
    else
        ok "no patch added, removed or renamed"
    fi
    printf 'commits:\n'
    git log --oneline "$before..$after" | sed 's/^/    /'
fi

step "rebase $PATCH_BRANCH"
git rev-parse --verify "$PATCH_BRANCH" >/dev/null 2>&1 \
    || fail "branch $PATCH_BRANCH does not exist"
git checkout "$PATCH_BRANCH" >/dev/null 2>&1 || fail "checkout $PATCH_BRANCH"
if git merge-base --is-ancestor "$MIRROR_BRANCH" "$PATCH_BRANCH"; then
    ok "$PATCH_BRANCH already based on current $MIRROR_BRANCH"
else
    if git rebase "$MIRROR_BRANCH"; then
        ok "$PATCH_BRANCH rebased onto $(git rev-parse --short "$MIRROR_BRANCH")"
    else
        printf '\nRebase stopped on a conflict. Resolve it, then run:\n'
        printf '  git rebase --continue\n'
        printf 'or abort with:\n'
        printf '  git rebase --abort\n'
        printf 'and rerun this script afterwards.\n'
        exit 1
    fi
fi

if [ "$BUILD" = 0 ]; then
    step "build skipped (--no-build)"
    note "the patch set is unproven until the image builds: run this without"
    note "--no-build, or docker compose build single, before serving."
else
    # The image build IS the gate: its Dockerfile applies every patches/*.patch
    # to the installed vLLM and then runs verify.sh --install, which fails the
    # build if any patch did not land. There is no venv on this box, so this is
    # the only place that check exists.
    step "build the image (applies every patch, runs verify.sh --install)"
    docker compose build single || fail \
        "image build - a patch no longer applies, or verify.sh rejected the install"
    ok "every patch applied and verify.sh --install passed"
fi

step "done"
printf 'patch branch: %s (%s)\n' "$PATCH_BRANCH" "$(git rev-parse --short HEAD)"
printf 'mirror:       %s (%s)\n' "$MIRROR_BRANCH" "$(git rev-parse --short "$MIRROR_BRANCH")"
if [ "$BUILD" = 1 ]; then
    printf 'image:        ghcr.io/syv-ai/qwen38-27b-rtx3090:latest, built locally from\n'
    printf '              %s - it shadows the upstream pull, which is what\n' "$PATCH_BRANCH"
    printf '              makes ai-server bin/llm serve the fork rather than upstream.\n'
fi
printf 'Nothing was pushed. Publish manually when ready.\n'
[ "$current_branch" = "$PATCH_BRANCH" ] || printf 'Note: you were on %s before; now on %s.\n' \
    "$current_branch" "$PATCH_BRANCH"
