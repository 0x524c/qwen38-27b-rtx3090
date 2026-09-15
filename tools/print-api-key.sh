#!/usr/bin/env bash
# Prints the API key of the running vLLM server, for pi's models.json:
#
#   "apiKey": "!bash <repo>/tools/print-api-key.sh"
#
# The key is read from the serving container's environment, so no copy of it sits
# in models.json (which is world-readable) and none is kept outside the container
# and the compose env file. pi resolves a "!command" apiKey at request time and
# treats a configured apiKey as auth present, which is what makes the models appear
# in /model and --list-models.
#
# The container is found by IMAGE, not by name: the compose project is named after
# the directory (qwen38-27b-rtx5090) while the image is ghcr.io/syv-ai/
# qwen38-27b-rtx3090, and the directory has been renamed once already. Matching on
# the image survives that, and only this project uses it.
set -u
IMAGE=${QWEN_IMAGE:-ghcr.io/syv-ai/qwen38-27b-rtx3090:latest}
ids=$(docker ps -q --filter "ancestor=${IMAGE}" 2>/dev/null)
for id in $ids; do
  if k=$(docker exec "$id" printenv VLLM_API_KEY 2>/dev/null) && [ -n "$k" ]; then
    printf '%s' "$k"
    exit 0
  fi
done
echo "print-api-key: no running container from ${IMAGE} has VLLM_API_KEY; start one with bash tools/serve.sh w4a16|nvfp4" >&2
exit 1
