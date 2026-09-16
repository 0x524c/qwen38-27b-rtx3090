# One command per thing this project does. Run `make` for the list.
#
# The long form lives in tools/serve.sh; these targets are thin wrappers so the same
# actions are reachable from the repository root. `make install` puts that same script
# on PATH as `qwen38`, which is the short form:
#
#     qwen38 start [fast|wide|quality|nvfp4] [--id NAME] [--model DIR]
#     qwen38 stop | status | list | logs [-f]

SHELL  := /usr/bin/env bash
.DEFAULT_GOAL := help

REPO   := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
SERVE  := bash $(REPO)/tools/serve.sh
BIN    ?= $(HOME)/bin
CLI    ?= qwen38
PROFILE ?= fast
SERVICE ?= single
LOGARGS ?=

.PHONY: help install uninstall start stop restart status list logs models check build gpu shell test

help: ## this list
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-11s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  profiles : fast (default)  wide (fan-out)  quality (INT6)  nvfp4"
	@echo "  override : make start PROFILE=wide      make logs LOGARGS=-f"
	@echo "  the CLI  : make install  then  $(CLI) start fast"

install: ## install the CLI as $(BIN)/$(CLI) (a symlink to tools/serve.sh)
	@mkdir -p $(BIN)
	@chmod +x $(REPO)/tools/serve.sh
	@ln -sfn $(REPO)/tools/serve.sh $(BIN)/$(CLI)
	@echo "installed $(BIN)/$(CLI) -> $(REPO)/tools/serve.sh"
	@case ":$$PATH:" in *":$(BIN):"*) echo "$(BIN) is on PATH";; \
	  *) echo "WARNING: $(BIN) is not on PATH; call it as $(BIN)/$(CLI)";; esac
	@echo
	@$(BIN)/$(CLI) --help | sed -n '1,8p'

uninstall: ## remove the installed CLI
	@rm -f $(BIN)/$(CLI)
	@echo "removed $(BIN)/$(CLI)"

start: ## start a profile (PROFILE=fast)
	@$(SERVE) start $(PROFILE)

stop: ## stop whatever this project is running
	@$(SERVE) stop

restart: ## stop then start (PROFILE=fast)
	@$(SERVE) restart $(PROFILE)

status: ## what is running, its pool, and the model ids it serves
	@$(SERVE) status

list: ## the profiles, the ids they serve, and what each one costs
	@$(SERVE) list

logs: ## container logs (make logs LOGARGS=-f)
	@$(SERVE) logs $(LOGARGS)

models: ## which checkpoints are on disk, and how to fetch the missing ones
	@echo "checkpoints under models/ (the container sees them at /app/models):"
	@for d in $(REPO)/models/*/; do \
	  printf "  %-44s %s\n" "$$(basename $$d)" "$$(du -sh $$d 2>/dev/null | cut -f1)"; \
	done
	@echo
	@echo "fetch a missing one inside the container:"
	@echo "  docker compose exec -T single /app/venv/bin/hf download Minachist/Qwen3.8-27B-INT6-Mixed-AutoRound --local-dir /app/models/Qwen3.8-27B-INT6-Mixed-AutoRound"
	@echo "  docker compose exec -T single /app/venv/bin/hf download QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 --local-dir /app/models/Qwen3.8-27B-QUASAR-NVFP4"

check: ## syntax-check the tools, then run the checkpoint gate's self-test
	@for f in $(REPO)/tools/*.sh; do bash -n "$$f" && printf "  ok   %s\n" "$$(basename $$f)"; done
	@python3 $(REPO)/tools/checkpoint-check.py --self-test | tail -1

build: ## rebuild the container image (needed after changing bench/ or docker/)
	@docker compose -f $(REPO)/docker-compose.yml build

gpu: ## power limit, temperature, and what is holding the GPU
	@nvidia-smi --query-gpu=name,power.limit,temperature.gpu,memory.used,memory.total --format=csv,noheader
	@nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
	@echo "thermal guard: $$(systemctl is-active cachyos-gpu-thermal-guard.service)  power cap: $$(systemctl is-active gpu-power-cap.service)"

shell: ## a shell inside the running container
	@docker compose exec $(SERVICE) bash

test: ## smoke test: health, served ids, and one short completion
	@key=$$(bash $(REPO)/tools/print-api-key.sh); \
	 code=$$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:18020/health); \
	 echo "  health: $$code"; [ "$$code" = 200 ] || exit 1; \
	 id=$$(curl -s --max-time 8 -H "Authorization: Bearer $$key" http://127.0.0.1:18020/v1/models \
	        | python3 -c 'import json,sys; print(",".join(m["id"] for m in json.load(sys.stdin)["data"]))'); \
	 echo "  ids: $$id"; \
	 first=$${id%%,*}; \
	 curl -s --max-time 180 -H "Authorization: Bearer $$key" -H 'Content-Type: application/json' \
	   http://127.0.0.1:18020/v1/chat/completions \
	   -d "{\"model\":\"$$first\",\"messages\":[{\"role\":\"user\",\"content\":\"Diga apenas: ok\"}],\"max_tokens\":16,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
	 | python3 -c 'import json,sys; d=json.load(sys.stdin); print("  reply:", repr(d["choices"][0]["message"]["content"].strip()[:40]), "· finish:", d["choices"][0]["finish_reason"])'
