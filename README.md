# RLM chat engine

Copied RLM orchestrator, llm-proxy, storage, and tool-runner. Bound for a single chat model plus an embedding sidecar through config — not through Python literals.

The original BORHAN / AIv2 tree is not modified.

## Layout

- `services/orchestrator` — RLM IPython harness, memory, compact, auto-continue
- `services/llm-proxy` — OpenAI-compatible gateway to vLLM / llm-d
- `services/storage` — sessions and LLM registry
- `services/tool-runner` — optional tools
- `skills/` — markdown skills the kernel loads with `skills.use(name)`
- `config/settings.yaml` — model ids, URLs, engine default
- `ui/` — small chat page
- `deploy/k3d` and `deploy/llm-d` — unused until GPU bring-up

## Unit tests (no GPU)

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r services/orchestrator/requirements-dev.txt
pip install -e packages/runtime-config
cd services/orchestrator
PYTHONPATH=.:../../packages/runtime-config/src pytest tests/test_skill_catalog.py tests/test_runtime_config_and_no_hardcode.py tests/test_compact_and_continue_config.py tests/test_rlm_skills_session_and_sandbox.py tests/test_ui_config_endpoint.py tests/test_self_compact.py tests/test_session_compactor.py tests/test_task_handoff.py tests/test_qdrant_memory.py tests/test_rlm_length_continuation.py tests/test_rlm_run_lifetime_chat.py tests/test_rlm_chat_force_answer.py
```

GPU, k3d, and llm-d stay off until you ask.

## Loop modes

Turn-based, goal, time, and proactive loops are **triggered automatically** by schedulers and API fields (`context.scheduled_tick`, `POST /loops/run`, etc.) — not by user slash commands (disabled by default). See [docs/LOOP_MODES.md](docs/LOOP_MODES.md), `GET /loop-modes`, and `scripts/run_full_checklist.sh`.
