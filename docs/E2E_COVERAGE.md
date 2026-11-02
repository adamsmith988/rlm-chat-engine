# E2E coverage map

See the checklist scripts: `scripts/e2e_full_checklist.py`, `scripts/e2e_checklist.py`.

Run: `bash scripts/run_full_checklist_watch.sh` → `.local/bringup/full-checklist.log`

Progress total: **26 steps** (18 API/stack + 8 chat GPU checks).

Full mapping is documented in chat history; this file is the quick index:

| Area | Checks |
|------|--------|
| Stack | orchestrator, llm-proxy, storage, vLLM :18080 |
| Loops | `/loop-modes`, `/loops/resolve`, slash disabled |
| Primitives | skills, budgets, refine/review, `/memory/status` |
| Chat E2E | conversation, ipython, short/long memory, session compact checkpoint, coherence, skills |

**Tool-runner:** `modules/tools` is a stub (no `tools.registry`); RLM ipython still works. Manifest 503 is expected until real tools are vendored.

Tunables: `E2E_*` env vars in `scripts/e2e_checklist.py`.
