# aiv2-model-contract

Versioned shared contracts between **mother** and **aiv2_node**.

## Install

```bash
pip install -e packages/aiv2-model-contract
# or PYTHONPATH=packages/aiv2-model-contract/src
```

## Modules

- `introduction` — Model Introduction v2 (`normalize_introduction`, `build_introduction`, …)
- `hf_pipeline_io`, `llm_taxonomy`, `model_capability_infer`, `run_schema_contract`, `models_paths`

## Schema

- `schemas/model-introduction-v2.json`
- `schemas/agent-runtimes-v1.json` — Agent API runtimes catalog shape (also shipped in the aiv2_node repo)

## Docs

- Mother: `docs/MODEL_INTRODUCTION_CONTRACT.md`
- Node repo: `docs/MODEL_INTRODUCTION_CONTRACT.md` (identical)
- ADR: `docs/adr/001-node-runtime-separation.md`
