# Model Catalog: Reasoning / Thinking Contract

**Audience:** `aiv2_node` (deploy/inspect), Storage model cards, llm-proxy catalog, agent-flow / RLM consumers  
**Status:** Design contract for implementation  
**Related code today:** `supports_reasoning` (bool only), Agent-node `enable_thinking`, RLM `preserve_thinking` (catalog + name fallback)

---

## 1. Problem

Agent runtimes (especially RLM multi-turn tool loops) must know **how** a model thinks, not only **whether** it can think:

1. Wrong `preserve_thinking` fills the context window (Gemma) or drops continuity (Qwen tool chains).
2. Wrong wire format leaves CoT stuck inside `content` or silently stripped by proxies.
3. Catalog today only has `capabilities.supports_reasoning: bool` — insufficient for deploy and runtime.

This document defines the **catalog fields**, **types**, **defaults**, and **ownership** so `aiv2_node` can detect/stamp them and mother services can consume them without hardcoding model-name heuristics.

---

## 2. Five layers (canonical)

| # | Layer | Catalog field (proposed) | Question answered |
|---|--------|---------------------------|-------------------|
| 1 | Capability | `reasoning.supported` | Does this model expose separable chain-of-thought at all? |
| 2 | Mode | `reasoning.mode` | Always-on vs toggleable vs none |
| 3 | Wire format | `reasoning.wire` | How CoT is enabled, emitted, and replayed on the API |
| 4 | History policy | `reasoning.history` | Whether / how prior turns’ CoT is sent back |
| 5 | Intensity | `reasoning.intensity` | Effort / token budget controls (if any) |

These layers are **orthogonal**. A model can be “supported + hybrid + qwen3 wire + preserve_all history + thinking_budget intensity”.

---

## 3. Proposed catalog schema

### 3.1 Top-level object on the model card

Store under `llm_local_models.metadata.reasoning` (preferred) **and** mirror a compact projection into `GET /catalog` → `capabilities` / `reasoning`.

```jsonc
{
  "reasoning": {
    "supported": true,
    "mode": "hybrid",
    "wire": {
      "family": "qwen3",
      "enable_param": "chat_template_kwargs.enable_thinking",
      "enable_default": true,
      "output_fields": ["reasoning_content", "reasoning"],
      "vllm_reasoning_parser": "qwen3",
      "inline_tags": ["<think>", "</think>"]
    },
    "history": {
      "policy": "preserve_all",
      "preserve_thinking_param": "chat_template_kwargs.preserve_thinking",
      "preserve_thinking_default": true,
      "replay_max_assistant_turns": 8,
      "requires_reasoning_on_assistant_replay": false
    },
    "intensity": {
      "kind": "none"
    }
  }
}
```

### 3.2 Field reference

#### Layer 1 — Capability

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `reasoning.supported` | `boolean` | yes | `true` if the serving stack can separate CoT from the user-visible answer (dedicated field and/or parseable tags). |
| `capabilities.supports_reasoning` | `boolean` | yes (legacy mirror) | **Must equal** `reasoning.supported`. Keep for backward compatibility. |

#### Layer 2 — Mode

| Field | Type | Values | Meaning |
|-------|------|--------|---------|
| `reasoning.mode` | `enum` | see below | How thinking is activated |

**`reasoning.mode` enum**

| Value | Meaning | Consumer behavior |
|-------|---------|-------------------|
| `none` | No separable CoT | Never set thinking kwargs; treat all text as answer |
| `hybrid` | Can turn thinking on/off per request | Honor Agent-node / request `enable_thinking` |
| `always_on` | Model always emits CoT when served as reasoning model | Ignore disable attempts (or only soft-hint); always parse reasoning fields |
| `unknown` | Not inspected yet | Fall back to safe defaults (see §6) |

#### Layer 3 — Wire format

| Field | Type | Required if supported | Meaning |
|-------|------|------------------------|---------|
| `reasoning.wire.family` | `enum` | yes | Logical family for adapters (not the HF repo name) |
| `reasoning.wire.enable_param` | `string` | if hybrid | Dotted request path, e.g. `chat_template_kwargs.enable_thinking` |
| `reasoning.wire.enable_default` | `boolean` | if hybrid | Server/template default when client omits the flag |
| `reasoning.wire.output_fields` | `string[]` | yes | Preference-ordered response fields: `reasoning_content`, `reasoning`, `reasoning_text` |
| `reasoning.wire.vllm_reasoning_parser` | `string \| null` | if vLLM | Value for `--reasoning-parser` (`qwen3`, `gemma4`, `deepseek_r1`, …) |
| `reasoning.wire.inline_tags` | `string[]` | optional | Tag pairs that may appear if field extraction fails |
| `reasoning.wire.channel_markers` | `string[]` | optional | e.g. Gemma `<|channel>thought` |
| `reasoning.wire.tool_call_parser` | `string \| null` | optional | Correlated tool parser (`qwen3_coder`, `gemma4`, …) — deploy hint |

**`reasoning.wire.family` enum (initial set)**

| Value | Typical enable | Typical output | Notes |
|-------|----------------|----------------|-------|
| `qwen3` | `chat_template_kwargs.enable_thinking` | `reasoning_content` | Also Qwen3.5 / 3.6 chat-template models |
| `gemma4` | `chat_template_kwargs.enable_thinking` (default **off**) | `reasoning` / channel thought | Multi-turn: prefer strip prior thoughts |
| `deepseek_r1` | often always-on | `<think>` / `reasoning_content` | QwQ often uses this parser |
| `openai_effort` | `reasoning_effort` | Responses / o-series style | Closed APIs |
| `anthropic_thinking` | `thinking` + budget | signed thinking blocks | Provider-native replay only |
| `granite` | `chat_template_kwargs.thinking` | parser-specific | vLLM granite |
| `generic_tags` | none / prompt | `<think>…</think>` in content | Last-resort scrape |
| `none` | — | — | No wire |

Adapters must use **`family` + `enable_param`**, not substring matches on `logical_id`.

#### Layer 4 — History policy

| Field | Type | Values / type | Meaning |
|-------|------|---------------|---------|
| `reasoning.history.policy` | `enum` | see below | How prior assistant CoT is treated on the **next** request |
| `reasoning.history.preserve_thinking_param` | `string \| null` | dotted path | Usually `chat_template_kwargs.preserve_thinking` |
| `reasoning.history.preserve_thinking_default` | `boolean \| null` | | Template/server default |
| `reasoning.history.replay_max_assistant_turns` | `integer` | ≥ 0 | Soft cap of assistant turns that may carry CoT on the wire (defense in depth) |
| `reasoning.history.requires_reasoning_on_assistant_replay` | `boolean` | | Some stacks require empty `reasoning_content` on all assistant msgs when thinking is on |

**`reasoning.history.policy` enum**

| Value | Meaning | Wire behavior |
|-------|---------|---------------|
| `strip` | Do not replay prior-turn CoT (Gemma4 doc default) | Omit `reasoning_*` except optionally the active tool-call chain after last user |
| `tool_chain_only` | Keep CoT only after last user message (active tool loop) | Strip completed turns; keep recent in-chain assistants |
| `preserve_all` | Replay historical CoT (Qwen agent / Prime-style) | Send `reasoning_content` on prior assistants; set `preserve_thinking=true` when param exists |
| `provider_native` | Provider owns replay (Anthropic signatures, etc.) | Do not invent OpenAI-style fields |
| `unknown` | Not classified | Use §6 fallbacks |

**Compatibility alias (required during migration):**

| Legacy / flat field | Maps to |
|---------------------|---------|
| `preserve_thinking: true` | `history.policy = preserve_all` + `preserve_thinking_default = true` |
| `preserve_thinking: false` | `history.policy = strip` (or `tool_chain_only` if node can detect tool loops) + `preserve_thinking_default = false` |
| `capabilities.preserve_thinking` | Same as flat `preserve_thinking` |

Mother RLM already reads `preserve_thinking` from catalog metadata / capabilities / `generation_defaults.chat_template_kwargs`. New nodes should populate **both** the structured `reasoning` object and the flat alias until all consumers migrate.

#### Layer 5 — Intensity

| Field | Type | Values | Meaning |
|-------|------|--------|---------|
| `reasoning.intensity.kind` | `enum` | `none`, `effort`, `token_budget`, `unknown` | Which knob exists |
| `reasoning.intensity.param` | `string \| null` | e.g. `reasoning_effort`, `thinking_budget`, `max_thinking_tokens` | Request path |
| `reasoning.intensity.values` | `string[] \| number[] \| null` | e.g. `["minimal","low","medium","high"]` | Allowed values |
| `reasoning.intensity.default` | `string \| number \| null` | | Default when UI omits |
| `reasoning.intensity.max_tokens` | `integer \| null` | | Hard ceiling if budget-based |

Many open-weight vLLM models have **`kind: none`** even when `supported: true` (CoT length is not separately capped).

---

## 4. Catalog projection (`GET /catalog`)

Each logical model entry should expose:

```jsonc
{
  "logical_id": "gemma4-31",
  "capabilities": {
    "supports_reasoning": true,
    "preserve_thinking": false
  },
  "reasoning": { /* full object from §3 */ },
  "generation_defaults": {
    "chat_template_kwargs": {
      "enable_thinking": false,
      "preserve_thinking": false
    }
  }
}
```

Rules:

1. `capabilities.supports_reasoning` ≡ `reasoning.supported`.
2. `capabilities.preserve_thinking` ≡ (`history.policy` in `preserve_all`) when policy is known; omit if `unknown`.
3. `generation_defaults.chat_template_kwargs` should reflect **safe defaults for that card**, not the user’s last Agent-node toggle.
4. Do not put secrets or node IPs in `reasoning`.

---

## 5. Ownership: what `aiv2_node` must implement

### 5.1 When to stamp

Populate `metadata.reasoning` at **inspect / deploy / endpoint-import enrich** time:

| Source | Action |
|--------|--------|
| vLLM serve args | Read `--reasoning-parser`, `--tool-call-parser`, `--default-chat-template-kwargs` |
| Hugging Face / GGUF card | Infer `wire.family` from known architectures when parser missing |
| OpenAI-compat probe | Optional: one-shot chat with thinking enabled; detect `reasoning_content` / channel markers |
| Manual admin override | Always wins over auto-detect |

### 5.2 Suggested detection matrix (initial)

| Evidence | Stamp |
|----------|--------|
| `--reasoning-parser qwen3` or Qwen3/3.5/3.6 chat template | `supported=true`, `mode=hybrid`, `wire.family=qwen3`, `history.policy=preserve_all`, `preserve_thinking_default=true`, `enable_default=true` |
| `--reasoning-parser gemma4` or Gemma 4 IT template | `supported=true`, `mode=hybrid`, `wire.family=gemma4`, `history.policy=strip` (prefer `tool_chain_only` if node can assert agent loop), `preserve_thinking_default=false`, `enable_default=false` |
| `--reasoning-parser deepseek_r1` | `supported=true`, `mode=always_on` (unless template says otherwise), `wire.family=deepseek_r1`, `history.policy=preserve_all` |
| No parser, no thinking markers | `supported=false`, `mode=none`, `wire.family=none`, `history.policy=strip` |
| Ambiguous | `supported` best-effort; `mode=unknown`; `history.policy=unknown` |

### 5.3 Deploy wiring

When launching vLLM, node should keep catalog and process aligned:

- Pass `--reasoning-parser` from `reasoning.wire.vllm_reasoning_parser` when set.
- Pass `--default-chat-template-kwargs` consistent with `enable_default` / `preserve_thinking_default`.
- On mismatch (catalog says `gemma4` but process has `qwen3`), mark deploy health **degraded** and surface in probe.

### 5.4 What mother consumes (agent-flow / RLM)

Precedence already intended on mother:

1. Explicit Agent-node / request override (`enable_thinking`, `preserve_thinking`)
2. Catalog `reasoning` / flat `preserve_thinking` / `generation_defaults.chat_template_kwargs`
3. Last-resort name heuristic (temporary; remove once catalog coverage is high)

Runtime duties (mother, not node):

- Persist full CoT in session/UI.
- Trim outbound history per `history.policy` + `replay_max_assistant_turns`.
- Pass `enable_param` / `preserve_thinking_param` on the chat body.
- Parse responses using `output_fields` then `inline_tags` / `channel_markers`.

---

## 6. Safe defaults when fields are missing

| Missing | Safe default for RLM / chat agents |
|---------|-------------------------------------|
| `supported` unset | Treat as `false` for auto-enable; still scrape `<think>` if present |
| `mode=unknown` | Do not force thinking on |
| `history.policy=unknown` | Cap wire replay to **2** recent assistant turns; do **not** set `preserve_thinking=true` |
| `intensity.kind` unset | `none` |
| Flat `preserve_thinking` unset | Do not invent `true` |

**Anti-pattern:** defaulting `preserve_thinking=true` globally. That is correct for many Qwen agent setups and harmful for Gemma4 long runs.

---

## 7. Worked examples

### 7.1 `gemma4-31` (external OpenAI-compat / vLLM Gemma4)

```json
{
  "reasoning": {
    "supported": true,
    "mode": "hybrid",
    "wire": {
      "family": "gemma4",
      "enable_param": "chat_template_kwargs.enable_thinking",
      "enable_default": false,
      "output_fields": ["reasoning", "reasoning_content"],
      "vllm_reasoning_parser": "gemma4",
      "channel_markers": ["<|channel>thought"]
    },
    "history": {
      "policy": "tool_chain_only",
      "preserve_thinking_param": "chat_template_kwargs.preserve_thinking",
      "preserve_thinking_default": false,
      "replay_max_assistant_turns": 2,
      "requires_reasoning_on_assistant_replay": false
    },
    "intensity": { "kind": "none" }
  },
  "capabilities": {
    "supports_reasoning": true,
    "preserve_thinking": false
  }
}
```

### 7.2 `qwen3-5-9b` / `qwen3-8-27` (vLLM Qwen3 family)

```json
{
  "reasoning": {
    "supported": true,
    "mode": "hybrid",
    "wire": {
      "family": "qwen3",
      "enable_param": "chat_template_kwargs.enable_thinking",
      "enable_default": true,
      "output_fields": ["reasoning_content", "reasoning"],
      "vllm_reasoning_parser": "qwen3",
      "inline_tags": ["<think>", "</think>"],
      "tool_call_parser": "qwen3_coder"
    },
    "history": {
      "policy": "preserve_all",
      "preserve_thinking_param": "chat_template_kwargs.preserve_thinking",
      "preserve_thinking_default": true,
      "replay_max_assistant_turns": 8,
      "requires_reasoning_on_assistant_replay": false
    },
    "intensity": { "kind": "none" }
  },
  "capabilities": {
    "supports_reasoning": true,
    "preserve_thinking": true
  }
}
```

---

## 8. Validation rules

1. If `supported=false` → `mode` must be `none`; `wire.family` must be `none`; history policy should be `strip`.
2. If `supported=true` → `wire.family` ≠ `none`; `output_fields` non-empty.
3. If `history.preserve_thinking_param` is set → `preserve_thinking_default` should be non-null.
4. `replay_max_assistant_turns`: required when `supported=true` (recommend 2 for strip/tool_chain_only, 8 for preserve_all).
5. Reject unknown enum values at write time (node + Storage).

---

## 9. Migration plan

1. **Schema:** accept `metadata.reasoning` on Storage write; project on catalog.
2. **Alias:** keep writing `capabilities.supports_reasoning` + `capabilities.preserve_thinking`.
3. **Backfill:** for existing cards (`gemma4-31`, `qwen3-*`), run a one-shot enricher from parser/family heuristics (same matrix as §5.2). Current cards incorrectly mark some chat models as `supports_reasoning=false`.
4. **Mother:** prefer structured `reasoning`; keep flat alias readers.
5. **Remove** name-substring fallbacks once ≥95% of chat cards have `reasoning.mode ≠ unknown`.

---

## 10. Non-goals

- Teaching the model *what* to think (that is system prompt / RLM doctrine).
- Replacing compaction / context budgets (history policy is a CoT policy, not a full context manager).
- Guaranteeing bit-identical CoT across provider switches (cross-model handoff may flatten thinking to text).

---

## 11. Implementation checklist for `aiv2_node`

- [ ] Extend inspect/deploy metadata writer with `reasoning` object (§3).
- [ ] Detect from vLLM args + template family (§5.2).
- [ ] Keep catalog aliases in sync (§4).
- [ ] Align serve flags with stamped parser/defaults (§5.3).
- [ ] Expose `reasoning` on node→mother model registration payload.
- [ ] Add unit tests: Qwen vs Gemma vs non-reasoning cards.
- [ ] Document override UI (admin) for manual `history.policy` / `supported` corrections.

---

## 12. References (behavioral)

- vLLM Reasoning Outputs — per-family `--reasoning-parser`, `enable_thinking` / `thinking` kwargs.
- Gemma 4 chat template — `preserve_thinking` default **false**; strip prior-turn thoughts except in active tool chain.
- Qwen3 thinking docs — hybrid `enable_thinking`; `preserve_thinking` for multi-turn / tool-call continuity.
- Prime Agent `thinkingFormat`: `openai` | `deepseek` | `zai` | `qwen` | `qwen-chat-template` (map into `wire.family` / enable param).
