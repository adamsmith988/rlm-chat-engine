# LLM Routing Contract for Tools & Plugins

**Audience:** authors of new tools under `modules/tools/` and plugins under `modules/plugins/`.  
**Status:** normative. Violating this contract is a bug.  
**Gateway implementation:** `services/tool-runner/src/routes/internal_llm.py`  
**Client helpers:** `modules/tools/tool_runner_llm_gateway.py`, `modules/tools/tool_binding_resolver.py`

---

## 1. Architecture (non-negotiable)

```
┌─────────────────────┐
│  Tool / Plugin code │  ← you write this
└──────────┬──────────┘
           │ HTTP only to tool-runner /internal/llm/*
           │ headers: X-AIv2-Tool-Id / X-AIv2-Plugin-Id
           │ body.model = "auto"  OR  an explicit logical_id
           ▼
┌─────────────────────┐
│     tool-runner     │  internal LLM gateway
│  /internal/llm/*    │  • rewrite model via binding
│                     │  • validate modality profile
└──────────┬──────────┘
           │ LLM_PROXY_URL (owned ONLY by tool-runner)
           ▼
┌─────────────────────┐
│  llm-proxy (LB)     │  resolve logical_id → backend pool
│  catalog + health   │  failover / GPU affinity / etc.
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Inference backends │  vLLM / Omni / Triton / TEI / Comfy / ASR …
│  (node IPs, ports)  │  NEVER addressed by tools/plugins
└─────────────────────┘
```

### Separation of concerns

| Layer | Owns | Must not own |
|-------|------|--------------|
| **Tool / plugin** | Pre/post-processing, I/O channels, calling `/internal/llm/*` | `LLM_PROXY_URL`, node IP/host, `inference_url`, Docker ports, backend selection |
| **Binding (admin)** | Map `tool.<id>.*` / `plugin.<id>.*` → **logical_id** | Physical host, port, runtime container |
| **tool-runner gateway** | Auth to internal routes, binding rewrite (`auto` → logical_id), forward | Business logic of tools |
| **llm-proxy** | Catalog, health, load-balance **logical_id → backend(s)** | Tool-specific preprocessing |

**Binding is only a link** between a tool/plugin slot and a logical model id.  
**Backend assignment** (which node, which IP, which port) is exclusively the load balancer’s job.

---

## 2. Hard prohibitions

Do **not** do any of the following in `modules/tools/**` or `modules/plugins/**` (except archived code):

1. Read or construct `LLM_PROXY_URL` / `llm-gateway:8005` / `llm-proxy:8005`.
2. HTTP to llm-proxy paths (`/v1/chat/completions`, `/v1/infer`, `/catalog`, `/binding/resolve`, …).
3. Use catalog `inference_url`, `node_id`, or any LAN/WAN IP:port for inference.
4. Hard-code a physical model server name as if it were a deploy address (e.g. dialing `http://192.168.x.x:8100`).
5. Import `llm_proxy_base()` from legacy helpers — it raises by design.
6. Bypass tool-runner “for speed” or “because I know the URL”.

If a capability is missing on `/internal/llm/*`, **extend the gateway** in tool-runner; do not open a side door from the tool.

---

## 3. Allowed client surface

### 3.1 Base URL

Inside the tool-runner process (normal case):

```text
TOOL_LLM_GATEWAY_URL  (default http://127.0.0.1:8002)
```

Always go through helpers — do not hand-roll absolute llm-proxy URLs.

```python
from modules.tools.tool_runner_llm_gateway import (
    AUTO_MODEL,
    chat_completions_url,
    query_rewrite_url,
    embeddings_url,
    images_generations_url,
    infer_url,
    rerank_url,
    audio_transcriptions_url,
    audio_diarize_url,
    audio_voiceprints_url,
    catalog_url,
    binding_resolve_url,
    comfy_url,
    internal_gateway_headers,
)
```

### 3.2 Gateway routes (tool-runner)

| Client helper | Method | Path | Upstream (tool-runner only) | Binding `purpose` |
|---------------|--------|------|-----------------------------|-------------------|
| `chat_completions_url()` | POST | `/internal/llm/chat/completions` | `/v1/chat/completions` | `chat` |
| `query_rewrite_url()` | POST | `/internal/llm/query_rewrite` | `/v1/chat/completions` | `query_rewrite` |
| `embeddings_url()` | POST | `/internal/llm/embeddings` | `/v1/embeddings` | `embed` |
| `images_generations_url()` | POST | `/internal/llm/images/generations` | `/v1/images/generations` | `image` |
| `infer_url()` | POST | `/internal/llm/infer` | `/v1/infer` | `infer` |
| `rerank_url()` | POST | `/internal/llm/rerank` | `/rerank` | `rerank` |
| `audio_transcriptions_url()` | POST | `/internal/llm/audio/transcriptions` | `/v1/audio/transcriptions` | `asr` |
| `audio_diarize_url()` | POST | `/internal/llm/audio/diarize` | `/v1/speech/diarize` | `diarize` |
| `audio_voiceprints_url()` | POST | `/internal/llm/audio/voiceprints` | `/v1/audio/voiceprints` | `voiceprint` |
| `catalog_url()` | GET | `/internal/llm/catalog` | `/catalog` | — |
| `binding_resolve_url()` | GET | `/internal/llm/binding/resolve` | `/binding/resolve` | — |
| `comfy_url(path)` | * | `/internal/llm/comfy/{path}` | `/v1/comfy/{path}` | model = logical_id (query/body) |

Streaming (`stream: true`) is **not** supported on the gateway; send `stream: false`.

### 3.3 Required headers

`internal_gateway_headers()` attaches:

| Header | Source | Role |
|--------|--------|------|
| `Content-Type: application/json` | helper (omit for multipart) | JSON bodies |
| `X-Tool-Runner-Internal-Secret` | `TOOL_RUNNER_INTERNAL_SECRET` | Auth when secret is set |
| `X-AIv2-Tool-Id` | `tool_call_context` | Binding lookup |
| `X-AIv2-Plugin-Id` | `tool_call_context` | Binding lookup |

tool-runner sets `tool_call_context(tool_id, plugin_id)` around every tool/plugin `run()`. You normally do **not** set these headers manually if you use the helper.

For multipart ASR/voiceprints:

```python
headers=internal_gateway_headers(content_type=None)
```

---

## 4. Model identity: `logical_id` vs binding vs backend

### 4.1 Terms

| Term | Meaning |
|------|---------|
| **logical_id** | Caller-facing model name in the catalog (e.g. `z-image-turbo`, `ornith-1-0-9b`). Stable across nodes. |
| **Binding route key** | Admin mapping, e.g. `tool.embed_image.infer` → `dinov2`. |
| **Physical / backend** | Adapter key + base URL inside llm-proxy (`openai_compat:…`, host:port). Invisible to tools. |
| **`AUTO_MODEL` (`"auto"`)** | Sentinel: gateway resolves binding → logical_id before forward. |

### 4.2 Binding resolution chain

When `model` is missing / `""` / `"auto"` / `"default"`, the gateway resolves (first hit wins):

```text
tool.<tool_id>.llm
tool.<tool_id>.<purpose>
plugin.<plugin_id>.llm
plugin.<plugin_id>.<purpose>
```

`<purpose>` comes from the gateway path (`chat`, `embed`, `image`, `infer`, `rerank`, `asr`, `diarize`, `voiceprint`, …).

Admin configures bindings via api-gateway, e.g.:

```http
PUT /admin/tools/{tool_id}/binding
PUT /admin/plugins/{plugin_id}/binding
```

If no binding exists and the client sent `auto`, the gateway returns **503** `binding_not_configured` with `tried_chain`. That is expected — configure the binding; do not hard-code a backend URL.

### 4.3 Explicit logical_id

Allowed when the UI or flow pins a catalog model:

```json
{ "model": "z-image-turbo", "prompt": "…" }
```

The gateway still validates the model against the catalog/modality profile, then llm-proxy load-balances that logical pool.  
**Never** pass an `inference_url` alongside it.

### 4.4 Prefer `auto` for ordinary tools

```python
payload = {
    "model": AUTO_MODEL,  # or omit "model"
    "input": text,
}
resp = httpx.post(
    embeddings_url(),
    headers=internal_gateway_headers(),
    json=payload,
    timeout=120.0,
)
```

Optional explicit resolve (still via tool-runner, not llm-proxy):

```python
from modules.tools.tool_binding_resolver import resolve_bound_model

logical_id, route_key, constraints = resolve_bound_model(
    route_suffix="infer",
    explicit_model=None,  # use binding
)
```

---

## 5. Protocol patterns (copy-paste)

### 5.1 Chat (OpenAI-compatible)

```python
from modules.tools.tool_runner_llm_gateway import (
    AUTO_MODEL,
    chat_completions_url,
    internal_gateway_headers,
)

body = {
    "model": AUTO_MODEL,
    "messages": [{"role": "user", "content": prompt}],
    "stream": False,
}
r = httpx.post(chat_completions_url(), headers=internal_gateway_headers(), json=body, timeout=300.0)
r.raise_for_status()
```

### 5.2 Embeddings

```python
from modules.tools.tool_runner_llm_gateway import AUTO_MODEL, embeddings_url, internal_gateway_headers

body = {"model": AUTO_MODEL, "input": text}
r = httpx.post(embeddings_url(), headers=internal_gateway_headers(), json=body, timeout=120.0)
```

### 5.3 Image generation (Omni / OpenAI images)

```python
from modules.tools.tool_runner_llm_gateway import images_generations_url, internal_gateway_headers

# Prefer catalog logical_id from UI, or AUTO_MODEL with image binding
body = {"model": "z-image-turbo", "prompt": prompt}
r = httpx.post(images_generations_url(), headers=internal_gateway_headers(), json=body, timeout=600.0)
```

### 5.4 Triton / tensor infer

```python
from modules.tools.tool_runner_llm_gateway import AUTO_MODEL, infer_url, internal_gateway_headers

body = {
    "model": AUTO_MODEL,
    "inputs": [{"name": "pixel_values", "shape": [...], "datatype": "FP32", "data": data_b64}],
    "outputs": ["embedding"],
}
r = httpx.post(infer_url(), headers=internal_gateway_headers(), json=body, timeout=60.0)
```

### 5.5 Rerank

```python
from modules.tools.tool_runner_llm_gateway import AUTO_MODEL, rerank_url, internal_gateway_headers

body = {"model": AUTO_MODEL, "query": query, "texts": documents}
r = httpx.post(rerank_url(), headers=internal_gateway_headers(), json=body, timeout=60.0)
```

### 5.6 ASR (multipart)

```python
from modules.tools.tool_runner_llm_gateway import (
    AUTO_MODEL,
    audio_transcriptions_url,
    internal_gateway_headers,
)

data = {"model": AUTO_MODEL, "response_format": "verbose_json"}
files = {"file": (path.name, path.open("rb"), mime)}
r = httpx.post(
    audio_transcriptions_url(),
    headers=internal_gateway_headers(content_type=None),
    data=data,
    files=files,
    timeout=300.0,
)
```

### 5.6b Diarize (multipart)

```python
from modules.tools.tool_runner_llm_gateway import (
    AUTO_MODEL,
    audio_diarize_url,
    internal_gateway_headers,
)

data = {"model": AUTO_MODEL}
files = {"file": (path.name, path.open("rb"), mime)}
r = httpx.post(
    audio_diarize_url(),
    headers=internal_gateway_headers(content_type=None),
    data=data,
    files=files,
    timeout=300.0,
)
```

### 5.6c Voiceprints (multipart)

```python
from modules.tools.tool_runner_llm_gateway import (
    AUTO_MODEL,
    audio_voiceprints_url,
    internal_gateway_headers,
)

data = {"model": AUTO_MODEL, "return_segment_embeddings": "true"}
files = {"file": (path.name, path.open("rb"), mime)}
r = httpx.post(
    audio_voiceprints_url(),
    headers=internal_gateway_headers(content_type=None),
    data=data,
    files=files,
    timeout=300.0,
)
```

### 5.7 ComfyUI (logical_id only)

```python
from modules.tools.tool_runner_llm_gateway import comfy_url, internal_gateway_headers

logical_id = "transformer"  # catalog logical_id — never a host URL
r = httpx.post(
    comfy_url("prompt") + f"?model={logical_id}",
    headers=internal_gateway_headers(),
    json={"model": logical_id, "prompt": graph, "client_id": client_id},
    timeout=600.0,
)
# history / view / object_info likewise via comfy_url(...)?model=<logical_id>
```

### 5.8 Catalog (pickers)

```python
from modules.tools.tool_runner_llm_gateway import catalog_url, internal_gateway_headers

r = httpx.get(catalog_url(), headers=internal_gateway_headers(), timeout=30.0)
# Use entry["logical_id"] only. Ignore any inference_url / node fields if present in payloads.
```

---

## 6. Agent-flow / plugins

Plugins run under the same contract. Agent-flow specifically:

- Model nodes pass **logical_id** from the catalog picker.
- Dispatch goes to `/internal/llm/*` (chat, images, embeddings, infer, rerank, ASR, Comfy).
- Node-saved `inference_url` / `node_id` are stripped and must not be reintroduced.
- Plugin id header: `X-AIv2-Plugin-Id: agent_flow` (see `modules/plugins/agent_flow/_shared.py` → `internal_llm_headers()`).

Plugin-scoped bindings:

```text
plugin.<plugin_id>.llm
plugin.<plugin_id>.<purpose>
```

Example: `plugin.map_explorer.geoclip_infer`.

### 6.1 Sanctioned exception: RLM streaming hot path

The RLM loop (`FlowRunner._internal_chat_stream_once`) runs **inside** the
tool-runner process. Bouncing every SSE turn through
`tool_runner_base()/internal/llm/chat/completions` made that same uvicorn proxy a
request to itself (double-hop) while also serving UI/progress HTTP. Phase 1
therefore lets **only** that path stream directly to llm-gateway
`/v1/chat/completions`:

- URL: `_shared.rlm_llm_direct_base()` → `AGENT_FLOW_RLM_LLM_URL`, else
  `LLM_PROXY_URL` (already in the tool-runner container env), else loopback.
  `AGENT_FLOW_RLM_LLM_URL=off` forces the legacy gateway path.
- Binding: `rlm_runtime/llm_direct.py` replicates the gateway chain
  (`tool.plugin_agent_flow_run.*` → `plugin.agent_flow.*`) via
  `GET /binding/resolve` and forwards `X-AIv2-Route-Key`. Still logical_id only —
  never `inference_url` / node IPs; llm-gateway keeps load-balancing backends.
- If the binding cannot be resolved, the call falls back to the loopback gateway
  so `503 binding_not_configured` etc. are unchanged.

Everything else (Model nodes, tool-call agents, non-stream fallback, all
`modules/tools/*`) remains on `/internal/llm/*` — §2 applies to them unchanged.

---

## 7. Adding a new capability

If your tool needs an LLM API that is not listed in §3.2:

1. **Add** a forwarder in `services/tool-runner/src/routes/internal_llm.py`.
2. **Add** a URL helper in `modules/tools/tool_runner_llm_gateway.py`.
3. **Wire** a binding `purpose` and modality profile in `llm_model_validator` / taxonomy as needed.
4. **Call** only the new `/internal/llm/...` helper from the tool.
5. **Do not** teach the tool `LLM_PROXY_URL`.

---

## 8. Checklist for PR review

- [ ] No `LLM_PROXY_URL`, `llm_proxy_base()`, or hard-coded `llm-gateway` / `llm-proxy` hosts in the tool/plugin.
- [ ] No `inference_url`, raw `http://IP:port`, or node agent URLs used for inference.
- [ ] All model HTTP goes through `tool_runner_llm_gateway` (or agent-flow `tool_runner_base()` + `/internal/llm/*`).
- [ ] `model` is `AUTO_MODEL` / omitted, or an explicit **logical_id** from catalog.
- [ ] Admin binding documented (`tool.<id>.<purpose>` or `plugin.<id>.<purpose>`).
- [ ] Multipart calls use `internal_gateway_headers(content_type=None)`.
- [ ] Errors mention binding / gateway, not “unreachable node IP”.

---

## 9. Failure modes (expected)

| Symptom | Meaning | Fix |
|---------|---------|-----|
| `503 binding_not_configured` | No route for tool/plugin + purpose | `PUT /admin/tools\|plugins/.../binding` |
| `400` modality / profile mismatch | Bound logical_id does not fit tool I/O profile | Rebind to a compatible catalog model |
| `502 llm_proxy_unreachable` | tool-runner cannot reach llm-proxy | Ops: llm-gateway service / `LLM_PROXY_URL` on tool-runner only |
| `All connection attempts failed` to a LAN IP | Tool bypassed gateway or stale catalog URL leaked | Remove direct URL; use logical_id path |
| Images/chat work but wrong node | LB chose another healthy backend in the pool | Expected; deactivate stale backends in registry if needed |

---

## 10. Reference implementations

| Pattern | Location |
|---------|----------|
| Chat via gateway | `modules/tools/graph_extract/tool.py`, `vision_info/tool.py` |
| Embeddings via gateway | `modules/tools/embed_text/tool.py` |
| Images via gateway | `modules/tools/image_generate/tool.py` |
| Infer via gateway | `modules/tools/embed_image/tool.py`, `embed_face/tool.py` |
| Rerank via gateway | `modules/tools/text_search/tool.py` |
| ASR via gateway | `modules/tools/audio_transcribe/tool.py` |
| Comfy / images / catalog (plugin) | `modules/plugins/agent_flow/model_dispatch.py` |
| Gateway forwarders | `services/tool-runner/src/routes/internal_llm.py` |

---

## 11. One-sentence rule

> **Tools and plugins request models only through tool-runner’s `/internal/llm/*` using a binding or a catalog `logical_id`; llm-proxy alone load-balances to backends. Never dial IPs, nodes, or `LLM_PROXY_URL` from tool/plugin code.**
