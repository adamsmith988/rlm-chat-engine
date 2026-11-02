# Hashcat presentation standard (GPU node → mother catalog)

Audience: node-2 / `aiv2_node` engineers who deploy hashcat and publish it into the mother catalog.

This document is the contract for **presenting** hashcat as a catalog model. Mother Agent Flow is a **courier**: it sends the Start-node file to whatever HTTP API you advertise. If the catalog and OpenAPI disagree, the courier will send the wrong body and the worker will reject it.

Give this file to the node team as-is. It is self-contained.

---

## 1. Mental model

Hashcat is **not** a chat model. Do not advertise OpenAI chat completions.

| Layer | Who owns it | What it must describe |
| --- | --- | --- |
| Worker HTTP API | Node (hashcat-worker) | Real request: method, path, `Content-Type`, field names |
| Model Introduction v2 | Node (inspect / activate) | Same API, in catalog form |
| Mother catalog | Mother (persists introduction) | Routing: `dispatch_key`, `inference_url`, `endpoint_path` |
| Agent Flow Model node | Mother | Copies the Start **file wire** onto that HTTP API |

**Rule:** the HTTP field that receives the archive bytes, the OpenAPI property name, and `run_schema.content_slots[].wire` must be the **same string**.

Today the live worker on node 2 requires:

```http
POST /internal/jobs
Content-Type: multipart/form-data
X-Internal-Key: <cluster internal key>

file=<binary archive>   # required
```

If the catalog instead says “JSON body with `archive_path` / `artifact_id`”, mother will not upload the file. The worker then returns FastAPI `422` (`body.file` required). That is a presentation bug, not a missing wire on the canvas.

---

## 2. Identity (fixed values)

Use these exact values on deploy / introduction. Do not substitute `chat`, `custom`, or `custom-library`.

| Field | Required value | Why |
| --- | --- | --- |
| `identity.logical_id` | `hashcat` | Catalog model name the canvas selects |
| `identity.canonical_model_key` | `hashcat` | Pool key across nodes |
| `identity.storage_path` | `system/hashcat` | Not a weight folder |
| `serve.runtime` | `hashcat` | Not `custom-library` |
| `serve.api` | `crack` | **Never** `chat` |
| `serve.api_protocol` | `hashcat` | Dispatch authority |
| `serve.dispatch_key` | `hashcat` | Agent Flow protocol key (not `chat`) |
| `serve.ui_profile` | `hashcat` | Canvas widgets |
| `serve.endpoint_path` | `/internal/jobs` | Joined onto `inference_url` |
| `io.input_modalities` | `["file"]` | Canvas file inlet (not text/chat) |
| `io.output_modalities` | `["text"]` | Mother waits for password / status text |
| `endpoint_capabilities.hashcat_worker` | `true` | Mother skips chat-style probes |

`artifacts.format` may be `other`. Do not invent a Hugging Face pipeline.

---

## 3. Introduction v2 (what mother must receive)

Emit a **full** introduction on inspect/activate. Do **not** leave `introduction.run_schema` null while a different `run_schema` exists on the side. Mother routing reads `serve.*` first; a null `serve.endpoint_path` makes the courier guess.

Minimum shape:

```json
{
  "schema_version": 2,
  "identity": {
    "logical_id": "hashcat",
    "display_name": "hashcat",
    "canonical_model_key": "hashcat",
    "storage_path": "system/hashcat"
  },
  "hub": {
    "pipeline_tag": null,
    "known_in_hf_catalog": false,
    "tags": ["hashcat", "crack"]
  },
  "artifacts": { "format": "other" },
  "io": {
    "input_modalities": ["file"],
    "output_modalities": ["text"],
    "confidence": 1.0,
    "needs_review": false,
    "usage_hint": "Encrypted archive file in; recovered password or job status text out."
  },
  "serve": {
    "runtime": "hashcat",
    "api": "crack",
    "api_protocol": "hashcat",
    "dispatch_key": "hashcat",
    "ui_profile": "hashcat",
    "endpoint_path": "/internal/jobs",
    "endpoint_capabilities": { "hashcat_worker": true },
    "mother_timeout_sec": 120
  },
  "run_schema": {
    "schema_version": 1,
    "family": "hashcat",
    "ui_profile": "hashcat",
    "api_protocol": "hashcat",
    "endpoint_path": "/internal/jobs",
    "request_encoding": "multipart",
    "input_modalities": ["file"],
    "output_modalities": ["text"],
    "content_slots": [
      {
        "name": "file",
        "wire": "file",
        "modality": "file",
        "required": true,
        "source": "wire_input",
        "encoding": "multipart"
      }
    ],
    "fields": [
      {
        "name": "file",
        "wire": "file",
        "type": "file",
        "media": "file",
        "value_kind": "file",
        "role": "io",
        "required": true,
        "label": "Encrypted archive",
        "accept": ".zip,.rar,.7z,*/*",
        "ui_group": "basic"
      },
      {
        "name": "attack_mode",
        "wire": "attack_mode",
        "type": "string",
        "role": "generation",
        "ui_group": "basic",
        "options": ["auto", "wordlist", "mask", "hybrid", "pipeline"],
        "default": "auto"
      },
      {
        "name": "wordlist_id",
        "wire": "wordlist_id",
        "type": "string",
        "role": "generation",
        "ui_group": "basic",
        "required": false
      },
      {
        "name": "mask",
        "wire": "mask",
        "type": "string",
        "role": "generation",
        "ui_group": "advanced",
        "required": false
      },
      {
        "name": "rules",
        "wire": "rules",
        "type": "string",
        "role": "generation",
        "ui_group": "advanced",
        "required": false
      },
      {
        "name": "distributed",
        "wire": "distributed",
        "type": "bool",
        "role": "generation",
        "ui_group": "advanced",
        "default": false
      }
    ],
    "recommended_defaults": {
      "attack_mode": "auto",
      "distributed": false
    }
  }
}
```

Notes on fields that are easy to get wrong:

- `run_schema.request_encoding` must be `"multipart"`. JSON is **not** the hashcat submit encoding.
- The file slot `type` must be `"file"`, never `"string"`. A string slot tells mother to send a path / `artifact_id` instead of bytes.
- Do not name the file slot `archive_path` unless the worker’s multipart part is also named `archive_path`. The current worker part is `file`.
- `io.needs_review` must be `false` once modalities are known. Do not ship the scanner stub *“I/O modalities unknown from model files”*.
- Copy the same `run_schema` into `introduction.run_schema`. Side-channel catalog-only schema is not enough if `serve.endpoint_path` is null.

---

## 4. Worker HTTP API (source of truth)

Mother joins `backends[].inference_url` + `serve.endpoint_path`.

Example:

```
inference_url  = http://192.168.0.112:8012
endpoint_path  = /internal/jobs
forward        = http://192.168.0.112:8012/internal/jobs
```

`inference_url` must be reachable **from mother**, not only from the node’s localhost.

### 4.1 Health

```http
GET /health
```

JSON, no auth required. Include GPU / hashcat version if available:

```json
{
  "ok": true,
  "status": "healthy",
  "gpu_available": true,
  "hashcat_version": "v7.1.2",
  "active_job_id": null,
  "queue_depth": 0
}
```

### 4.2 Submit job

```http
POST /internal/jobs
Content-Type: multipart/form-data
X-Internal-Key: <INTERNAL_API_KEY>
```

| Part | Required | Type | Notes |
| --- | --- | --- | --- |
| `file` | **yes** | binary upload | Encrypted zip / rar / 7z (or hash file). This is the Start-node wire. |
| `tool_id` | no | string | Ignore or echo; Model courier identity is `hashcat`, not `archive_crack`. |
| `attack_mode` | no | string | `auto` / `wordlist` / `mask` / … |
| `wordlist` | no | string | |
| `wordlist_id` | no | string | |
| `mask` | no | string | |
| `rules` | no | string | |
| `attack_plan` | no | string | JSON string if used |
| `user_id` | no | string | |
| `session_id` | no | string | |
| `distributed` | no | bool | |
| `node_ids` | no | string | Comma-separated if used |

**Do not require JSON.** A `Content-Type: application/json` body with `{ "file": { "artifact_id": "…" } }` is not a file upload. FastAPI `File(...)` will answer:

```json
{"detail":[{"type":"missing","loc":["body","file"],"msg":"Field required","input":null}]}
```

Optional JSON accept is allowed **in addition** to multipart, never instead of it, and only if OpenAPI documents it. Mother’s hashcat courier will use multipart.

### 4.3 Poll / list / cancel

```http
GET    /internal/jobs
GET    /internal/jobs/{job_id}
DELETE /internal/jobs/{job_id}
```

All `/internal/*` routes require `X-Internal-Key` matching the cluster key.

### 4.4 Job envelope (async)

Submit may return immediately:

```json
{
  "ok": true,
  "job_id": "j_…",
  "job": {
    "job_id": "j_…",
    "status": "queued"
  }
}
```

Pending statuses mother treats as in-flight: `queued`, `pending`, `running`, `accepted`, `submitted`, `created`, `starting`, or empty.

Terminal success: `completed`, `complete`, `success`, `succeeded`, `done`.

Terminal failure: `failed`, `error`, `cancelled`, `canceled`.

When the catalog output is text, put the recovered password (or a clear status string) in one of: `text`, `output`, `result`, `content`, `password`, `plaintext`. Mother polls `GET {endpoint}/{job_id}` until a text field exists.

`password_not_found` as `error_code` is a completed text result, not a courier crash.

### 4.5 OpenAPI

`GET /openapi.json` must match the real submit route:

- `POST /internal/jobs` → `multipart/form-data`
- required property: `file` with `contentMediaType: application/octet-stream`
- no required JSON `archive_path` / `artifact_id` unless those parts actually exist

Introduction `content_slots` is a projection of this OpenAPI. If they drift, the catalog is wrong.

---

## 5. What mother can courier today

Agent Flow Model nodes send **two** request `Content-Type`s:

| Encoding | Used for |
| --- | --- |
| `application/json` | chat, images, embeddings, TTS, rerank, Triton, Comfy, custom-library |
| `multipart/form-data` | ASR (`/v1/audio/transcriptions`) |

Hashcat must be advertised as **`dispatch_key=hashcat` + multipart**, analogous to ASR (file in, text out), **not** as custom-library JSON.

Do not expect mother to:

- call the `archive_crack` **tool** from a Model node
- POST JSON `artifact_id` and have the worker fetch MinIO itself (unless you document and implement that extra API)
- invent chat messages around a file-in model

---

## 6. Anti-patterns (do not ship)

| Anti-pattern | What happens on mother |
| --- | --- |
| `serve.api = "chat"` or `dispatch_key` omitted | Risk of chat completions; file wire dropped |
| `runtime = custom-library` | Forced JSON `/v1/custom/generate` courier |
| `content_slots.wire = "archive_path"` while HTTP part is `file` | Bytes never land in `file` |
| Field `type: "string"` + `value_kind: "file"` | Mother sends a path string / artifact id, not bytes |
| `introduction.run_schema = null` and `serve.endpoint_path = null` | Courier cannot know `/internal/jobs` from serve |
| `io.needs_review = true` with unknown-modality stub | Operator cannot trust I/O |
| Worker accepts only multipart, catalog implies JSON | `422 body.file required` |
| Advertising `artifact_id` as the file | Worker OpenAPI has no such field |

---

## 7. Acceptance checklist (node 2)

Before telling mother “hashcat is presented”:

1. `GET http://<node-ip>:8012/health` from **mother** returns `ok: true`.
2. `GET /openapi.json` shows `POST /internal/jobs` as **multipart**, required `file`.
3. This curl from mother succeeds (auth + a small encrypted zip):

```bash
curl -sS -H "X-Internal-Key: $INTERNAL_API_KEY" \
  -F "file=@sample.zip;type=application/octet-stream" \
  -F "attack_mode=auto" \
  "http://<node-ip>:8012/internal/jobs"
```

4. Introduction on activate matches section 3 (`dispatch_key=hashcat`, `api=crack`, `endpoint_path=/internal/jobs`, file slot named `file`, `type=file`, `request_encoding=multipart`).
5. Catalog row `backends[].inference_url` is that same `http://<node-ip>:8012` (no in-cluster `hashcat-worker` hostname unless mother actually routes there).
6. A JSON POST of `{"model":"hashcat"}` to `/internal/jobs` is **rejected** (missing `file`) — that proves you did not silently switch the contract to JSON without documenting it.

When those six pass, mother can courier a canvas file wire onto hashcat without guessing.
