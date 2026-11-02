#!/usr/bin/env python3
"""CPU embedding sidecar. Vectors are deterministic from tokens (no model download).

Used so Qdrant short/long-term memory works without a second GPU process.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

DIM = int(os.environ.get("EMBED_DIM", "384"))
MODEL = os.environ.get("EMBED_MODEL_SERVED_NAME", "memory-embed")


def embed_one(text: str) -> List[float]:
    vec = [0.0] * DIM
    for tok in str(text or "").lower().split():
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % DIM] += 1.0
        vec[(h // DIM) % DIM] += 0.35
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


class EmbedReq(BaseModel):
    input: str | List[str]
    model: str | None = None


app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "dim": DIM}


@app.post("/v1/embeddings")
def embeddings(req: EmbedReq):
    texts = req.input if isinstance(req.input, list) else [req.input]
    data = [
        {"object": "embedding", "index": i, "embedding": embed_one(t)}
        for i, t in enumerate(texts)
    ]
    return {"object": "list", "model": req.model or MODEL, "data": data}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "18001")))
