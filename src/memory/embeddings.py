"""Embedding provider for the credit-scoring agent.

Primary provider is **Voyage AI** (on-message for the MongoDB GTM). It degrades
gracefully so the demo can be rehearsed — and the tests can run — with no cloud
credentials at all:

    Voyage  ->  AWS Bedrock (Titan)  ->  offline deterministic local embedding

The local fallback is a hashed bag-of-words projected into ``EMBED_DIM`` and
L2-normalised. It is *not* semantically deep, but tokens that overlap produce
higher cosine similarity, which is enough to make vector retrieval visibly work
on stage without a network connection.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from functools import lru_cache
from typing import List

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def embed_dim() -> int:
    try:
        return int(_env("EMBED_DIM", "1024"))
    except ValueError:
        return 1024


def _l2_normalise(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


# --------------------------------------------------------------------------- #
# Offline deterministic fallback
# --------------------------------------------------------------------------- #
def _local_embedding(text: str, dim: int) -> List[float]:
    """Deterministic hashed bag-of-words embedding (no network required)."""
    vec = [0.0] * dim
    tokens = _TOKEN_RE.findall((text or "").lower())
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign
    return _l2_normalise(vec)


# --------------------------------------------------------------------------- #
# Voyage AI
# --------------------------------------------------------------------------- #
def _voyage_embedding(text: str) -> List[float]:
    import voyageai  # imported lazily so it is an optional dependency

    client = voyageai.Client(api_key=_env("VOYAGE_API_KEY"))
    model = _env("VOYAGE_MODEL", "voyage-3")
    result = client.embed([text], model=model, input_type="document")
    return result.embeddings[0]


# --------------------------------------------------------------------------- #
# AWS Bedrock (Titan / Cohere embeddings)
# --------------------------------------------------------------------------- #
def _bedrock_embedding(text: str) -> List[float]:
    import json

    import boto3

    client = boto3.client("bedrock-runtime", region_name=_env("AWS_REGION", "us-east-1"))
    model_id = _env("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
    body = json.dumps({"inputText": text})
    resp = client.invoke_model(modelId=model_id, body=body)
    payload = json.loads(resp["body"].read())
    return payload["embedding"]


@lru_cache(maxsize=1)
def _resolve_provider() -> str:
    """Pick a provider once, based on config and what is actually reachable."""
    configured = _env("EMBED_PROVIDER", "auto").lower()
    if configured in ("voyage", "bedrock", "local"):
        return configured
    # auto-detect
    if _env("VOYAGE_API_KEY"):
        return "voyage"
    if _env("BEDROCK_EMBED_MODEL_ID") and (_env("AWS_ACCESS_KEY_ID") or _env("AWS_PROFILE")):
        return "bedrock"
    return "local"


def active_provider() -> str:
    return _resolve_provider()


def embed_text(text: str) -> List[float]:
    """Return an embedding for ``text`` using the best available provider.

    Any provider error falls back to the offline local embedding so a live demo
    never hard-fails on a transient cloud issue.
    """
    provider = _resolve_provider()
    dim = embed_dim()
    try:
        if provider == "voyage":
            return _voyage_embedding(text)
        if provider == "bedrock":
            return _bedrock_embedding(text)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[embeddings] provider '{provider}' failed ({exc}); using local fallback")
    return _local_embedding(text, dim)


def embed_many(texts: List[str]) -> List[List[float]]:
    return [embed_text(t) for t in texts]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
