"""Credit-card product recommendations.

Two backends, selected automatically:

* **Atlas Vector Search** over Voyage/Bedrock embeddings of the product catalog
  (on-message with the workshop story) when MongoDB + embeddings are available.
* **TF-IDF cosine** over the local ``cc_products.json`` as an offline fallback
  so the endpoint always returns something during rehearsal and tests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine

_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "cc_products.json"

with _DATA_PATH.open() as f:
    _PRODUCTS = json.load(f)

_TEXTS = [p.get("text", "") for p in _PRODUCTS]
_VECTOR = TfidfVectorizer(stop_words="english")
_MATRIX = _VECTOR.fit_transform(_TEXTS)


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _vector_search_recommend(query: str, top_k: int) -> List[Dict[str, str]]:
    """Atlas $vectorSearch over the products collection. Raises on any problem
    so the caller can fall back to TF-IDF."""
    from pymongo import MongoClient

    from src.memory.embeddings import embed_text

    client = MongoClient(_env("MONGODB_URI"), serverSelectionTimeoutMS=4000)
    db = client[_env("MONGODB_DB", "bfsi-genai")]
    qvec = embed_text(query)
    pipeline = [
        {
            "$vectorSearch": {
                "index": _env("PRODUCTS_VECTOR_INDEX", "products_vector_index"),
                "path": "embedding",
                "queryVector": qvec,
                "numCandidates": max(50, top_k * 10),
                "limit": top_k,
            }
        },
        {"$project": {"title": 1, "text": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]
    docs = list(db["cc_products"].aggregate(pipeline))
    if not docs:
        raise RuntimeError("no vector results")
    return [
        {
            "title": d.get("title", "Unknown Product"),
            "description": d.get("text", ""),
            "score": round(d.get("score", 0), 4),
        }
        for d in docs
    ]


def _tfidf_recommend(query: str, top_k: int) -> List[Dict[str, str]]:
    q_vec = _VECTOR.transform([query])
    sims = _sk_cosine(q_vec, _MATRIX).ravel()
    if not np.any(sims):
        return []
    top_idxs = sims.argsort()[-top_k:][::-1]
    results: List[Dict[str, str]] = []
    for idx in top_idxs:
        prod = _PRODUCTS[idx]
        results.append(
            {
                "title": prod.get("title", "Unknown Product"),
                "description": prod.get("text", ""),
                "score": round(float(sims[idx]), 4),
            }
        )
    return results


def recommend_products(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    if not query:
        return []
    if _env("MONGODB_URI"):
        try:
            return _vector_search_recommend(query, top_k)
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[recommendations] vector search unavailable ({exc}); using TF-IDF")
    return _tfidf_recommend(query, top_k)
