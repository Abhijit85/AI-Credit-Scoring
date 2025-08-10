import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "cc_products.json"

# Load products and prepare vectorizer at import time
with _DATA_PATH.open() as f:
    _PRODUCTS = json.load(f)

_TEXTS = [p.get("text", "") for p in _PRODUCTS]
_VECTOR = TfidfVectorizer(stop_words="english")
_MATRIX = _VECTOR.fit_transform(_TEXTS)


def recommend_products(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """Return top_k product recommendations similar to the query.

    Args:
        query: Free-text description of customer needs.
        top_k: Number of recommendations to return.
    """
    if not query:
        return []
    q_vec = _VECTOR.transform([query])
    sims = cosine_similarity(q_vec, _MATRIX).ravel()
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
            }
        )
    return results
