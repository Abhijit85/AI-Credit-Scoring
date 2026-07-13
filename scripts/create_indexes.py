"""Create Atlas Vector Search indexes for the credit-scoring memory.

Requires an Atlas cluster (M10+ or a Search-enabled tier) and a MONGODB_URI.
Creates vector indexes on:

    decisions.embedding    (long-term memory retrieval)
    policies.embedding     (RAG grounding)
    cc_products.embedding  (product recommendations)

If your driver/cluster does not support programmatic search-index creation,
the script prints the JSON definitions so you can paste them into the Atlas UI
(Atlas Search -> Create Index -> JSON editor).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _definition(dim: int) -> dict:
    return {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": dim,
                "similarity": "cosine",
            }
        ]
    }


def main() -> None:
    from dotenv import load_dotenv

    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
    load_dotenv(root / "backend" / ".env", override=True)

    dim = int(_env("EMBED_DIM", "1024"))
    targets = [
        ("decisions", _env("DECISIONS_VECTOR_INDEX", "decisions_vector_index")),
        ("policies", _env("POLICIES_VECTOR_INDEX", "policies_vector_index")),
        ("cc_products", _env("PRODUCTS_VECTOR_INDEX", "products_vector_index")),
    ]
    definition = _definition(dim)

    uri = _env("MONGODB_URI")
    if not uri:
        print("No MONGODB_URI set. Paste these definitions into the Atlas UI:\n")
        for coll, name in targets:
            print(f"# collection: {coll}   index name: {name}")
            print(json.dumps(definition, indent=2))
            print()
        return

    from pymongo import MongoClient
    from pymongo.operations import SearchIndexModel

    client = MongoClient(uri)
    db = client[_env("MONGODB_DB", "bfsi-genai")]
    for coll, name in targets:
        try:
            model = SearchIndexModel(definition=definition, name=name, type="vectorSearch")
            db[coll].create_search_index(model=model)
            print(f"Requested vector index '{name}' on '{coll}' (build may take a minute).")
        except Exception as exc:
            print(f"Could not auto-create '{name}' on '{coll}': {exc}")
            print("Paste this into Atlas Search (JSON editor):")
            print(json.dumps(definition, indent=2))


if __name__ == "__main__":
    main()
