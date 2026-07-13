"""Long-term memory for the credit-scoring agent, backed by MongoDB Atlas.

This is the durable memory layer from the workshop architecture:

    * ``decisions``  - every credit decision + rationale + embedding (write-back)
    * ``policies``   - lending policy snippets used for RAG grounding

Retrieval uses Atlas ``$vectorSearch`` when a vector index is available and
falls back to an in-Python cosine scan otherwise, so the same code runs against
a full Atlas cluster on stage or a plain local MongoDB (or no MongoDB at all)
during development.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .embeddings import cosine_similarity, embed_text

try:  # pymongo is optional for the pure-offline path
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
except Exception:  # pragma: no cover
    MongoClient = None  # type: ignore
    PyMongoError = Exception  # type: ignore


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


class _InMemoryStore:
    """Minimal stand-in used when no MONGODB_URI is configured."""

    def __init__(self) -> None:
        self.decisions: List[Dict[str, Any]] = []
        self.policies: List[Dict[str, Any]] = []


class LongTermMemory:
    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None) -> None:
        self.uri = uri if uri is not None else _env("MONGODB_URI")
        self.db_name = db_name or _env("MONGODB_DB", "bfsi-genai")
        self._mem = _InMemoryStore()
        self.client = None
        self.db = None
        if self.uri and MongoClient is not None:
            try:
                self.client = MongoClient(self.uri, serverSelectionTimeoutMS=4000)
                self.db = self.client[self.db_name]
            except Exception as exc:  # pragma: no cover - network dependent
                print(f"[long_term] MongoDB connection failed ({exc}); using in-memory store")
                self.client = None
                self.db = None

    @property
    def backend(self) -> str:
        return "mongodb" if self.db is not None else "in-memory"

    # ------------------------------------------------------------------ #
    # Write-back
    # ------------------------------------------------------------------ #
    def store_decision(self, record: Dict[str, Any], embedding: Optional[List[float]] = None) -> str:
        doc = dict(record)
        doc.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        if embedding is None:
            embedding = embed_text(self._decision_text(doc))
        doc["embedding"] = embedding
        if self.db is not None:
            try:
                res = self.db["decisions"].insert_one(doc)
                return str(res.inserted_id)
            except PyMongoError as exc:  # pragma: no cover
                print(f"[long_term] insert failed ({exc}); using in-memory store")
        doc.setdefault("_id", f"mem-{len(self._mem.decisions) + 1}")
        self._mem.decisions.append(doc)
        return str(doc["_id"])

    # ------------------------------------------------------------------ #
    # Retrieval (RAG)
    # ------------------------------------------------------------------ #
    def similar_decisions(self, embedding: List[float], k: int = 3,
                          exclude_applicant: Optional[str] = None) -> List[Dict[str, Any]]:
        docs = self._vector_search("decisions", _env("DECISIONS_VECTOR_INDEX", "decisions_vector_index"),
                                    embedding, k, self._mem.decisions)
        if exclude_applicant:
            docs = [d for d in docs if d.get("applicant_id") != exclude_applicant]
        return docs[:k]

    def similar_policies(self, embedding: List[float], k: int = 3) -> List[Dict[str, Any]]:
        return self._vector_search("policies", _env("POLICIES_VECTOR_INDEX", "policies_vector_index"),
                                   embedding, k, self._mem.policies)

    # ------------------------------------------------------------------ #
    # Policy loading (for seeding)
    # ------------------------------------------------------------------ #
    def upsert_policies(self, policies: List[Dict[str, Any]]) -> int:
        count = 0
        for pol in policies:
            doc = dict(pol)
            if "embedding" not in doc:
                doc["embedding"] = embed_text(doc.get("text", ""))
            if self.db is not None:
                try:
                    self.db["policies"].update_one(
                        {"policy_id": doc.get("policy_id")}, {"$set": doc}, upsert=True
                    )
                    count += 1
                    continue
                except PyMongoError as exc:  # pragma: no cover
                    print(f"[long_term] policy upsert failed ({exc}); using in-memory store")
            self._mem.policies.append(doc)
            count += 1
        return count

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _vector_search(self, collection: str, index_name: str, embedding: List[float],
                       k: int, fallback_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.db is not None:
            try:
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": index_name,
                            "path": "embedding",
                            "queryVector": embedding,
                            "numCandidates": max(50, k * 10),
                            "limit": k,
                        }
                    },
                    {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                ]
                results = list(self.db[collection].aggregate(pipeline))
                if results:
                    return [self._clean(d) for d in results]
            except PyMongoError as exc:  # pragma: no cover
                print(f"[long_term] $vectorSearch on '{collection}' unavailable "
                      f"({exc}); falling back to cosine scan")
            # Fallback: pull docs and score in Python (works on any MongoDB)
            try:
                docs = list(self.db[collection].find({"embedding": {"$exists": True}}).limit(500))
                return self._cosine_rank(docs, embedding, k)
            except PyMongoError:  # pragma: no cover
                pass
        return self._cosine_rank(fallback_docs, embedding, k)

    def _cosine_rank(self, docs: List[Dict[str, Any]], embedding: List[float],
                     k: int) -> List[Dict[str, Any]]:
        scored = []
        for d in docs:
            emb = d.get("embedding")
            if not emb:
                continue
            score = cosine_similarity(embedding, emb)
            item = self._clean(d)
            item["score"] = round(score, 4)
            scored.append(item)
        scored.sort(key=lambda x: x.get("score", 0), reverse=True)
        return scored[:k]

    @staticmethod
    def _clean(doc: Dict[str, Any]) -> Dict[str, Any]:
        out = {k: v for k, v in doc.items() if k != "embedding"}
        if "_id" in out:
            out["_id"] = str(out["_id"])
        return out

    @staticmethod
    def _decision_text(doc: Dict[str, Any]) -> str:
        parts = [
            doc.get("Name", ""),
            f"age {doc.get('Age', '')}",
            f"occupation {doc.get('Occupation', '')}",
            f"income {doc.get('Annual_Income', '')}",
            f"utilization {doc.get('Credit_Utilization_Ratio', '')}",
            f"delayed payments {doc.get('Num_of_Delayed_Payment', '')}",
            f"outstanding {doc.get('Outstanding_Debt', '')}",
            f"band {doc.get('band', '')}",
            doc.get("summary", ""),
        ]
        return " ".join(str(p) for p in parts if p)


# Convenience singleton for the API layer
_DEFAULT: Optional[LongTermMemory] = None


def get_memory() -> LongTermMemory:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = LongTermMemory()
    return _DEFAULT
