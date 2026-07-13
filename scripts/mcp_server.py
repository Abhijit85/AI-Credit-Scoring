"""MongoDB MCP server exposing long-term memory as agent tools.

This is the "MongoDB MCP server exposed to the agent / Quick Desktop" piece of
the workshop architecture. It exposes three read tools over the credit memory:

    find_similar_applicants(description, k)  -> nearest past decisions
    search_policies(query, k)                -> relevant lending policies
    get_decision(applicant_id)               -> a specific stored decision

Run (stdio transport, e.g. from Quick Desktop / an MCP client):
    python scripts/mcp_server.py

Requires the `mcp` package:  pip install mcp
Falls back with a clear message if it is not installed.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.embeddings import embed_text  # noqa: E402
from src.memory.long_term import get_memory  # noqa: E402

try:
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover - optional dependency
    print("The 'mcp' package is required: pip install mcp", file=sys.stderr)
    raise SystemExit(1)

mcp = FastMCP("credit-memory")


@mcp.tool()
def find_similar_applicants(description: str, k: int = 3) -> list:
    """Find the most similar past credit decisions to a free-text applicant
    description, using vector search over long-term memory."""
    mem = get_memory()
    vec = embed_text(description)
    return mem.similar_decisions(vec, k=k)


@mcp.tool()
def search_policies(query: str, k: int = 3) -> list:
    """Retrieve lending policy snippets relevant to a query."""
    mem = get_memory()
    vec = embed_text(query)
    return mem.similar_policies(vec, k=k)


@mcp.tool()
def get_decision(applicant_id: str) -> dict:
    """Fetch a specific stored decision by applicant_id (e.g. SEED-0007)."""
    mem = get_memory()
    if mem.db is not None:
        doc = mem.db["decisions"].find_one({"applicant_id": applicant_id})
        if doc:
            return mem._clean(doc)
    for d in mem._mem.decisions:
        if d.get("applicant_id") == applicant_id:
            return mem._clean(d)
    return {"error": f"No decision found for {applicant_id}"}


if __name__ == "__main__":
    mcp.run()
