# Rebuild changelog — agentic credit intelligence

Turned the linear credit-scoring pipeline into an agentic credit analyst with
durable memory, matching the workshop architecture.

## Added
- `src/memory/embeddings.py` — Voyage → Bedrock → offline-local embeddings.
- `src/memory/long_term.py` — MongoDB Atlas long-term memory + `$vectorSearch`
  (cosine-scan and in-memory fallbacks) over `decisions` and `policies`.
- `src/agent/session.py` — AWS AgentCore short-term session memory (local fallback).
- `src/agent/credit_agent.py` — the retrieve → reason → explain → write-back loop.
- `data/policies.json` — lending policies for RAG grounding.
- `scripts/seed_memory.py` — seed synthetic applicants/decisions/policies.
- `scripts/create_indexes.py` — create Atlas vector-search indexes (or print defs).
- `scripts/mcp_server.py` — MongoDB MCP server exposing memory as agent tools.
- `tests/test_agent_loop.py` — fully-offline tests incl. the write-back "smarter
  next time" behaviour.
- `.gitignore`, `CHANGES.md`.

## Changed
- `backend/main.py` — `/score` now runs the agent loop; added `/health`. Response
  is backward compatible and adds `band`, `similar_cases`, `policies_cited`, `meta`.
- `src/recommendations/service.py` — Atlas Vector Search with TF-IDF fallback.
- `backend/requirements.txt` — added pymongo, numpy, voyageai, pytest, httpx.
- `README.md` — resolved a leftover git merge conflict; documented the new
  architecture, fallbacks, and quickstart.

## Security
- Removed real OpenAI / MongoDB / AWS credentials that were committed in
  `backend/.env`; replaced with placeholders. **Rotate those keys.**
