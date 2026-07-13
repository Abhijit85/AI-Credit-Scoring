# Agentic Credit Intelligence

A full-stack credit-scoring system, rebuilt as an **agentic credit analyst** with
durable memory. It shows the workshop architecture end to end:

> **MongoDB Atlas** = long-term memory + vector search &nbsp;·&nbsp;
> **AWS AgentCore** = short-term session memory &nbsp;·&nbsp;
> **Amazon Bedrock (Claude)** = reasoning &nbsp;·&nbsp;
> **Voyage AI** = embeddings &nbsp;·&nbsp;
> **MongoDB MCP server** = the agent's tool to query data

Built with FastAPI (backend), React + Tailwind (frontend), and scikit-learn
(offline fallback).

## The agent loop

Every `/score` call runs four steps:

1. **Retrieve (RAG)** — embed the applicant (Voyage), vector-search similar past
   **decisions** (long-term memory) and relevant **policies** in MongoDB Atlas.
2. **Reason** — deterministic, auditable rule-based features produce the score;
   AgentCore holds the working session.
3. **Explain** — Bedrock (Claude) writes a plain-language, **cited** rationale
   referencing the retrieved cases and policy.
4. **Write-back** — the decision + its embedding are persisted, so the next
   evaluation is smarter. *This closing loop is the point.*

```
User → React UI → FastAPI /score
                     │
                     ├─ rule-based screening gate (validators.py)
                     ▼
              CreditAgent.evaluate()
                     │  retrieve ─→ reason ─→ explain ─→ write-back
        ┌────────────┼───────────────────────────┐
        ▼            ▼                             ▼
   Voyage embed   Bedrock (Claude)          MongoDB Atlas
                                       decisions · policies · cc_products
                                       (Vector Search + system of record)
```

### Runs anywhere — graceful fallbacks

Every cloud dependency degrades so the demo can be rehearsed (and tested) with
**no credentials at all**:

| Layer | Primary | Fallback chain |
|-------|---------|----------------|
| Embeddings | Voyage AI | → Bedrock Titan → offline hashed local embedding |
| Long-term memory | Atlas `$vectorSearch` | → in-Python cosine scan → in-memory store |
| Session memory | AWS AgentCore | → local in-process session |
| Reasoning | Bedrock Claude | → deterministic rationale |
| Recommendations | Atlas Vector Search | → TF-IDF over `data/cc_products.json` |

The `meta` block in each `/score` response reports which backend was actually
used, e.g. `{"embedding_provider": "local", "memory_backend": "in-memory", ...}`.

## Project layout

```
backend/main.py               FastAPI app; /score runs the agent loop
backend/validators.py         rule-based screening gate
src/memory/embeddings.py      Voyage → Bedrock → local embeddings
src/memory/long_term.py       Atlas long-term memory + vector search
src/agent/session.py          AgentCore short-term session memory
src/agent/credit_agent.py     retrieve → reason → explain → write-back
src/recommendations/service.py vector-search (fallback TF-IDF) product recs
src/llm/service.py            Bedrock Claude wrapper
data/policies.json            lending policies for RAG grounding
scripts/seed_memory.py        seed synthetic applicants + decisions + policies
scripts/create_indexes.py     create Atlas vector-search indexes
scripts/mcp_server.py         MongoDB MCP server (memory tools)
tests/                        offline tests for the whole loop
```

## Quickstart

### 1. Backend
```bash
cd backend
cp .env.example .env      # fill in your values (see below)
pip install -r requirements.txt
uvicorn main:app --reload --app-dir ..
```
> Launched from the repo root instead? `uvicorn backend.main:app --reload`

Try it with **no configuration** — it will use the offline fallbacks:
```bash
curl -s localhost:8000/health
```

### 2. Seed memory (so retrieval has neighbours)
```bash
python scripts/seed_memory.py --count 30    # persists to Atlas if MONGODB_URI is set
```

### 3. Create Atlas vector indexes (Atlas only)
```bash
python scripts/create_indexes.py            # or paste the printed JSON into the Atlas UI
```

### 4. MongoDB MCP server (optional; for Quick Desktop / MCP clients)
```bash
pip install mcp
python scripts/mcp_server.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev        # http://localhost:5173/
```

## Configuration

Copy `.env.example` to `backend/.env`. Key variables:

```bash
# MongoDB Atlas
MONGODB_URI=              # mongodb+srv://... ; blank => in-memory store
MONGODB_DB=bfsi-genai

# Embeddings (auto-detects: Voyage → Bedrock → local)
EMBED_PROVIDER=auto       # auto | voyage | bedrock | local
VOYAGE_API_KEY=
VOYAGE_MODEL=voyage-3
EMBED_DIM=1024

# AWS / Bedrock
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_SESSION_TOKEN=        # optional
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=us.anthropic.claude-3-7-sonnet-20250219-v1:0
BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0

# AgentCore short-term memory (auto ⇒ AgentCore if AGENTCORE_MEMORY_ID set)
AGENT_SESSION_BACKEND=auto
AGENTCORE_MEMORY_ID=

# Atlas vector index names
DECISIONS_VECTOR_INDEX=decisions_vector_index
POLICIES_VECTOR_INDEX=policies_vector_index
PRODUCTS_VECTOR_INDEX=products_vector_index
```

> ⚠️ **Security:** never commit real secrets. `backend/.env` is gitignored; only
> `.env.example` ships. If a credential was ever committed, rotate it.

### Bedrock inference profiles

Some Anthropic models (e.g. Claude 3.5 Haiku) can't be invoked on-demand and
require an **inference profile** ID/ARN. Set `BEDROCK_MODEL_ID` to the profile
ID (e.g. `us.anthropic.claude-3-5-haiku-20241022-v1:0`) and `AWS_REGION` to the
hosting region. On-demand models can be set directly as `BEDROCK_MODEL_ID`.

## API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Reports which memory/session backends are active |
| `POST /score` | Runs the agent loop; returns score, band, `similar_cases`, `policies_cited`, `summary`, `meta` |
| `POST /similar_products` | Vector-search (fallback TF-IDF) product recommendations |

## Tests
```bash
pip install pytest httpx
pytest -q         # runs fully offline via the local fallbacks
```

## Demo tip

Run `seed_memory.py` first, then score a thin-file applicant twice: the first
run retrieves seeded neighbours, the second run retrieves the decision the first
run just wrote back — visibly "smarter" the second time. That write-back loop is
the on-stage moment.
