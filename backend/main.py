"""FastAPI entrypoint for the agentic credit-scoring service.

The `/score` endpoint runs the full agent loop (retrieve -> reason -> explain ->
write-back). The response stays backward compatible with the original API and
adds the new agentic fields: `band`, `similar_cases`, `policies_cited`, and
`meta` (which backends are actually in play).
"""
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Make the project root importable whether launched from repo root or backend/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.validators import evaluate_rules  # noqa: E402
from src.agent.credit_agent import get_agent  # noqa: E402
from src.recommendations.service import recommend_products  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / "backend" / ".env", override=True)

if os.getenv("AWS_PROFILE") == "":
    os.environ.pop("AWS_PROFILE", None)

app = FastAPI(title="AI Credit Scoring API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "AI Credit Scoring API is live.", "version": "2.0.0"}


@app.get("/health")
def health():
    agent = get_agent()
    return {
        "status": "ok",
        "memory_backend": agent.memory.backend,
        "session_backend": agent.session.backend,
    }


class CreditInput(BaseModel):
    Name: str
    ssn: str
    Age: str
    Occupation: str
    Annual_Income: str
    Monthly_Inhand_Salary: str
    Num_Bank_Accounts: str
    Num_Credit_Card: str
    Interest_Rate: str
    Num_of_Loan: str
    Type_of_Loan: str
    Delay_from_due_date: str
    Num_of_Delayed_Payment: str
    Credit_Mix: str
    Outstanding_Debt: str
    Credit_Utilization_Ratio: str
    Credit_History_Age: str
    Total_EMI_per_month: str


class QueryDescription(BaseModel):
    description: str


@app.post("/score")
def score_credit(payload: CreditInput):
    profile = payload.dict()

    # Rule-based screening gate (unchanged) — hard rejects and flags short-circuit.
    profile["missing_fields"] = [k for k, v in profile.items() if v in (None, "")]
    screening = evaluate_rules(profile)
    profile.pop("missing_fields", None)
    if screening["status"] == "reject":
        return {
            "status": "rejected",
            "reason": screening["rule"],
            "description": screening["description"],
        }
    if screening["flags"]:
        return {"status": "flagged", "flags": screening["flags"]}

    # Full agent loop: retrieve -> reason -> explain -> write-back.
    try:
        return get_agent().evaluate(profile)
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"Something went wrong: {exc}"}


@app.post("/similar_products")
def similar_products(query: QueryDescription):
    """Credit-card recommendations for a free-text description (vector search)."""
    try:
        suggestions = recommend_products(query.description)
        return {"results": suggestions}
    except Exception as exc:
        return {"error": f"Product recommendation failed: {exc}"}
