"""Offline tests for the agentic credit loop.

These run with no cloud credentials: embeddings use the local fallback, memory
uses the in-memory store, reasoning uses the deterministic rationale.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force fully-offline backends regardless of ambient env.
os.environ["MONGODB_URI"] = ""
os.environ["EMBED_PROVIDER"] = "local"
os.environ["AGENT_SESSION_BACKEND"] = "local"

from src.agent.credit_agent import CreditAgent, band_for, compute_features  # noqa: E402
from src.memory.embeddings import cosine_similarity, embed_text  # noqa: E402
from src.memory.long_term import LongTermMemory  # noqa: E402


def _applicant(**over):
    base = {
        "Name": "Test User", "ssn": "T-0001", "Age": "23", "Occupation": "Analyst",
        "Annual_Income": "60000", "Monthly_Inhand_Salary": "4600",
        "Num_Bank_Accounts": "2", "Num_Credit_Card": "1", "Interest_Rate": "12",
        "Num_of_Loan": "0", "Type_of_Loan": "None", "Delay_from_due_date": "2",
        "Num_of_Delayed_Payment": "1", "Credit_Mix": "Good", "Outstanding_Debt": "1200",
        "Credit_Utilization_Ratio": "22", "Credit_History_Age": "2 Years",
        "Total_EMI_per_month": "200",
    }
    base.update(over)
    return base


def test_features_and_band_monotonic():
    strong = compute_features(_applicant(Num_of_Delayed_Payment="0",
                                         Credit_Utilization_Ratio="10", Outstanding_Debt="0"))
    weak = compute_features(_applicant(Num_of_Delayed_Payment="8",
                                       Credit_Utilization_Ratio="80", Outstanding_Debt="20000"))
    assert strong["credit_score"] > weak["credit_score"]
    assert band_for(strong["credit_score"]) in {"Approve", "Review", "Decline"}
    assert band_for(730) == "Approve"
    assert band_for(660) == "Review"
    assert band_for(600) == "Decline"


def test_embeddings_similarity_signal():
    a = embed_text("young analyst stable income low utilization")
    b = embed_text("young analyst stable income low utilization")
    c = embed_text("retired driver high debt many delayed payments")
    assert cosine_similarity(a, b) > cosine_similarity(a, c)
    assert len(a) == int(os.environ.get("EMBED_DIM", "1024"))


def test_evaluate_returns_structured_result():
    agent = CreditAgent(memory=LongTermMemory(uri=""))
    result = agent.evaluate(_applicant())
    assert result["status"] == "ok"
    for key in ("credit_score_estimate", "band", "summary", "similar_cases",
                "policies_cited", "meta", "decision_id"):
        assert key in result
    assert result["meta"]["memory_backend"] == "in-memory"
    assert result["meta"]["reasoning"] in {"deterministic-fallback", "bedrock-llm"}


def test_write_back_makes_next_retrieval_smarter():
    mem = LongTermMemory(uri="")
    agent = CreditAgent(memory=mem)

    first = agent.evaluate(_applicant(ssn="A-1"))
    # A second, similar applicant should now retrieve the first as a neighbour.
    second = agent.evaluate(_applicant(ssn="A-2", Name="Similar Person"))
    ids = [c["applicant_id"] for c in second["similar_cases"]]
    assert "A-1" in ids


def test_policies_are_retrieved_after_seeding():
    mem = LongTermMemory(uri="")
    mem.upsert_policies([
        {"policy_id": "young-saver-v3", "title": "Thin-file young applicants",
         "text": "young applicants stable income low utilization thin credit file"},
        {"policy_id": "debt-load-v1", "title": "Debt",
         "text": "outstanding debt high emi debt to income consolidation"},
    ])
    agent = CreditAgent(memory=mem)
    result = agent.evaluate(_applicant())
    assert len(result["policies_cited"]) >= 1
