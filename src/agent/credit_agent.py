"""Agentic credit-analyst orchestration.

Implements the workshop loop end to end:

    retrieve (RAG)  ->  reason  ->  explain  ->  write-back

* **retrieve** - embed the applicant, vector-search similar past *decisions*
  (long-term memory) and relevant *policies* (grounding).
* **reason**   - deterministic rule-based features feed the score; AgentCore
  holds the working session.
* **explain**  - Bedrock (Claude) writes a plain-language, *cited* rationale;
  a deterministic fallback is used if the LLM is unavailable.
* **write-back** - persist the decision + embedding so the next evaluation is
  smarter. This closing loop is the demo's punchline.

Every external dependency degrades gracefully, so the whole loop runs offline
for rehearsal and tests.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.memory.embeddings import active_provider, embed_text
from src.memory.long_term import LongTermMemory, get_memory

from .session import SessionMemory, get_session_memory


# --------------------------------------------------------------------------- #
# Deterministic, auditable feature scoring (reused by API + seed script)
# --------------------------------------------------------------------------- #
def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def compute_features(profile: Dict[str, Any]) -> Dict[str, int]:
    """Rule-based, explainable feature components (0-30 each)."""
    repayment = max(0, 30 - _to_int(profile.get("Num_of_Delayed_Payment"))
                    - _to_int(profile.get("Delay_from_due_date")) // 10)
    utilization = max(0, 30 - _to_float(profile.get("Credit_Utilization_Ratio")) // 3)
    outstanding = max(0, 30 - _to_float(profile.get("Outstanding_Debt")) / 1000)
    inquiries = max(0, 30 - _to_int(profile.get("Num_Credit_Card")))
    credit_score = min(850, 500 + repayment + utilization + outstanding + inquiries)
    return {
        "repayment": int(repayment),
        "utilization": int(utilization),
        "outstanding": int(outstanding),
        "inquiries": int(inquiries),
        "credit_score": round(credit_score),
    }


def band_for(score: int) -> str:
    if score >= 720:
        return "Approve"
    if score >= 640:
        return "Review"
    return "Decline"


def applicant_narrative(profile: Dict[str, Any]) -> str:
    return (
        f"{profile.get('Name', 'Applicant')} is a {profile.get('Age', '?')}-year-old "
        f"{profile.get('Occupation', 'worker')} with annual income "
        f"{profile.get('Annual_Income', '?')}. Credit utilization "
        f"{profile.get('Credit_Utilization_Ratio', '?')}%, "
        f"{profile.get('Num_of_Delayed_Payment', '?')} delayed payments, "
        f"outstanding debt {profile.get('Outstanding_Debt', '?')}, "
        f"{profile.get('Num_Credit_Card', '?')} credit cards, "
        f"credit history age {profile.get('Credit_History_Age', '?')}."
    )


def _deterministic_rationale(profile: Dict[str, Any], features: Dict[str, int],
                             band: str, similar: List[Dict[str, Any]],
                             policies: List[Dict[str, Any]]) -> str:
    strengths, concerns = [], []
    if _to_float(profile.get("Credit_Utilization_Ratio")) <= 30:
        strengths.append("credit utilization is within a healthy range")
    else:
        concerns.append("credit utilization is elevated")
    if _to_int(profile.get("Num_of_Delayed_Payment")) <= 2:
        strengths.append("payment history shows few delays")
    else:
        concerns.append("multiple delayed payments on record")
    if _to_float(profile.get("Outstanding_Debt")) > 5000:
        concerns.append("outstanding debt is high")

    cited = ", ".join(s.get("applicant_id", s.get("_id", "?")) for s in similar[:2]) or "none on file"
    pol = policies[0].get("policy_id") if policies else "n/a"
    return (
        f"### Summary\n"
        f"Estimated score {features['credit_score']} — decision band **{band}**.\n\n"
        f"### Key Strengths\n- " + ("\n- ".join(strengths) or "limited positive signals") + "\n\n"
        f"### Areas of Concern\n- " + ("\n- ".join(concerns) or "no material concerns") + "\n\n"
        f"### Grounding\n- Comparable prior applicants: {cited}\n- Policy applied: {pol}\n\n"
        f"### Recommendations\n- Keep utilization under 30% and avoid new delayed payments."
    )


def _llm_rationale(profile: Dict[str, Any], features: Dict[str, int], band: str,
                   similar: List[Dict[str, Any]], policies: List[Dict[str, Any]]) -> Optional[str]:
    """Try the Bedrock LLM with retrieved context; return None on any failure."""
    try:
        from src.llm.service import summarize_credit_profile
    except Exception:
        return None

    similar_txt = "\n".join(
        f"- {s.get('applicant_id', s.get('_id'))}: band {s.get('band')}, "
        f"score {s.get('credit_score')}" for s in similar[:3]
    ) or "- none on file"
    policy_txt = "\n".join(
        f"- {p.get('policy_id')}: {p.get('text', '')[:160]}" for p in policies[:2]
    ) or "- no specific policy"

    prompt = f"""
Applicant profile:
- Name: {profile.get('Name')}
- Age: {profile.get('Age')}
- Occupation: {profile.get('Occupation')}
- Annual Income: {profile.get('Annual_Income')}
- Credit Utilization Ratio: {profile.get('Credit_Utilization_Ratio')}
- Delayed Payments: {profile.get('Num_of_Delayed_Payment')}
- Outstanding Debt: {profile.get('Outstanding_Debt')}

Rule-based score: {features['credit_score']} (decision band: {band}).

Comparable past decisions retrieved from long-term memory:
{similar_txt}

Relevant lending policy retrieved:
{policy_txt}

Write a concise markdown risk assessment with headings Summary, Key Strengths,
Areas of Concern, and Recommendations. Explicitly reference the comparable cases
and the policy you were given. Do not ask the user any questions.
"""
    try:
        return summarize_credit_profile(prompt)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[agent] LLM rationale failed ({exc}); using deterministic fallback")
        return None


class CreditAgent:
    def __init__(self, memory: Optional[LongTermMemory] = None,
                 session: Optional[SessionMemory] = None) -> None:
        self.memory = memory or get_memory()
        self.session = session or get_session_memory()

    def evaluate(self, profile: Dict[str, Any], top_k: int = 3,
                 store: bool = True) -> Dict[str, Any]:
        sid = self.session.create_session()
        try:
            # 1. Reason: deterministic features
            features = compute_features(profile)
            band = band_for(features["credit_score"])
            self.session.remember(sid, "features", features)

            # 2. Retrieve (RAG): embed + vector search over memory + policies
            narrative = applicant_narrative(profile)
            query_vec = embed_text(narrative)
            similar = self.memory.similar_decisions(
                query_vec, k=top_k, exclude_applicant=str(profile.get("ssn") or profile.get("Name"))
            )
            policies = self.memory.similar_policies(query_vec, k=2)
            self.session.remember(sid, "retrieved", {"similar": len(similar), "policies": len(policies)})

            # 3. Explain: cited rationale (LLM, deterministic fallback)
            rationale = _llm_rationale(profile, features, band, similar, policies)
            used_llm = rationale is not None
            if rationale is None:
                rationale = _deterministic_rationale(profile, features, band, similar, policies)

            recommendations = self._recommendations(profile)

            result = {
                "status": "ok",
                "applicant_id": str(profile.get("ssn") or profile.get("Name")),
                "credit_score_estimate": features["credit_score"],
                "band": band,
                "repayment": features["repayment"],
                "utilization": features["utilization"],
                "outstanding": features["outstanding"],
                "inquiries": features["inquiries"],
                "summary": rationale,
                "recommendations": recommendations
                or ["Maintain current credit habits for gradual improvement."],
                "similar_cases": [
                    {
                        "id": s.get("_id"),
                        "applicant_id": s.get("applicant_id"),
                        "band": s.get("band"),
                        "credit_score": s.get("credit_score"),
                        "score": s.get("score"),
                    }
                    for s in similar
                ],
                "policies_cited": [
                    {"policy_id": p.get("policy_id"), "title": p.get("title")} for p in policies
                ],
                "meta": {
                    "embedding_provider": active_provider(),
                    "memory_backend": self.memory.backend,
                    "session_backend": self.session.backend,
                    "reasoning": "bedrock-llm" if used_llm else "deterministic-fallback",
                },
            }

            # 4. Write-back: persist decision + embedding for next time
            if store:
                record = dict(profile)
                record.update({
                    "applicant_id": result["applicant_id"],
                    "credit_score": features["credit_score"],
                    "band": band,
                    "summary": rationale,
                    "recommendations": recommendations,
                    **{k: features[k] for k in ("repayment", "utilization", "outstanding", "inquiries")},
                })
                decision_id = self.memory.store_decision(record, embedding=query_vec)
                result["decision_id"] = decision_id

            return result
        finally:
            self.session.close(sid)

    @staticmethod
    def _recommendations(profile: Dict[str, Any]) -> List[str]:
        recs = []
        if _to_float(profile.get("Credit_Utilization_Ratio")) > 35:
            recs.append("Reduce credit utilization below 30%")
        if _to_int(profile.get("Num_of_Delayed_Payment")) > 3:
            recs.append("Avoid delayed payments by enabling auto-pay")
        if _to_float(profile.get("Outstanding_Debt")) > 5000:
            recs.append("Consolidate loans if outstanding debt is high")
        return recs


_DEFAULT: Optional[CreditAgent] = None


def get_agent() -> CreditAgent:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = CreditAgent()
    return _DEFAULT
