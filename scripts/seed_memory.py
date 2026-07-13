"""Seed long-term memory with synthetic applicants, decisions and policies.

Generates a realistic-but-fake population so that vector retrieval returns
meaningful neighbours on stage. Safe to run repeatedly. Uses Faker (already a
dependency). NO real customer data.

Usage:
    python scripts/seed_memory.py --count 30

Works with or without MongoDB configured — without MONGODB_URI it exercises the
in-memory store (useful for a dry run).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / "backend" / ".env", override=True)

from src.agent.credit_agent import applicant_narrative, band_for, compute_features  # noqa: E402
from src.memory.embeddings import active_provider, embed_text  # noqa: E402
from src.memory.long_term import LongTermMemory  # noqa: E402

OCCUPATIONS = ["Teacher", "Engineer", "Nurse", "Analyst", "Driver", "Designer",
               "Clerk", "Developer", "Manager", "Technician", "Student", "Chef"]


def _make_applicant(faker, idx: int) -> dict:
    age = random.randint(21, 62)
    income = random.randint(28000, 180000)
    util = round(random.uniform(5, 85), 1)
    delayed = random.randint(0, 8)
    outstanding = random.randint(0, 22000)
    return {
        "Name": faker.name(),
        "ssn": f"SEED-{idx:04d}",
        "Age": str(age),
        "Occupation": random.choice(OCCUPATIONS),
        "Annual_Income": str(income),
        "Monthly_Inhand_Salary": str(round(income / 13, 2)),
        "Num_Bank_Accounts": str(random.randint(1, 5)),
        "Num_Credit_Card": str(random.randint(0, 8)),
        "Interest_Rate": str(random.randint(5, 30)),
        "Num_of_Loan": str(random.randint(0, 4)),
        "Type_of_Loan": random.choice(["Auto", "Personal", "Student", "Home", "None"]),
        "Delay_from_due_date": str(random.randint(0, 40)),
        "Num_of_Delayed_Payment": str(delayed),
        "Credit_Mix": random.choice(["Good", "Standard", "Bad"]),
        "Outstanding_Debt": str(outstanding),
        "Credit_Utilization_Ratio": str(util),
        "Credit_History_Age": f"{random.randint(0, 22)} Years",
        "Total_EMI_per_month": str(random.randint(0, 3000)),
    }


def seed(count: int) -> None:
    try:
        from faker import Faker
        faker = Faker()
    except Exception:
        class _F:  # minimal fallback if Faker missing
            def name(self):
                return random.choice(["Alex Doe", "Sam Lee", "Jordan Kim", "Riley Fox"])
        faker = _F()

    mem = LongTermMemory()
    print(f"Embedding provider: {active_provider()}")
    print(f"Memory backend: {mem.backend}")

    # Policies
    pol_path = Path(__file__).resolve().parent.parent / "data" / "policies.json"
    with pol_path.open() as f:
        policies = json.load(f)
    n_pol = mem.upsert_policies(policies)
    print(f"Upserted {n_pol} policies.")

    # Applicants -> decisions (write-back)
    for i in range(count):
        profile = _make_applicant(faker, i)
        features = compute_features(profile)
        band = band_for(features["credit_score"])
        record = dict(profile)
        record.update({
            "applicant_id": profile["ssn"],
            "credit_score": features["credit_score"],
            "band": band,
            "summary": f"Seed decision for {profile['Name']}: band {band}.",
            **{k: features[k] for k in ("repayment", "utilization", "outstanding", "inquiries")},
        })
        emb = embed_text(applicant_narrative(profile))
        mem.store_decision(record, embedding=emb)

    print(f"Seeded {count} decisions into '{mem.backend}' backend.")
    if mem.backend == "in-memory":
        print("NOTE: no MONGODB_URI set — data lived only for this process. "
              "Set MONGODB_URI to persist into Atlas.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=30)
    args = ap.parse_args()
    seed(args.count)
