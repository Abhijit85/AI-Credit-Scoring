"""Generate a synthetic credit dataset with required event metadata.

This script creates a CSV file that includes the features used by the
application as well as the event metadata fields required by AWS Fraud
Detector (EVENT_TIMESTAMP, EVENT_LABEL, ENTITY_ID, EVENT_ID, ENTITY_TYPE,
and LABEL_TIMESTAMP). Column names for feature data remain lowercase while
metadata columns are uppercase to match Fraud Detector conventions.
"""

from faker import Faker
import csv
import random
import os
import uuid
from datetime import timezone

faker = Faker()
credit_mix = ["good", "fair", "poor"]
loan_types = ["auto", "home", "education", "business", "none"]

rows = 1_000_000  # ≈300 MB output (adjust as needed)
output_file = "credit-training.csv"

base_columns = [
    "name", "age", "occupation", "annual_income", "monthly_inhand_salary",
    "num_bank_accounts", "num_credit_card", "interest_rate", "num_of_loan",
    "type_of_loan", "delay_from_due_date", "num_of_delayed_payment", "credit_mix",
    "outstanding_debt", "credit_utilization_ratio", "credit_history_age",
    "total_emi_per_month"
]

metadata_columns = [
    "EVENT_TIMESTAMP", "EVENT_LABEL", "ENTITY_ID", "EVENT_ID", "ENTITY_TYPE", "LABEL_TIMESTAMP"
]

header = base_columns + [col for col in metadata_columns if col not in base_columns]

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for _ in range(rows):
        loans = random.sample(loan_types, k=random.randint(1, 2))
        event_dt = faker.date_time_between("-1y", "now", tzinfo=timezone.utc)
        event_timestamp = event_dt.isoformat()
        label = random.choices(["legit", "fraud"], weights=[0.8, 0.2])[0]
        label_dt = faker.date_time_between(start_date=event_dt, end_date="now", tzinfo=timezone.utc)
        label_timestamp = label_dt.isoformat()
        entity_id = faker.ssn()
        event_id = uuid.uuid4().hex
        entity_type = "customer"

        writer.writerow([
            faker.name(),
            random.randint(21, 60),
            faker.job(),
            random.randint(30000, 150000),
            random.randint(2000, 10000),
            random.randint(1, 7),
            random.randint(1, 6),
            random.randint(5, 20),
            random.randint(0, 3),
            "|".join(l for l in loans if l != "none") or "none",
            random.randint(0, 15),
            random.randint(0, 5),
            random.choice(credit_mix),
            random.randint(1000, 20000),
            round(random.uniform(20, 80), 1),
            f"{random.randint(1, 15)} Years and {random.randint(0, 11)} Months",
            random.randint(300, 900),
            event_timestamp,
            label,
            entity_id,
            event_id,
            entity_type,
            label_timestamp,
        ])

print(f"{output_file} created, size ≈ {os.path.getsize(output_file)/1_000_000:.1f} MB")
