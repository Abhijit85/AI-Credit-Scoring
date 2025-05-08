from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os

# Load OpenAI key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

class CreditInput(BaseModel):
    Name: str
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

@app.post("/score")
def score_credit(input: CreditInput):
    repayment = max(0, 30 - int(input.Num_of_Delayed_Payment) - int(input.Delay_from_due_date) // 10)
    utilization = max(0, 30 - float(input.Credit_Utilization_Ratio) // 3)
    outstanding = max(0, 30 - float(input.Outstanding_Debt) / 1000)
    inquiries = max(0, 30 - int(input.Num_Credit_Card))
    credit_score = min(850, 500 + repayment + utilization + outstanding + inquiries)

    # LLM-based summary
    summary_prompt = f"""
    Based on this financial profile:
    - Name: {input.Name}
    - Age: {input.Age}
    - Occupation: {input.Occupation}
    - Annual Income: {input.Annual_Income}
    - Credit Utilization Ratio: {input.Credit_Utilization_Ratio}
    - Number of Delayed Payments: {input.Num_of_Delayed_Payment}
    - Outstanding Debt: {input.Outstanding_Debt}

    Summarize their credit risk and recommend the most important steps to improve their score.
    """

    gpt_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a credit analyst helping users understand their credit risk."},
            {"role": "user", "content": summary_prompt}
        ],
        max_tokens=200
    )
    explanation = gpt_response['choices'][0]['message']['content'].strip()

    recommendations = [
        "Reduce credit utilization below 30%" if float(input.Credit_Utilization_Ratio) > 35 else "",
        "Avoid delayed payments by enabling auto-pay" if int(input.Num_of_Delayed_Payment) > 3 else "",
        "Consolidate loans if outstanding debt is high" if float(input.Outstanding_Debt) > 5000 else ""
    ]
    recommendations = [r for r in recommendations if r]

    return {
        "credit_score_estimate": round(credit_score),
        "repayment": int(repayment),
        "utilization": int(utilization),
        "outstanding": int(outstanding),
        "inquiries": int(inquiries),
        "summary": explanation,
        "recommendations": recommendations or ["Maintain current credit habits for gradual improvement."]
    }
