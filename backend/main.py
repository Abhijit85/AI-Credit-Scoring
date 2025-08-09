from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
import json
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from validators import evaluate_rules

from src.llm.service import summarize_credit_profile
from src.llm.bedrock_runtime import BedrockInvoker, format_user_message, extract_text

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")

llm = BedrockInvoker(aws_region=AWS_REGION, api_key=os.getenv("BEDROCK_API_KEY"))

mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client["bfsi-genai"]
collection = db["user_profiles"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "AI Credit Scoring API is live."}

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


def invoke_scoring_service(profile: dict, service: str = os.getenv("MODEL_SERVICE", "frauddetector")):
    """Send profile data to the configured AWS service and return the result."""
    try:
        if service == "frauddetector":
            fd = boto3.client("frauddetector")
            response = fd.get_event_prediction(
                detectorId=os.getenv("FRAUD_DETECTOR_NAME"),
                eventId=profile.get("Name", "event"),
                eventTypeName=os.getenv("FRAUD_DETECTOR_EVENT_TYPE"),
                eventVariables={k: str(v) for k, v in profile.items()},
            )
            model_scores = response.get("modelScores", [])
            score = None
            if model_scores:
                scores = model_scores[0].get("scores", {})
                score = scores.get("anomalyScore") or next(iter(scores.values()), None)
            return {"anomaly_score": score}
        elif service == "sagemaker-runtime":
            sm = boto3.client("sagemaker-runtime")
            response = sm.invoke_endpoint(
                EndpointName=os.getenv("SAGEMAKER_ENDPOINT_NAME"),
                ContentType="application/json",
                Body=json.dumps(profile),
            )
            body = response["Body"].read()
            try:
                payload = json.loads(body)
            except Exception:
                payload = body.decode("utf-8")
            return {"recommendation": payload}
    except Exception as e:
        return {"service_error": str(e)}
    return {}

@app.post("/score")
def score_credit(input: CreditInput):
    profile = input.dict()
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

    try:
        repayment = max(0, 30 - int(input.Num_of_Delayed_Payment) - int(input.Delay_from_due_date) // 10)
        utilization = max(0, 30 - float(input.Credit_Utilization_Ratio) // 3)
        outstanding = max(0, 30 - float(input.Outstanding_Debt) / 1000)
        inquiries = max(0, 30 - int(input.Num_Credit_Card))
        credit_score = min(850, 500 + repayment + utilization + outstanding + inquiries)

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

        try:
            explanation = summarize_credit_profile(summary_prompt)
        except Exception as e:
            explanation = f"LLM summary failed: {str(e)}"

        recommendations = [
            "Reduce credit utilization below 30%" if float(input.Credit_Utilization_Ratio) > 35 else "",
            "Avoid delayed payments by enabling auto-pay" if int(input.Num_of_Delayed_Payment) > 3 else "",
            "Consolidate loans if outstanding debt is high" if float(input.Outstanding_Debt) > 5000 else ""
        ]
        recommendations = [r for r in recommendations if r]

        record = profile.copy()
        record.update({
            "credit_score": round(credit_score),
            "repayment": int(repayment),
            "utilization": int(utilization),
            "outstanding": int(outstanding),
            "inquiries": int(inquiries),
            "summary": explanation,
            "recommendations": recommendations
        })

        service_result = invoke_scoring_service(record)
        record.update(service_result)
        print("Inserting into MongoDB:", record)

        collection.insert_one(record)

        return {
            "status": "ok",
            "credit_score_estimate": round(credit_score),
            "repayment": int(repayment),
            "utilization": int(utilization),
            "outstanding": int(outstanding),
            "inquiries": int(inquiries),
            "summary": explanation,
            "recommendations": recommendations or ["Maintain current credit habits for gradual improvement."],
            **service_result,
        }
    except Exception as e:
        return {"error": f"Something went wrong: {str(e)}"}

@app.post("/similar_products")
def similar_products(query: QueryDescription):
    try:
        prompt = (
            f"Customer description: {query.description}\n"
            "Suggest three relevant credit card products in JSON format with 'title' and 'description'."
        )
        response = llm.invoke_messages(
            messages=[format_user_message(prompt)],
            system_prompt="You are a helpful financial assistant recommending credit card products.",
            max_tokens=200,
        )
        suggestions = extract_text(response).strip()
        return {"results": suggestions}
    except Exception as e:
        return {"error": f"Text generation failed: {str(e)}"}
