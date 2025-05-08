from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
from pymongo import MongoClient
# from openai.embeddings_utils import get_embedding

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client["bfsi-genai"]
collection = db["user_profiles"]
products = db["cc_products"]  # this should contain the vector index and embeddings

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@app.get("/")
def read_root():
    return {"message": "AI Credit Scoring API is live."}

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

class QueryDescription(BaseModel):
    description: str

@app.post("/score")
def score_credit(input: CreditInput):
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
            gpt_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a credit analyst helping users understand their credit risk."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=200
            )
            explanation = gpt_response.choices[0].message.content.strip()
        except Exception as e:
            explanation = f"LLM summary failed: {str(e)}"

        recommendations = [
            "Reduce credit utilization below 30%" if float(input.Credit_Utilization_Ratio) > 35 else "",
            "Avoid delayed payments by enabling auto-pay" if int(input.Num_of_Delayed_Payment) > 3 else "",
            "Consolidate loans if outstanding debt is high" if float(input.Outstanding_Debt) > 5000 else ""
        ]
        recommendations = [r for r in recommendations if r]

        record = input.dict()
        record.update({
            "credit_score": round(credit_score),
            "repayment": int(repayment),
            "utilization": int(utilization),
            "outstanding": int(outstanding),
            "inquiries": int(inquiries),
            "summary": explanation,
            "recommendations": recommendations
        })
        print("Inserting into MongoDB:", record)

        collection.insert_one(record)

        return {
            "credit_score_estimate": round(credit_score),
            "repayment": int(repayment),
            "utilization": int(utilization),
            "outstanding": int(outstanding),
            "inquiries": int(inquiries),
            "summary": explanation,
            "recommendations": recommendations or ["Maintain current credit habits for gradual improvement."]
        }
    except Exception as e:
        return {"error": f"Something went wrong: {str(e)}"}

@app.post("/similar_products")
def similar_products(query: QueryDescription):
    try:
        embedding = client.embeddings.create(input=query.description, model="text-embedding-ada-002").data[0].embedding

        results = products.aggregate([
            {
                "$vectorSearch": {
                    "index": "product_embedding_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 100,
                    "limit": 3
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "title": 1,
                    "text": 1,
                    "source": 1
                }
            }
        ])

        return {"results": list(results)}
    except Exception as e:
        return {"error": f"Vector search failed: {str(e)}"}