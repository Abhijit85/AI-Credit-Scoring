import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

llm = ChatBedrock(
    model=os.getenv(
        "BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    ),
    temperature=0.7,
    max_tokens=2048,
    streaming=False,
    region_name=os.getenv("AWS_REGION", "us-east-1"),
)


def summarize_credit_profile(prompt: str) -> str:
    """Generate a short LLM summary for a credit profile using Bedrock."""

    messages = [
        SystemMessage(
            content=(
                "You are a credit analyst helping users understand their credit risk "
                "briefly and clearly."
            )
        ),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    return response.content
