import os
from dotenv import load_dotenv
from .bedrock_runtime import BedrockInvoker, format_user_message, extract_text

load_dotenv()

invoker = BedrockInvoker(aws_region=os.getenv("AWS_REGION"))

def summarize_credit_profile(prompt: str) -> str:
    """
    Generates a short LLM summary for a credit profile using Bedrock.
    Uses inference profile if configured; otherwise falls back to modelId.
    """
    system = "You are a credit analyst helping users understand their credit risk briefly and clearly."
    response = invoker.invoke_messages(
        messages=[format_user_message(prompt)],
        system_prompt=system,
        max_tokens=200,
    )
    txt = extract_text(response)
    if not txt:
        raise RuntimeError("Empty response from Bedrock/Anthropic.")
    return txt
