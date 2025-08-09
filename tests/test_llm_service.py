import os
import sys

sys.path.append(os.getcwd())

from langchain_aws import ChatBedrock

from src.llm.service import llm


def test_llm_initialization():
    assert isinstance(llm, ChatBedrock)
    expected_model = os.getenv(
        "BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    )
    assert llm.model_id == expected_model

