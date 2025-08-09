import os
import sys
sys.path.append(os.getcwd())
import pytest
from src.llm.bedrock_runtime import BedrockInvoker, format_user_message

@pytest.mark.unit
def test_requires_configuration(monkeypatch):
    # Ensure neither profile nor modelId is set
    monkeypatch.delenv("BEDROCK_TEXT_INFERENCE_PROFILE_ID", raising=False)
    monkeypatch.delenv("BEDROCK_TEXT_INFERENCE_PROFILE_ARN", raising=False)
    monkeypatch.delenv("BEDROCK_TEXT_MODEL_ID", raising=False)
    inv = BedrockInvoker(aws_region=os.getenv("AWS_REGION", "us-west-2"))
    with pytest.raises(RuntimeError):
        inv.invoke_messages(messages=[format_user_message("hi")], max_tokens=1)

@pytest.mark.unit
def test_prefers_profile_when_available(monkeypatch):
    monkeypatch.setenv("BEDROCK_TEXT_INFERENCE_PROFILE_ID", "us.anthropic.claude-3-5-haiku-20241022-v1:0")
    monkeypatch.setenv("BEDROCK_TEXT_MODEL_ID", "anthropic.claude-v2:1")
    inv = BedrockInvoker(aws_region=os.getenv("AWS_REGION", "us-west-2"))
    # private method allowed in unit test context
    kwargs = inv._build_invoke_kwargs('{"foo":"bar"}')
    if "inferenceProfileId" in inv._params_supported:
        assert kwargs.get("inferenceProfileId") == "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        assert kwargs.get("modelId") == "anthropic.claude-v2:1"
    else:
        # Older boto3 versions don't expose inference profile parameters and we
        # fall back to modelId to keep invocations valid.
        assert kwargs.get("modelId") == "anthropic.claude-v2:1"


@pytest.mark.unit
def test_profile_requires_model_id(monkeypatch):
    monkeypatch.setenv("BEDROCK_TEXT_INFERENCE_PROFILE_ID", "profile-id")
    monkeypatch.delenv("BEDROCK_TEXT_MODEL_ID", raising=False)
    inv = BedrockInvoker(aws_region=os.getenv("AWS_REGION", "us-west-2"))
    with pytest.raises(RuntimeError):
        inv._build_invoke_kwargs('{}')
