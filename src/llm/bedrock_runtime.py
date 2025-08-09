import json
import os
from typing import Dict, Any, Optional, Set
from urllib.parse import quote

import boto3
from botocore.config import Config
import requests

ANTHROPIC_VERSION = "bedrock-2023-05-31"


class BedrockInvoker:
    """
    Thin wrapper around Bedrock Runtime that supports:
      - Inference Profiles (preferred for Claude 3.5 Haiku)
      - Fallback to direct modelId if profile is not configured
      - Optional cross-region routing via targetModelRegion
    """
    def __init__(
        self,
        aws_region: Optional[str] = None,
        timeout_sec: int = 60,
        api_key: Optional[str] = None,
    ):
        self.aws_region = aws_region or os.getenv("AWS_REGION", "us-west-2")
        self.api_key = api_key or os.getenv("BEDROCK_API_KEY")
        self.bedrock = None
        self._params_supported: Set[str] = set()
        if not self.api_key:
            self.bedrock = boto3.client(
                "bedrock-runtime",
                region_name=self.aws_region,
                config=Config(read_timeout=timeout_sec, retries={"max_attempts": 3}),
            )
            op = self.bedrock.meta.service_model.operation_model("InvokeModel")
            self._params_supported = set(op.input_shape.members.keys())
        # Env-driven configuration
        self.model_id = os.getenv("BEDROCK_TEXT_MODEL_ID", "").strip()
        self.profile_id = os.getenv("BEDROCK_TEXT_INFERENCE_PROFILE_ID", "").strip()
        self.profile_arn = os.getenv("BEDROCK_TEXT_INFERENCE_PROFILE_ARN", "").strip()
        self.target_model_region = os.getenv("BEDROCK_TEXT_REGION", "").strip()

    def _build_invoke_kwargs(self, body_json: str) -> Dict[str, Any]:
        """
        Build kwargs for bedrock.invoke_model() choosing profile ARN/ID first,
        then falling back to modelId if no profile configured.
        """
        kwargs: Dict[str, Any] = {
            "contentType": "application/json",
            "accept": "application/json",
            "body": body_json,
        }

        # Prefer inference profile (ARN > ID) when provided
        if self.profile_arn:
            kwargs["inferenceProfileArn"] = self.profile_arn
        elif self.profile_id:
            kwargs["inferenceProfileId"] = self.profile_id
        elif self.model_id:
            # Fallback: on-demand modelId (only if profile not set)
            kwargs["modelId"] = self.model_id
        else:
            raise RuntimeError(
                "Bedrock not configured: set BEDROCK_TEXT_INFERENCE_PROFILE_ID (or ARN) "
                "for Claude 3.5 Haiku, or BEDROCK_TEXT_MODEL_ID for on-demand models."
            )

        # Optional cross-region target if provided
        if self.target_model_region:
            kwargs["targetModelRegion"] = self.target_model_region

        return kwargs

    def invoke_messages(
        self,
        messages: list,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Invoke Anthropic chat-style messages API on Bedrock.
        """
        payload: Dict[str, Any] = {
            "anthropic_version": ANTHROPIC_VERSION,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if extra:
            payload.update(extra)

        body_json = json.dumps(payload)
        if self.api_key:
            # Prefer an inference profile (ARN or ID) when available for higher
            # throughput before falling back to a direct model invocation.
            if self.profile_arn:
                identifier = quote(self.profile_arn, safe="")
                url = (
                    f"https://bedrock-runtime.{self.aws_region}.amazonaws.com/"
                    f"inference-profiles/{identifier}/model/invoke"
                )
            elif self.profile_id:
                identifier = quote(self.profile_id, safe="")
                url = (
                    f"https://bedrock-runtime.{self.aws_region}.amazonaws.com/"
                    f"inference-profiles/{identifier}/model/invoke"
                )
            elif self.model_id:
                identifier = quote(self.model_id, safe="")
                url = (
                    f"https://bedrock-runtime.{self.aws_region}.amazonaws.com/"
                    f"model/{identifier}/invoke"
                )
            else:
                raise RuntimeError(
                    "Bedrock not configured: set BEDROCK_TEXT_INFERENCE_PROFILE_ID (or ARN) or BEDROCK_TEXT_MODEL_ID",
                )

            if self.target_model_region:
                url += f"?targetModelRegion={quote(self.target_model_region, safe='')}"

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-api-key": self.api_key,
            }
            resp = requests.post(url, headers=headers, data=body_json)
            resp.raise_for_status()
            return resp.json()

        kwargs = self._build_invoke_kwargs(body_json)
        call_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in self._params_supported or k in {"contentType", "accept", "body"}
        }
        resp = self.bedrock.invoke_model(**call_kwargs)
        raw = resp["body"].read()
        return json.loads(raw)


def format_user_message(text: str) -> Dict[str, Any]:
    return {"role": "user", "content": text}


def extract_text(response: Dict[str, Any]) -> str:
    """
    Extract assistant text from Anthropic Bedrock response:
    response["content"] is a list; first item often has {"type":"text","text": "..."}
    """
    try:
        parts = response.get("content", [])
        if parts and isinstance(parts, list):
            first = parts[0]
            if isinstance(first, dict) and "text" in first:
                return first["text"]
        # Fallback: some SDKs wrap differently
        return response.get("output", {}).get("text", "")
    except Exception:
        return ""
