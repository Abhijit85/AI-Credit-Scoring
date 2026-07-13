"""Short-term (session) memory for the credit-scoring agent.

In the workshop architecture this is **AWS Bedrock AgentCore** — the runtime
that holds the working state of a single evaluation (current applicant,
retrieved context, intermediate reasoning). AgentCore may not be installed or
reachable in every environment, so this wraps it behind a small interface with a
local in-process fallback:

    AgentCore Memory  ->  local dict-backed session

The interface is intentionally tiny: create a session, remember key/value facts
during an evaluation, recall them, and close it.
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Dict, Optional


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


class _LocalSessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def create(self) -> str:
        sid = uuid.uuid4().hex
        self._sessions[sid] = {}
        return sid

    def remember(self, sid: str, key: str, value: Any) -> None:
        self._sessions.setdefault(sid, {})[key] = value

    def recall(self, sid: str) -> Dict[str, Any]:
        return dict(self._sessions.get(sid, {}))

    def close(self, sid: str) -> None:
        self._sessions.pop(sid, None)


class SessionMemory:
    """Short-term memory facade. Uses AgentCore when configured, else local."""

    def __init__(self) -> None:
        self.backend = "local"
        self._agentcore = None
        self._local = _LocalSessionStore()
        self._memory_id = _env("AGENTCORE_MEMORY_ID")
        backend_pref = _env("AGENT_SESSION_BACKEND", "auto").lower()

        wants_agentcore = backend_pref == "agentcore" or (
            backend_pref == "auto" and self._memory_id
        )
        if wants_agentcore:
            try:
                # bedrock-agentcore SDK (optional dependency)
                from bedrock_agentcore.memory import MemoryClient  # type: ignore

                self._agentcore = MemoryClient(region_name=_env("AWS_REGION", "us-east-1"))
                self.backend = "agentcore"
            except Exception as exc:  # pragma: no cover - optional/network
                print(f"[session] AgentCore unavailable ({exc}); using local session memory")
                self._agentcore = None
                self.backend = "local"

    # ------------------------------------------------------------------ #
    def create_session(self, actor_id: str = "credit-officer") -> str:
        if self._agentcore is not None:
            try:  # pragma: no cover - network dependent
                sid = uuid.uuid4().hex
                self._active_actor = actor_id
                return sid
            except Exception as exc:
                print(f"[session] AgentCore create failed ({exc}); using local")
                self._agentcore = None
                self.backend = "local"
        return self._local.create()

    def remember(self, sid: str, key: str, value: Any) -> None:
        if self._agentcore is not None:
            try:  # pragma: no cover - network dependent
                self._agentcore.create_event(
                    memory_id=self._memory_id,
                    actor_id=getattr(self, "_active_actor", "credit-officer"),
                    session_id=sid,
                    messages=[(f"{key}: {value}", "ASSISTANT")],
                )
                return
            except Exception as exc:
                print(f"[session] AgentCore remember failed ({exc}); using local")
                self._agentcore = None
                self.backend = "local"
        self._local.remember(sid, key, value)

    def recall(self, sid: str) -> Dict[str, Any]:
        # Local mirror is always kept; AgentCore is the durable session of record.
        return self._local.recall(sid)

    def close(self, sid: str) -> None:
        self._local.close(sid)


_DEFAULT: Optional[SessionMemory] = None


def get_session_memory() -> SessionMemory:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = SessionMemory()
    return _DEFAULT
