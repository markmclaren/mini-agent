"""LLM Client interface supporting OpenAI, OpenRouter, LocalAI/Ollama, and Mock mode."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


@dataclass
class ToolCall:
    """Represents a single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Standardized response from the LLM."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    raw_message: Any = None


class BaseLLMClient:
    """Abstract base class for LLM clients."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        raise NotImplementedError


class OpenAILLMClient(BaseLLMClient):
    """Universal client using OpenAI-compatible API (supports OpenAI, OpenRouter, Ollama, etc.)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.provider = "custom"
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        custom_base = base_url or os.environ.get("OPENAI_BASE_URL")

        if api_key:
            self.api_key = api_key
            self.base_url = custom_base
            self.model = model or "gpt-4o-mini"
        elif openrouter_key:
            self.provider = "openrouter"
            self.api_key = openrouter_key
            self.base_url = custom_base or "https://openrouter.ai/api/v1"
            # Default to a reliable free model that supports tool calling
            self.model = model or os.environ.get("AGENT_MODEL") or "nvidia/nemotron-3.5-lightning:free"
        elif openai_key:
            self.provider = "openai"
            self.api_key = openai_key
            self.base_url = custom_base
            self.model = model or os.environ.get("AGENT_MODEL") or "gpt-4o-mini"
        elif custom_base:
            self.provider = "custom_base"
            self.api_key = "dummy-key"
            self.base_url = custom_base
            self.model = model or os.environ.get("AGENT_MODEL") or "llama3.2"
        else:
            raise ValueError(
                "No API key provided. Set OPENROUTER_API_KEY, OPENAI_API_KEY, or run with --mock."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []

        # 1. Standard OpenAI tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                args: dict[str, Any] = {}
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    # In case of partially malformed JSON string, try loose regex
                    args = {"raw_input": tc.function.arguments}

                tool_calls.append(
                    ToolCall(
                        id=tc.id or f"call_{len(tool_calls)}",
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        # 2. Fallback text parser in case model output tool call in message content
        if not tool_calls and msg.content:
            fallback_calls = self._parse_fallback_tool_calls(msg.content)
            if fallback_calls:
                tool_calls.extend(fallback_calls)

        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            raw_message=msg,
        )

    def _parse_fallback_tool_calls(self, text: str) -> list[ToolCall]:
        """Backup parser for models returning tool invocations in plain text format."""
        calls: list[ToolCall] = []
        # Pattern for ```json { "action": "tool_name", "action_input": {...} } ```
        pattern = r"```(?:json)?\s*\{\s*\"action\"\s*:\s*\"([^\"]+)\"\s*,\s*\"action_input\"\s*:\s*(\{.*?\})\s*\}\s*```"
        matches = re.finditer(pattern, text, re.DOTALL)
        for i, match in enumerate(matches):
            tool_name = match.group(1)
            raw_args = match.group(2)
            try:
                args = json.loads(raw_args)
            except Exception:
                args = {}
            calls.append(ToolCall(id=f"fallback_call_{i}", name=tool_name, arguments=args))
        return calls


class MockLLMClient(BaseLLMClient):
    """Deterministic simulation client for testing and offline demonstrations."""

    def __init__(self, model: str = "mock-agent-v1") -> None:
        self.model = model
        self.step_count = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        self.step_count += 1
        last_msg = messages[-1]

        # If previous message is a tool result, formulate final answer
        if last_msg.get("role") == "tool":
            tool_content = last_msg.get("content", "")
            return LLMResponse(
                content=(
                    f"Based on the tool execution results:\n\n"
                    f"```\n{tool_content}\n```\n\n"
                    f"The task was successfully processed and analyzed."
                ),
                tool_calls=[],
                finish_reason="stop",
            )

        # Find user prompt
        user_query = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_query = m.get("content", "")
                break

        query_lower = user_query.lower()

        # Route to appropriate tool based on query content
        if "calc" in query_lower or "math" in query_lower or any(c in query_lower for c in "+*/^"):
            return LLMResponse(
                content="I need to calculate this using Python.",
                tool_calls=[
                    ToolCall(
                        id="mock_call_1",
                        name="execute_python",
                        arguments={"code": "import math\nprint(42 * 1337)"},
                    )
                ],
                finish_reason="tool_calls",
            )
        elif "time" in query_lower or "system" in query_lower or "date" in query_lower:
            return LLMResponse(
                content="Checking current system information and time.",
                tool_calls=[
                    ToolCall(
                        id="mock_call_2",
                        name="system_info",
                        arguments={},
                    )
                ],
                finish_reason="tool_calls",
            )
        elif "search" in query_lower or "who" in query_lower or "what" in query_lower:
            return LLMResponse(
                content=f"Searching the web for information regarding: '{user_query}'.",
                tool_calls=[
                    ToolCall(
                        id="mock_call_3",
                        name="search_web",
                        arguments={"query": user_query, "max_results": 3},
                    )
                ],
                finish_reason="tool_calls",
            )
        else:
            return LLMResponse(
                content=(
                    f"[Mock Mode] Hello! I received your prompt: '{user_query}'.\n"
                    f"To connect to live LLMs, set OPENROUTER_API_KEY or OPENAI_API_KEY."
                ),
                tool_calls=[],
                finish_reason="stop",
            )
