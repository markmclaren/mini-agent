"""Core autonomous agent ReAct execution loop."""

from __future__ import annotations

import json
from typing import Any

from mini_agent.llm import BaseLLMClient, ToolCall
from mini_agent.tools import ToolRegistry, default_registry
from mini_agent import ui

DEFAULT_SYSTEM_PROMPT = """You are MiniAgent, an autonomous AI research and task assistant.
You have access to a suite of real-world tools:
- `search_web`: Search the web for up-to-date facts, current events, and references.
- `fetch_page`: Retrieve and extract clean text from any URL.
- `execute_python`: Execute Python code in a REPL environment (useful for math, data parsing, logic, string operations).
- `read_file`: Read contents of a local file.
- `write_file`: Write text to a file in the workspace.
- `list_files`: List files in a local directory.
- `system_info`: Inspect current date, time, OS, and platform.

Guidelines:
1. Always think step by step before acting.
2. If a query requires fresh information, ALWAYS search the web rather than hallucinating.
3. If a problem involves calculations, dates, or complex processing, ALWAYS execute Python code to verify the answer.
4. Chain multiple tools if needed to solve complex multi-part problems.
5. Once you have all necessary information, provide a thorough, structured, and helpful final answer without calling any more tools.
"""


class Agent:
    """Autonomous ReAct agent that reasons, calls tools, and solves tasks."""

    def __init__(
        self,
        llm: BaseLLMClient,
        registry: ToolRegistry | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 10,
        verbose: bool = True,
    ) -> None:
        self.llm = llm
        self.registry = registry or default_registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.verbose = verbose
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def reset(self) -> None:
        """Reset conversation memory."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def run(self, user_prompt: str) -> str:
        """Execute a user query through the ReAct loop until completion."""
        self.messages.append({"role": "user", "content": user_prompt})
        tool_schemas = self.registry.list_schemas()

        step = 0
        while step < self.max_steps:
            step += 1
            if self.verbose:
                ui.print_step(step, self.max_steps)

            response = self.llm.complete(self.messages, tools=tool_schemas)

            # Display thoughts / reasoning if provided
            if response.content and self.verbose:
                ui.print_thought(response.content)

            # If no tool calls were requested, this is the final answer!
            if not response.tool_calls:
                final_answer = response.content or "(No content returned)"
                if self.verbose:
                    ui.print_final_answer(final_answer)
                self.messages.append({"role": "assistant", "content": final_answer})
                return final_answer

            # Record assistant message with tool calls in history
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ],
            }
            self.messages.append(assistant_msg)

            # Execute each requested tool call
            for tc in response.tool_calls:
                if self.verbose:
                    ui.print_tool_call(tc.name, tc.arguments)

                # Dispatch tool call
                observation = self.registry.execute(tc.name, tc.arguments)

                if self.verbose:
                    ui.print_tool_result(tc.name, observation)

                # Append tool observation to messages
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": observation,
                    }
                )

        # Max steps reached guard
        if self.verbose:
            ui.print_error(f"Reached maximum step limit ({self.max_steps}). Summarizing findings...")

        self.messages.append(
            {
                "role": "user",
                "content": "You have reached the maximum step limit. Based on all observations above, provide your best final response now.",
            }
        )
        final_summary = self.llm.complete(self.messages, tools=None)
        answer = final_summary.content or "Completed without final summary."
        if self.verbose:
            ui.print_final_answer(answer)
        self.messages.append({"role": "assistant", "content": answer})
        return answer
