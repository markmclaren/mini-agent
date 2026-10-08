"""Unit and integration tests for MiniAgent components."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mini_agent.agent import Agent
from mini_agent.llm import MockLLMClient, ToolCall
from mini_agent.tools import (
    ToolRegistry,
    execute_python,
    fetch_page,
    list_files,
    read_file,
    system_info,
    write_file,
)


class TestTools:
    """Test suite for built-in tools."""

    def test_execute_python_basic(self) -> None:
        result = execute_python("print(2 + 2)")
        assert "4" in result

    def test_execute_python_expression_return(self) -> None:
        result = execute_python("math.sqrt(144)")
        assert "12.0" in result

    def test_execute_python_error_handling(self) -> None:
        result = execute_python("1 / 0")
        assert "ZeroDivisionError" in result

    def test_file_operations(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.txt"
        write_res = write_file(str(test_file), "Hello Agent World!")
        assert "Successfully wrote" in write_res
        assert test_file.exists()

        read_res = read_file(str(test_file))
        assert read_res == "Hello Agent World!"

    def test_system_info(self) -> None:
        info_str = system_info()
        data = json.loads(info_str)
        assert "local_time" in data
        assert "system" in data
        assert "python_version" in data


class TestToolRegistry:
    """Test suite for ToolRegistry schema generation and execution."""

    def test_custom_tool_registration(self) -> None:
        registry = ToolRegistry()

        @registry.register
        def greet(name: str, shout: bool = False) -> str:
            """Say hello to a user."""
            msg = f"Hello, {name}!"
            return msg.upper() if shout else msg

        schemas = registry.list_schemas()
        assert len(schemas) == 1
        fn_schema = schemas[0]["function"]
        assert fn_schema["name"] == "greet"
        assert "name" in fn_schema["parameters"]["properties"]
        assert "shout" in fn_schema["parameters"]["properties"]
        assert fn_schema["parameters"]["required"] == ["name"]

        # Test execution
        out = registry.execute("greet", {"name": "Alice"})
        assert out == "Hello, Alice!"

        out_shout = registry.execute("greet", {"name": "Bob", "shout": True})
        assert out_shout == "HELLO, BOB!"

    def test_unknown_tool_execution(self) -> None:
        registry = ToolRegistry()
        out = registry.execute("non_existent", {})
        assert "Error: Tool 'non_existent' not found" in out


class TestAgentLoop:
    """Test suite for the Agent ReAct loop."""

    def test_agent_run_with_mock_llm(self) -> None:
        mock_llm = MockLLMClient()
        agent = Agent(llm=mock_llm, max_steps=5, verbose=False)

        # Math query triggers execute_python tool call in mock LLM
        answer = agent.run("Calculate 42 * 1337 using python")
        assert "56154" in answer
        assert len(agent.messages) > 2

    def test_agent_reset(self) -> None:
        mock_llm = MockLLMClient()
        agent = Agent(llm=mock_llm, verbose=False)
        agent.run("Hello")
        assert len(agent.messages) > 1
        agent.reset()
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"
