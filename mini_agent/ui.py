"""Rich terminal user interface for displaying agent thoughts, actions, and results."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme

custom_theme = Theme(
    {
        "info": "dim cyan",
        "warning": "yellow",
        "error": "bold red",
        "thought": "italic magenta",
        "tool": "bold cyan",
        "success": "bold green",
    }
)

console = Console(theme=custom_theme)


def print_banner(model: str, provider: str, tools: list[str]) -> None:
    """Print welcoming banner displaying model and tools."""
    banner_text = Text()
    banner_text.append("🤖 MiniAgent ", style="bold bright_cyan")
    banner_text.append("• Lightweight ReAct Autonomous Assistant\n", style="dim white")
    banner_text.append(f"Model: {model}  |  Provider: {provider}  |  Tools: {len(tools)}\n", style="cyan")
    banner_text.append("Loaded: ", style="dim")
    banner_text.append(", ".join(tools), style="yellow")

    console.print(
        Panel(
            banner_text,
            border_style="bright_blue",
            padding=(0, 2),
        )
    )


def print_step(step: int, max_steps: int) -> None:
    """Print separator for current step."""
    console.print(f"\n[dim white]─── Step {step}/{max_steps} ──────────────────────────────────────[/dim white]")


def print_thought(thought: str) -> None:
    """Print reasoning thought."""
    if not thought.strip():
        return
    console.print(
        Panel(
            Text(thought.strip(), style="thought"),
            title="🤔 Thought",
            title_align="left",
            border_style="magenta",
            padding=(0, 1),
        )
    )


def print_tool_call(name: str, arguments: dict[str, Any]) -> None:
    """Print tool call invocation."""
    args_json = json.dumps(arguments, indent=2)
    syntax = Syntax(args_json, "json", theme="monokai", word_wrap=True)

    console.print(
        Panel(
            syntax,
            title=f"⚡ Tool Call: [bold yellow]{name}[/bold yellow]",
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
        )
    )


def print_tool_result(name: str, result: str, max_display_lines: int = 20) -> None:
    """Print observation result returned by tool."""
    lines = result.splitlines()
    truncated = False
    if len(lines) > max_display_lines:
        display_text = "\n".join(lines[:max_display_lines]) + f"\n... [{len(lines) - max_display_lines} lines hidden]"
        truncated = True
    else:
        display_text = result

    # Check if result looks like code or JSON
    if result.strip().startswith("{") and result.strip().endswith("}"):
        content = Syntax(display_text, "json", theme="monokai", word_wrap=True)
    else:
        content = Text(display_text, style="white")

    console.print(
        Panel(
            content,
            title=f"👁️ Observation: [dim]{name}[/dim]" + (" (truncated)" if truncated else ""),
            title_align="left",
            border_style="green",
            padding=(0, 1),
        )
    )


def print_final_answer(answer: str) -> None:
    """Print the final answer rendered as Markdown."""
    console.print(
        Panel(
            Markdown(answer),
            title="✨ Final Answer",
            title_align="left",
            border_style="bright_green",
            padding=(1, 2),
        )
    )


def print_error(message: str) -> None:
    """Print error message."""
    console.print(f"[bold red]❌ Error:[/bold red] {message}")
