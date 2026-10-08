"""Main CLI entrypoint for MiniAgent."""

from __future__ import annotations

import argparse
import os
import sys

from mini_agent import Agent, MockLLMClient, OpenAILLMClient, default_registry, ui


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MiniAgent - A lightweight, autonomous ReAct AI agent in Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single task
  python main.py "Search for the latest Python version and compute days since release"

  # Run in interactive REPL mode
  python main.py -i

  # Run with custom model
  python main.py --model "openai/gpt-4o-mini" "Summarize the files in this directory"

  # Run offline / mock simulation mode
  python main.py --mock "What is 42 * 1337?"
        """,
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Prompt or task for the agent to execute. If omitted, starts interactive mode.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive multi-turn conversational mode.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Model name to use (default: nvidia/nemotron-3.5-lightning:free for OpenRouter, gpt-4o-mini for OpenAI)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for OpenAI-compatible endpoint (e.g., http://localhost:11434/v1 for Ollama)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key override (or set OPENROUTER_API_KEY / OPENAI_API_KEY in environment)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with Mock LLM simulation mode (no API key needed, executes real tools)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum ReAct reasoning steps per turn (default: 10)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Quiet mode: suppress intermediate thoughts and tool observations",
    )

    return parser.parse_args()


def init_llm(args: argparse.Namespace) -> tuple[Any, str, str]:
    """Initialize the LLM client based on arguments and environment."""
    if args.mock:
        client = MockLLMClient(model=args.model or "mock-agent-v1")
        return client, client.model, "mock-simulation"

    # Check for keys or base URL
    has_key = bool(
        args.api_key
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or args.base_url
        or os.environ.get("OPENAI_BASE_URL")
    )

    if not has_key:
        ui.console.print(
            "[yellow]⚠️ No API key found in environment (OPENROUTER_API_KEY or OPENAI_API_KEY).[/yellow]\n"
            "[cyan]Falling back to offline [bold]--mock[/bold] simulation mode so you can see the agent in action![/cyan]\n"
            "[dim]Tip: Set OPENROUTER_API_KEY or OPENAI_API_KEY to connect to live models.[/dim]\n"
        )
        client = MockLLMClient(model=args.model or "mock-agent-v1")
        return client, client.model, "mock-simulation"

    try:
        client = OpenAILLMClient(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
        )
        return client, client.model, client.provider
    except Exception as exc:
        ui.console.print(f"[red]Error initializing LLM client: {exc}[/red]")
        ui.console.print("[dim]Falling back to mock simulation mode.[/dim]\n")
        mock = MockLLMClient(model="mock-fallback")
        return mock, mock.model, "mock-simulation"


def main() -> None:
    args = parse_args()

    llm, model_name, provider = init_llm(args)
    tool_names = list(default_registry._tools.keys())

    agent = Agent(
        llm=llm,
        registry=default_registry,
        max_steps=args.max_steps,
        verbose=not args.quiet,
    )

    if not args.quiet:
        ui.print_banner(model_name, provider, tool_names)

    # Interactive mode if requested or if no single prompt was passed
    if args.interactive or args.prompt is None:
        ui.console.print(
            "[dim]Entering interactive mode. Special commands: [bold]/tools[/bold], [bold]/reset[/bold], [bold]/quit[/bold][/dim]\n"
        )
        while True:
            try:
                user_input = ui.console.input("\n[bold green]User ❯ [/bold green]").strip()
            except (KeyboardInterrupt, EOFError):
                ui.console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue

            if user_input in ("/quit", "/exit", ":q"):
                ui.console.print("[dim]Goodbye![/dim]")
                break
            elif user_input == "/reset":
                agent.reset()
                ui.console.print("[yellow]🔄 Agent conversation memory reset.[/yellow]")
                continue
            elif user_input == "/tools":
                ui.console.print("\n[bold cyan]Available Tools:[/bold cyan]")
                for name, fn in default_registry._tools.items():
                    doc = (fn.__doc__ or "").strip().split("\n")[0]
                    ui.console.print(f"  • [bold yellow]{name:<16}[/bold yellow] {doc}")
                continue

            agent.run(user_input)

    else:
        # Single-prompt execution
        agent.run(args.prompt)


if __name__ == "__main__":
    main()
