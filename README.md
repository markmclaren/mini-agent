# 🤖 MiniAgent

> A small, transparent, and genuinely useful autonomous AI agent in Python. Built from first principles with zero framework bloat.

MiniAgent implements the **ReAct (Reasoning + Acting)** pattern with real-world tools, multi-step planning, self-correction, and a beautiful terminal UI powered by [Rich](https://github.com/Textualize/rich).

---

## ✨ Features

- **Autonomous ReAct Loop**: Seamlessly alternates between reasoning (thought), calling tools (actions), inspecting outputs (observations), and synthesizing final answers.
- **7 Built-in Practical Tools**:
  - `search_web`: Live search with automatic Wikipedia fallback for fresh facts and reference information.
  - `fetch_page`: Scrapes and strips web pages to extract clean, readable text.
  - `execute_python`: Sandboxed REPL capturing stdout/stderr and evaluating expressions (useful for math, datetime arithmetic, data parsing).
  - `read_file` / `write_file` / `list_files`: Workspace file reader and writer.
  - `system_info`: Inspects local time, operating system, architecture, and workspace directory.
- **Trivial Tool Registration**: Register any Python function with `@tool`—type hints and docstrings are automatically converted into OpenAI-compatible JSON function schemas.
- **Universal LLM Backend**:
  - Automatically detects `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, or local endpoints (`OPENAI_BASE_URL` for Ollama / LM Studio).
  - Works with any model (`nvidia/nemotron-3.5-lightning:free`, `gpt-4o-mini`, `llama3.2`, etc.).
  - Includes `--mock` simulation mode for offline testing and demos without an API key.
- **Interactive REPL & CLI**: Single-prompt task runner and interactive chat mode with memory, `/reset`, and `/tools` commands.
- **Fully Tested**: Tested with `pytest` covering schema generation, tool execution, and agent loops.

---

## 🚀 Quickstart

### 1. Requirements

- Python 3.10+ (or [uv](https://github.com/astral-sh/uv))

### 2. Installation

Clone or enter the directory and install dependencies:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 3. Running Tasks

#### Live Model Task Execution:
```bash
# Run a multi-step research and calculation task
uv run python main.py "What is today's date and calculate the square root of 987654321?"

# Research and save results to a local file
uv run python main.py "Search for Python 3.14 features and save a summary to python_314.txt"
```

#### Interactive Multi-Turn Mode:
```bash
uv run python main.py -i
```

Special interactive commands:
- `/tools` — Inspect all loaded tools and their descriptions.
- `/reset` — Clear conversation history and reset memory.
- `/quit` — Exit the session.

#### Offline / Mock Mode (Zero Setup):
```bash
uv run python main.py --mock "Calculate 42 * 1337"
```

---

## 🔧 Using with Local LLMs (Ollama, LM Studio)

MiniAgent supports any OpenAI-compatible server:

```bash
# Example with local Ollama
uv run python main.py --base-url "http://localhost:11434/v1" --model "llama3.2" "List the files in this directory"
```

---

## 🧩 Adding Custom Tools in 4 Lines

Adding new capabilities to MiniAgent requires no manual JSON schemas:

```python
from mini_agent import Agent, OpenAILLMClient, tool

# Define and register your custom function
@tool
def get_stock_price(ticker: str) -> str:
    """Fetch current stock price for a given ticker symbol."""
    # Your logic here
    return f"{ticker.upper()} is currently trading at $150.25"

# Create agent and run
agent = Agent(llm=OpenAILLMClient())
agent.run("What is the stock price of AAPL?")
```

---

## 📁 Project Structure

```text
agent-test/
├── main.py              # CLI entrypoint (args, interactive mode, runner)
├── pyproject.toml       # Dependencies and package configuration
├── README.md            # Documentation and guide
├── mini_agent/
│   ├── __init__.py      # Public exports
│   ├── agent.py         # Core ReAct loop, step guard, conversation memory
│   ├── llm.py           # Universal OpenAI-compatible client + MockLLM
│   ├── tools.py         # Tool registry, @tool decorator, built-in tools
│   └── ui.py            # Rich terminal styling (thought panels, tool cards)
└── tests/
    └── test_agent.py    # Unit & integration tests
```

---

## 🧪 Running Tests

```bash
uv run pytest -v
```
