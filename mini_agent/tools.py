"""Tool registry and built-in tools for MiniAgent.

Provides a decorator @tool for registering Python functions as agent tools,
auto-generating OpenAI-compatible JSON schemas from type hints and docstrings.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import io
import json
import os
import platform
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, get_type_hints

import httpx
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except ImportError:
        DDGS = None  # type: ignore


class ToolRegistry:
    """Manages available tools and provides schema generation and dispatch."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}

    def register(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Any:
        """Decorator to register a function as a tool."""

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or f.__name__
            tool_desc = description or (inspect.getdoc(f) or "No description provided.").strip()
            schema = self._build_schema(f, tool_name, tool_desc)

            self._tools[tool_name] = f
            self._schemas[tool_name] = schema
            return f

        if func is None:
            return decorator
        return decorator(func)

    def get_tool(self, name: str) -> Callable[..., Any] | None:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def list_schemas(self) -> list[dict[str, Any]]:
        """Return all registered tools as OpenAI-compatible function schemas."""
        return list(self._schemas.values())

    def execute(self, name: str, arguments: dict[str, Any] | str) -> str:
        """Execute a tool with provided arguments and return string result."""
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found. Available tools: {list(self._tools.keys())}"

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return f"Error: Could not parse arguments as JSON: {exc}"

        try:
            result = tool(**arguments)
            if result is None:
                return "Success: (no return value)"
            if isinstance(result, (dict, list)):
                return json.dumps(result, indent=2, default=str)
            return str(result)
        except Exception as exc:
            return f"Error executing tool '{name}': {type(exc).__name__}: {exc}"

    def _build_schema(
        self, func: Callable[..., Any], name: str, description: str
    ) -> dict[str, Any]:
        """Generate OpenAI function calling schema from function signature and type hints."""
        sig = inspect.signature(func)
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        properties: dict[str, Any] = {}
        required: list[str] = []

        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = hints.get(param_name, str)
            json_type = type_mapping.get(param_type, "string")

            prop: dict[str, Any] = {
                "type": json_type,
                "description": f"Parameter '{param_name}'",
            }

            if param.default is inspect.Parameter.empty:
                required.append(param_name)
            else:
                prop["default"] = param.default

            properties[param_name] = prop

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# Global default registry
default_registry = ToolRegistry()
tool = default_registry.register


# =====================================================================
# Built-in Tools
# =====================================================================


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for up-to-date facts, documentation, news, or articles.
    Uses live search with automatic Wikipedia fallback.
    """
    query = query.strip()
    if not query:
        return "Error: Empty search query provided."

    results: list[dict[str, str]] = []

    # 1. Try DuckDuckGo if available
    if DDGS is not None:
        try:
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(query, max_results=max_results))
                for item in ddg_results:
                    results.append({
                        "title": item.get("title", "Untitled"),
                        "url": item.get("href", ""),
                        "snippet": item.get("body", ""),
                    })
        except Exception:
            # Fall back to Wikipedia below
            pass

    # 2. Fallback to Wikipedia search if no results or error
    if not results:
        try:
            wiki_headers = {
                "User-Agent": "MiniAgentBot/1.0 (https://github.com/agent-app; bot@example.com)"
            }
            with httpx.Client(timeout=8.0) as client:
                resp = client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": query,
                        "format": "json",
                        "srlimit": max_results,
                    },
                    headers=wiki_headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    search_items = data.get("query", {}).get("search", [])
                    for item in search_items:
                        title = item.get("title", "")
                        raw_snippet = item.get("snippet", "")
                        clean_snippet = re.sub(r"<[^>]+>", "", raw_snippet)
                        results.append({
                            "title": f"Wikipedia: {title}",
                            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                            "snippet": clean_snippet,
                        })
        except Exception as exc:
            if not results:
                return f"Search error across providers: {exc}"

    if not results:
        return f"No search results found for query: '{query}'"

    formatted: list[str] = [f"Found {len(results)} results for '{query}':\n"]
    for i, r in enumerate(results, 1):
        formatted.append(f"{i}. **{r['title']}**")
        if r['url']:
            formatted.append(f"   URL: {r['url']}")
        if r['snippet']:
            formatted.append(f"   Snippet: {r['snippet']}")
        formatted.append("")

    return "\n".join(formatted)


@tool
def fetch_page(url: str, max_chars: int = 3500) -> str:
    """Fetch and extract clean readable text content from any web page URL."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            html = resp.text

        soup = BeautifulSoup(html, "html.parser")

        # Strip scripts, styles, navigations, footers
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned = "\n".join(lines)

        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars] + f"\n\n... [Content truncated at {max_chars} characters]"

        return cleaned if cleaned else "Fetched page but found no readable text."

    except Exception as exc:
        return f"Error fetching {url}: {type(exc).__name__}: {exc}"


@tool
def execute_python(code: str) -> str:
    """Execute Python code in an interactive REPL environment.
    Captures stdout, stderr, and automatically prints the value of the last expression.
    Useful for math, string manipulation, data processing, algorithms, and tests.
    """
    code = code.strip()
    if not code:
        return "Error: Empty Python code string."

    buf_out = io.StringIO()
    buf_err = io.StringIO()
    namespace: dict[str, Any] = {
        "math": __import__("math"),
        "datetime": __import__("datetime"),
        "json": __import__("json"),
        "re": __import__("re"),
        "collections": __import__("collections"),
        "statistics": __import__("statistics"),
        "random": __import__("random"),
    }

    try:
        tree = ast.parse(code)
        if not tree.body:
            return "No executable code found."

        last_expr = None
        if isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body.pop()

        exec_bytecode = compile(tree, filename="<agent-code>", mode="exec")

        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            exec(exec_bytecode, namespace)

            if last_expr is not None:
                eval_bytecode = compile(
                    ast.Expression(last_expr.value), filename="<agent-code>", mode="eval"
                )
                val = eval(eval_bytecode, namespace)
                # If stdout is empty and expression returned a value, display it
                if val is not None and not buf_out.getvalue().strip():
                    print(repr(val))

        stdout = buf_out.getvalue()
        stderr = buf_err.getvalue()

        output = stdout
        if stderr:
            output += f"\n[STDERR]:\n{stderr}"

        return output if output.strip() else "Executed successfully with no output."

    except Exception:
        return f"Python Execution Error:\n{traceback.format_exc()}"


@tool
def read_file(filepath: str, max_chars: int = 5000) -> str:
    """Read contents of a text file from the local workspace."""
    path = Path(filepath).expanduser().resolve()
    if not path.exists():
        return f"Error: File '{filepath}' does not exist."
    if not path.is_file():
        return f"Error: '{filepath}' is not a file."

    try:
        content = path.read_text(encoding="utf-8")
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n... [Truncated at {max_chars} chars]"
        return content
    except Exception as exc:
        return f"Error reading file '{filepath}': {exc}"


@tool
def write_file(filepath: str, content: str) -> str:
    """Write or overwrite text content to a file in the workspace."""
    try:
        path = Path(filepath).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to '{filepath}'."
    except Exception as exc:
        return f"Error writing file '{filepath}': {exc}"


@tool
def list_files(directory: str = ".") -> str:
    """List files and subdirectories in the specified directory."""
    path = Path(directory).expanduser().resolve()
    if not path.exists():
        return f"Error: Directory '{directory}' does not exist."
    if not path.is_dir():
        return f"Error: '{directory}' is not a directory."

    items = []
    try:
        for entry in sorted(path.iterdir()):
            kind = "DIR " if entry.is_dir() else "FILE"
            size = entry.stat().st_size if entry.is_file() else 0
            items.append(f"[{kind}] {entry.name:<30} ({size} bytes)")
        return "\n".join(items) if items else f"Directory '{directory}' is empty."
    except Exception as exc:
        return f"Error listing directory '{directory}': {exc}"


@tool
def system_info() -> str:
    """Get system information including current local time, OS, and platform."""
    now = datetime.now()
    return json.dumps(
        {
            "local_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "iso_time": now.isoformat(),
            "weekday": now.strftime("%A"),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": sys.version.split()[0],
            "working_directory": os.getcwd(),
        },
        indent=2,
    )
