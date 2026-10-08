"""Demonstration of how to import and use MiniAgent in custom Python applications."""

from mini_agent import Agent, OpenAILLMClient, MockLLMClient, tool, default_registry
import os


# 1. Register a custom domain-specific tool
@tool
def get_user_status(username: str) -> str:
    """Check membership status and tier for a given username."""
    database = {
        "alice": {"tier": "Pro", "active": True, "projects": 12},
        "bob": {"tier": "Free", "active": True, "projects": 2},
    }
    user = database.get(username.lower())
    if not user:
        return f"User '{username}' was not found in membership registry."
    return f"User {username} is on the {user['tier']} tier with {user['projects']} active projects."


def main():
    # Use real LLM if API key is present, otherwise mock simulation
    has_key = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    llm = OpenAILLMClient() if has_key else MockLLMClient()

    agent = Agent(llm=llm, max_steps=5)

    prompt = "Check the user status of Alice and calculate if she qualifies for enterprise (requires > 10 projects)."
    print(f"\nRunning query: {prompt}\n")
    response = agent.run(prompt)


if __name__ == "__main__":
    main()
