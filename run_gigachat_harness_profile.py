"""Run deepagents + GigaChat with the custom HarnessProfile enabled."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_gigachat import GigaChat

from gigachat_harness_profile import register_gigachat_harness_profile
from dotenv import load_dotenv

load_dotenv()


def _extract_text(result: Any) -> str:
    if isinstance(result, dict):
        messages = result.get("messages") or []
        if messages:
            last = messages[-1]
            content = getattr(last, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(last, dict) and isinstance(last.get("content"), str):
                return last["content"]
    return str(result)


def main() -> None:
    register_gigachat_harness_profile()

    model = GigaChat()
    backend = FilesystemBackend(root_dir=Path.cwd(), virtual_mode=True)

    agent = create_deep_agent(
        model=model,
        tools=[],
        backend=backend,
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Move all configuration (env variables, arguments, defaults) into a single module, add validation for required parameters with human-readable errors, and remove duplicated env-reading logic from other files.",
                }
            ]
        }
    )

    print("=== Agent response ===")
    print(_extract_text(result))
    print("======================")


if __name__ == "__main__":
    main()
