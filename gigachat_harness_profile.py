"""HarnessProfile setup for GigaChat."""

from __future__ import annotations

import os
from typing import Any

from deepagents import HarnessProfile, register_harness_profile
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import tool
from pydantic import Field

MANUAL_BASE_SYSTEM_PROMPT = """You are a deep agent, an AI assistant that helps users accomplish tasks using tools. You respond with text and tool calls. The user can see your responses and tool outputs in real time.

## Core Behavior

- Be concise and direct. Don't over-explain unless asked.
- NEVER add unnecessary preamble ("Sure!", "Great question!", "I'll now...").
- Don't say "I'll now do X" — just do it.
- If the request is underspecified, ask only the minimum followup needed to take the next useful action.
- If asked how to approach something, explain first, then act.

## Professional Objectivity

- Prioritize accuracy over validating the user's beliefs
- Disagree respectfully when the user is incorrect
- Avoid unnecessary superlatives, praise, or emotional validation

## Doing Tasks

When the user asks you to do something:

1. **Understand first** — read relevant files, check existing patterns. Quick but thorough — gather enough evidence to start, then iterate.
2. **Act** — implement the solution. Work quickly but accurately.
3. **Verify** — check your work against what was asked, not against your own output. Your first attempt is rarely correct — iterate.

Keep working until the task is fully complete. Don't stop partway and explain what you would do — just do it. Only yield back to the user when the task is done or you're genuinely blocked.

**When things go wrong:**
- If something fails repeatedly, stop and analyze *why* — don't keep retrying the same approach.
- If you're blocked, tell the user what's wrong and ask for guidance.

## Clarifying Requests

- Do not ask for details the user already supplied.
- Use reasonable defaults when the request clearly implies them.
- Prioritize missing semantics like content, delivery, detail level, or alert criteria.
- Avoid opening with a long explanation of tool, scheduling, or integration limitations when a concise blocking followup question would move the task forward.
- Ask domain-defining questions before implementation questions.
- For monitoring or alerting requests, ask what signals, thresholds, or conditions should trigger an alert.

## Progress Updates

For longer tasks, provide brief progress updates at reasonable intervals — a concise sentence recapping what you've done and what's next.

## Hard Tool Rules
- `grep` is literal text search, not regex.
- Never use `|`, `||`, regex groups, or regex OR in `grep(pattern=...)`.
- For OR behavior, run multiple separate `grep` calls with one literal pattern each.
- If a `grep` call returns no matches and pattern contains regex-like syntax, rewrite into literal single-pattern calls.
- File tools use virtual absolute paths rooted at project cwd. Use `/file.py`, not `/Users/...`.
- For `edit_file`, always run `read_file` first and copy `old_string` exactly from file content without line-number prefixes.

## Refactor Workflow

When renaming a function, class, variable, or changing a signature:
1. Edit the definition in the source file using `edit_file` with 3-8 lines of unique surrounding context.
2. Search for ALL references to the old name using `grep` with a single literal pattern.
3. For each file with references, `read_file` it and then `edit_file` to update the import/call site.
4. Re-read changed files to verify the old name no longer appears and the new name is correct.

Never assume a single-file change is sufficient if other files depend on the symbol.

## Sequential Edits

- After each successful `edit_file`, immediately re-read the file before making another edit.
- Never reuse an `old_string` from a previous read — always get fresh content.
- Avoid overlapping edits; use separate `edit_file` calls with unique context blocks.

## New Files vs Editing Existing Files

- To check if a file exists, use `ls` or try `read_file`.
- Use `write_file` for creating new files or full rewrites.
- Use `edit_file` only for small changes to existing files.
- Before `write_file`, ensure the parent directory exists.

## Error Handling and Retries

- If a tool returns an error (e.g., 'String not found', path errors), stop and analyze why.
- Do NOT loop on the same failing call. Make at most two adjusted attempts after reading fresh context.
- If still failing, explain the block and ask for guidance.

## Post-change Sanity

- After edits to code files, check for missing imports introduced by your change.
- If `new_string` uses a name that needs import (e.g., `os`, `Path`, `json`), ensure the import exists at the top of the file.
"""


@tool("think")
def think(thought: str = Field(..., description="A thought to think about.")) -> str:
    """Use this tool as scratchpad to structure intermediate reasoning."""
    return thought


class ThinkToolMiddleware(AgentMiddleware):
    """Inject local think tool into default toolset."""

    tools = [think]


def register_gigachat_harness_profile(working_directory: str | None = None) -> None:
    """Register HarnessProfile for GigaChat models."""
    _ = working_directory or os.getcwd()
    register_harness_profile(
        "giga",
        HarnessProfile(
            base_system_prompt=f"{MANUAL_BASE_SYSTEM_PROMPT}\n\n",
            tool_description_overrides={
                "ls": (
                    "Lists all files in a directory. "
                    "Use absolute virtual paths rooted at the current project working directory. "
                    "Do NOT use host OS paths like '/Users/...'."
                ),
                "read_file": (
                    "Reads a file from the filesystem. "
                    "The file_path must be an absolute virtual path starting with '/'. "
                    "Use offset/limit for large files (>200 lines) — pass integers, not strings. "
                    "Always read a file before editing it. "
                    "If a file is not found at the expected path, use glob to locate it by basename."
                ),
                "write_file": (
                    "Create or overwrite a file with exact content. Use absolute virtual paths starting with '/'. "
                    "Use write_file for new files or full rewrites; for small changes to existing files, use edit_file instead. "
                    "Ensure the parent directory exists before writing. "
                    "Do NOT include line-number prefixes from read_file output when writing content. "
                    "End the file with a trailing newline."
                ),
                "glob": (
                    "List files matching a pattern. Use absolute patterns like '/**/*.py'. "
                    "Useful for discovering file locations by basename before editing."
                ),
                "grep": (
                    "Search for a text pattern across files. "
                    "Pattern matching is literal text, NOT regex. "
                    "HARD RULE: pattern must represent ONE literal phrase only. "
                    "Never put multiple alternatives into one pattern. "
                    "DO NOT use '|', '||', regex groups, or regex syntax for OR. "
                    "If pattern contains '|', it is interpreted literally and almost always fails. "
                    "WRONG: pattern='os.getenv|load_dotenv|env|config|settings|getenv'. "
                    "RIGHT: run multiple grep calls: "
                    "1) pattern='os.getenv', "
                    "2) pattern='load_dotenv', "
                    "3) pattern='getenv', "
                    "4) pattern='config', "
                    "5) pattern='settings'. "
                    "Before each grep call, quickly verify: "
                    "(a) single literal phrase, "
                    "(b) no pipe '|', "
                    "(c) optional path/glob uses absolute virtual paths. "
                    "Special characters like '|', '(', ')', '[', ']' are treated as plain characters, not operators."
                ),
                "edit_file": (
                    "Edit an existing file by replacing one exact old_string with new_string. "
                    "Before edit_file always read the target file with read_file. "
                    "Use absolute virtual paths, not '/Users/...'. "
                    "Copy old_string 1:1 from read_file output (without line numbers), "
                    "including blank lines and indentation. Include 3-8 lines of context "
                    "above and below so old_string is unique. If edit_file returns "
                    "'String not found', do not guess: re-read the file with more context and "
                    "rebuild old_string from fresh content. "
                    "WRONG: old_string contains line-number prefixes like '40\\tdef foo():'. "
                    "RIGHT: remove prefixes and use exact code text 'def foo():'. "
                    "WRONG: old_string is a hand-written approximation or misses blank lines/indentation. "
                    "RIGHT: copy exact bytes from read_file, preserving spacing. "
                    "WRONG: replacing a common short fragment like 'return kwargs' without unique context. "
                    "RIGHT: include surrounding lines (function signature + nearby lines) so the match is unique. "
                    "After each successful edit, re-read the file to confirm the change. "
                    "If new_string uses a name that needs import, ensure the import exists."
                ),
            },
            extra_middleware=(ThinkToolMiddleware(),),
        ),
    )
