# GigaChat Harness Profile

A minimal repository with a custom `HarnessProfile` for `deepagents` + `langchain-gigachat`.

## Structure

- `gigachat_harness_profile.py` - registers the GigaChat profile (`register_gigachat_harness_profile`)
- `run_gigachat_harness_profile.py` - example runner that uses this profile

## Requirements

- Python 3.12+
- `uv` (for dependency installation and execution)

## Installation

```bash
uv sync
```

## Configuration

Create a `.env` file and provide one of the authentication options:

- `GIGACHAT_CREDENTIALS`
- or `GIGACHAT_USER` + `GIGACHAT_PASSWORD`

## Run

```bash
uv run python run_gigachat_harness_profile.py
```

## Important Notes About Filesystem Tools

The example runner uses `FilesystemBackend(..., virtual_mode=True)`, so filesystem tool paths must be virtual absolute paths, for example:

- `/gigachat_harness_profile.py`
- `/run_gigachat_harness_profile.py`

Do not use host OS paths such as `/Users/...`.
