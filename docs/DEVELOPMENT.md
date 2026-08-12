# Development Guide

## Setup

Use Poetry with an in-project virtual environment. This keeps each git worktree isolated while still reusing Poetry and pip download caches.

```shell
pipx install poetry==2.1.3
PYTHON_BIN="$(pyenv which python3 2>/dev/null || command -v python3)"
"$PYTHON_BIN" -m venv .venv
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
poetry install --only dev
```

PyTorch and the model runtime dependencies are intentionally not installed for the default development setup. The lightweight test and lint workflow does not need to load the OCR model.

## Development Workflow

### Run Tests

```shell
poetry run python test.py
```

### Run Lint

Check code quality with pylint:

```shell
poetry run pylint --disable=import-error doc_page_extractor
```

### Build Package

Clean old builds and create distribution files:

```shell
python build.py
```

## Before Submitting PR

Make sure all checks pass:

```shell
poetry run python test.py
poetry run pylint --disable=import-error doc_page_extractor
```

## VGE Worktree Workflow

VGE uses `.conductor/settings.toml` for worktree lifecycle commands:

- `setup` creates or updates the in-project `.venv` with Poetry.
- `run` executes the lightweight checks: parser tests and pylint.
- `archive` is shown by VGE as cleanup and removes generated build and cache files while keeping `.venv` for faster reuse.
