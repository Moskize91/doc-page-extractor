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

For machine-specific local values, copy the environment template:

```shell
cp .env.template .env
```

`.env` is ignored by git. The package does not automatically load it; source it only for scripts or development adapters that need those values:

```shell
set -a && source .env && set +a
```

For VGE/Conductor worktrees, `setup` creates `.env` automatically. It copies `$DOC_PAGE_EXTRACTOR_ENV_FILE` when set, otherwise `~/.config/doc-page-extractor/.env` when present, otherwise `.env.template`.

`DOC_PAGE_EXTRACTOR_BACKEND` is mutually exclusive:

- `fixture`: local fixed-response backend; no CUDA and no network.
- `vendor`: OpenAI-compatible remote OCR backend.
- `local`: local Hugging Face DeepSeek-OCR backend; requires CUDA.

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

### macOS Model-Free Development

macOS development should use `create_page_extractor_with_model()` with a fixture or remote backend. Do not call `create_page_extractor().load_models()` unless you are on a CUDA-capable Linux/NVIDIA environment.

```python
from pathlib import Path
from doc_page_extractor import create_page_extractor_with_model


class FixtureOCRModel:
    def download(self, revision: str | None) -> None:
        pass

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def generate(self, prompt, image_path: Path, output_path: Path, size, context, device_number) -> str:
        return "<|ref|>sample<|/ref|><|det|>[[100, 100, 500, 200]]<|/det|>hello"


extractor = create_page_extractor_with_model(FixtureOCRModel())
```

### Vendor OCR Sample

After filling private Vendor settings in `.env`, run:

```shell
poetry run python scripts/vendor_ocr_sample.py
```

The sample reads `tests/images/friendly-title.png`, calls the configured OpenAI-compatible OCR endpoint, and routes the response through `create_page_extractor_with_model()`. Successful output includes the image path, layout count, the first few `ref`/`det` pairs, text previews, and token usage. Use `--image path/to/image.png` to try another image.

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

## Agent Documentation

Agents should start with the repository-level `AGENTS.md`. It routes work to focused reference documents under `references/` and keeps CUDA-specific model work separate from macOS-safe development.
