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

For VGE/Conductor worktrees, `setup` creates `.env` automatically from `.env.template` when missing.

`.env` now stores multiple backend configurations at the same time:

- `DOC_PAGE_EXTRACTOR_DEEPSEEK_VENDOR_*` for the OpenAI-compatible DeepSeek Vendor.
- `DOC_PAGE_EXTRACTOR_BAIDU_*` for Baidu Unlimited-OCR.
- `DOC_PAGE_EXTRACTOR_MODEL_PATH` and `DOC_PAGE_EXTRACTOR_LOCAL_ONLY` for the local CUDA path.

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

### OCR Sample

After filling private settings in `.env`, run:

```shell
poetry run python scripts/ocr_sample.py --adapter deepseek-vendor --image tests/images/friendly-title.png
poetry run python scripts/ocr_sample.py --adapter baidu --image tests/images/friendly-title.png
poetry run python scripts/ocr_sample.py --adapter both --image tests/images/friendly-title.png
```

The sample reads `tests/images/friendly-title.png`, runs the configured DeepSeek Vendor backend, the Baidu cloud backend, or both, and prints layout summaries, text previews, and elapsed time. Use `--image path/to/image.png` to try another image.

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
