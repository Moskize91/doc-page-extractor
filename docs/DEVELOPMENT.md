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

- `DPE_DEEPSEEK_OCR_*` for DeepSeek OCR Vendor.
- `DPE_DEEPSEEK_OCR2_*` for DeepSeek OCR 2 Vendor.
- `DPE_UNLIMITED_OCR_*` for Unlimited OCR.
- `DPE_DEEPSEEK_LOCAL_MODEL_PATH` and `DPE_DEEPSEEK_LOCAL_ONLY` for the local CUDA path.

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

macOS development should use `create_page_extractor_with_adapter()` for new backend work, or `create_page_extractor_with_model()` when testing the legacy DeepSeek model protocol. Do not call `create_page_extractor().load_models()` unless you are on a CUDA-capable Linux/NVIDIA environment.

New adapter code should implement the unified OCR adapter protocol and return layout results directly:

```python
from doc_page_extractor import (
    DeepSeekOCR2VendorConfig,
    DeepSeekOCRVendorConfig,
    UnlimitedOCRConfig,
    create_deepseek_ocr2_vendor_page_extractor,
    create_deepseek_ocr_vendor_page_extractor,
    create_unlimited_ocr_page_extractor,
)

deepseek_ocr = create_deepseek_ocr_vendor_page_extractor(
    DeepSeekOCRVendorConfig.from_env()
)
deepseek_ocr2 = create_deepseek_ocr2_vendor_page_extractor(
    DeepSeekOCR2VendorConfig.from_env()
)
unlimited_ocr = create_unlimited_ocr_page_extractor(UnlimitedOCRConfig.from_env())
```

### Layout Contract

New code should prefer `Layout.kind` for stable layout semantics. `Layout.ref`
is kept for backward compatibility with the original DeepSeek-style API, while
`Layout.type` and `Layout.raw` preserve provider-specific data and should not be
treated as stable cross-provider contracts.

Use `extract_page_results()` when you need `OCRPageResult.structured`; keep using
`extract()` when the legacy flat `list[Layout]` result is enough.

The legacy model protocol remains available for small fixtures:

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
poetry run python scripts/ocr_sample.py --adapter deepseek-ocr-vendor --image tests/images/friendly-title.png
poetry run python scripts/ocr_sample.py --adapter deepseek-ocr2-vendor --image tests/images/friendly-title.png
poetry run python scripts/ocr_sample.py --adapter unlimited-ocr --image tests/images/friendly-title.png
```

The sample reads `tests/images/friendly-title.png`, runs the configured OCR adapter, and prints layout summaries, including `ref`, `kind`, provider `type`, text previews, and elapsed time. Use `--image path/to/image.png` to try another image.

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
