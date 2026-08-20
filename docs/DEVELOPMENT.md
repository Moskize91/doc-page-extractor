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

`.env` is ignored by git. The package does not automatically load it; local
debugging scripts read it explicitly when they need private OCR settings.

For VGE/Conductor worktrees, `setup` creates `.env` automatically from `.env.template` when missing.

`.env` now stores multiple backend configurations at the same time:

- `DEEPSEEK_OCR_*` for DeepSeek OCR Vendor.
- `DEEPSEEK_OCR2_*` for DeepSeek OCR 2 Vendor.
- `UNLIMITED_OCR_*` for Unlimited OCR.
- `DEEPSEEK_LOCAL_MODEL_PATH` and `DEEPSEEK_LOCAL_ONLY` for the local CUDA path.

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

macOS development should use `create_page_extractor_with_adapter()` for remote backend work and fake adapters. Do not call `create_ocr_page_extractor().load_ocr_model()` unless you are on a CUDA-capable Linux/NVIDIA environment.

Adapter code implements the OCR adapter protocol and returns page results directly:

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
    DeepSeekOCRVendorConfig(
        base_url="https://example.test/openai",
        api_key="...",
        model="deepseek-ocr",
    )
)
deepseek_ocr2 = create_deepseek_ocr2_vendor_page_extractor(
    DeepSeekOCR2VendorConfig(
        base_url="https://example.test/openai",
        api_key="...",
        model="deepseek-ocr2",
    )
)
unlimited_ocr = create_unlimited_ocr_page_extractor(
    UnlimitedOCRConfig(
        ak="...",
        sk="...",
    )
)
```

### Layout Contract

`Layout.kind` is the stable layout semantic. `Layout.type` and `Layout.raw`
preserve provider-specific data and should not be treated as stable
cross-provider contracts.

```python
from PIL import Image

for image, result in extractor.extract_page_results(
    image=Image.open("page.png"),
    size="gundam",
    stages=1,
):
    for layout in result.layouts:
        print(layout.kind, layout.det, layout.text)
```

### OCR Sample

After filling private settings in `.env`, run:

```shell
poetry run python scripts/ocr_sample.py --adapter deepseek-ocr-vendor --image tests/images/friendly-title.png
poetry run python scripts/ocr_sample.py --adapter deepseek-ocr2-vendor --image tests/images/friendly-title.png
poetry run python scripts/ocr_sample.py --adapter unlimited-ocr --image tests/images/friendly-title.png
```

The sample reads `tests/images/friendly-title.png`, runs the configured OCR adapter, and prints layout summaries, including `kind`, provider `type`, text previews, and elapsed time. Use `--image path/to/image.png` to try another image.

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
