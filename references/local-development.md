# macOS Local Development

## Default Setup

Use Poetry with an in-project virtual environment:

```shell
pipx install poetry==2.1.3
PYTHON_BIN="$(pyenv which python3 2>/dev/null || command -v python3)"
"$PYTHON_BIN" -m venv .venv
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
poetry install --only dev
```

This setup intentionally avoids CUDA PyTorch and model downloads. It is enough for parser tests, package import checks, lint, and development-model tests.

## Local Environment File

Copy `.env.template` to `.env` for machine-specific values:

```shell
cp .env.template .env
```

`.env` is ignored by git. It may contain private Vendor API settings or local model paths. The library does not automatically load `.env`; source it when a script or manual command needs those values:

```shell
set -a && source .env && set +a
```

Keep `.env.template` free of secrets.

## Verification

Default checks for macOS:

```shell
poetry run python test.py
poetry run pylint --disable=import-error doc_page_extractor
```

Do not run `main.py`, `download.py`, or `PageExtractor.load_models()` on macOS unless the task explicitly asks for a real backend experiment.

## Development Backend Pattern

Use `create_page_extractor_with_model()` for local tests that need the full extraction loop without CUDA:

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

This exercises image saving, response parsing, layout construction, staged redaction, and context token accounting if the fixture updates `context`.

## What macOS Cannot Prove

macOS checks do not validate CUDA availability, GPU memory behavior, Hugging Face remote code compatibility, `flash_attn`, or multi-GPU device routing. Those require a Linux/NVIDIA environment.
