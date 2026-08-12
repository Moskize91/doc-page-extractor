#!/usr/bin/env python3
"""Run a minimal Vendor OCR sample through PageExtractor."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from doc_page_extractor.extractor import create_page_extractor_with_model
from doc_page_extractor.types import DeepSeekOCRSize, ExtractionContext

_DEFAULT_IMAGE = Path("tests/images/friendly-title.png")
_DEFAULT_SIZE: DeepSeekOCRSize = "gundam"
_MAX_TOKENS = 8000
_TEXT_PREVIEW_LIMIT = 120


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    _load_dotenv(project_root / ".env")

    image_path = _resolve_path(project_root, args.image)
    config = _VendorConfig.from_env()
    if config.backend != "vendor":
        raise SystemExit(
            "DOC_PAGE_EXTRACTOR_BACKEND must be 'vendor' to run this sample. "
            f"Current value: {config.backend!r}"
        )

    extractor = create_page_extractor_with_model(_VendorOCRModel(config))
    context = ExtractionContext(check_aborted=lambda: False)

    print(f"Image: {image_path}")
    print(f"Vendor URL: {config.base_url}")
    print(f"Vendor model: {config.model}")

    for stage_index, (_, layouts) in enumerate(
        extractor.extract(
            image=Image.open(image_path),
            size=args.size,
            stages=1,
            context=context,
        ),
        start=1,
    ):
        print(f"Stage: {stage_index}")
        print(f"Layouts: {len(layouts)}")
        for index, layout in enumerate(layouts[: args.limit], start=1):
            print(
                json.dumps(
                    {
                        "index": index,
                        "ref": layout.ref,
                        "det": layout.det,
                        "text": _preview_text(layout.text),
                    },
                    ensure_ascii=False,
                )
            )
        if not layouts:
            raise SystemExit("Vendor OCR returned no parseable layouts.")

    print(f"Tokens: input={context.input_tokens} output={context.output_tokens}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Vendor OCR against a sample image through create_page_extractor_with_model().",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=_DEFAULT_IMAGE,
        help=f"Image path relative to the project root. Default: {_DEFAULT_IMAGE}",
    )
    parser.add_argument(
        "--size",
        choices=("tiny", "small", "base", "large", "gundam"),
        default=_DEFAULT_SIZE,
        help=f"DeepSeek-OCR size preset. Default: {_DEFAULT_SIZE}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum layouts to print. Default: 3",
    )
    return parser.parse_args()


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Copy .env.template to .env and fill Vendor settings first."
        )
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _resolve_path(project_root: Path, path: Path) -> Path:
    resolved = path if path.is_absolute() else project_root / path
    if not resolved.exists():
        raise SystemExit(f"Image does not exist: {resolved}")
    return resolved


def _preview_text(text: str | None) -> str | None:
    if text is None:
        return None
    normalized = " ".join(text.split())
    if len(normalized) <= _TEXT_PREVIEW_LIMIT:
        return normalized
    return f"{normalized[:_TEXT_PREVIEW_LIMIT]}..."


class _VendorConfig:
    def __init__(
        self,
        *,
        backend: str,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float | None,
        top_p: float | None,
    ) -> None:
        self.backend = backend
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    @classmethod
    def from_env(cls) -> "_VendorConfig":
        return cls(
            backend=_required_env("DOC_PAGE_EXTRACTOR_BACKEND"),
            base_url=_required_env("DOC_PAGE_EXTRACTOR_VENDOR_BASE_URL"),
            api_key=_required_env("DOC_PAGE_EXTRACTOR_VENDOR_API_KEY"),
            model=_required_env("DOC_PAGE_EXTRACTOR_VENDOR_MODEL"),
            temperature=_optional_float("DOC_PAGE_EXTRACTOR_VENDOR_TEMPERATURE"),
            top_p=_optional_float("DOC_PAGE_EXTRACTOR_VENDOR_TOP_P"),
        )


class _VendorOCRModel:
    def __init__(self, config: _VendorConfig) -> None:
        self._config = config

    def download(self, revision: str | None) -> None:
        del revision

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def generate(
        self,
        prompt: str,
        image_path: Path,
        output_path: Path,
        size: DeepSeekOCRSize,
        context: ExtractionContext | None,
        device_number: int | None,
    ) -> str:
        del output_path, size, device_number
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(image_path)},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": _MAX_TOKENS,
            "stream": False,
        }
        if self._config.temperature is not None:
            payload["temperature"] = self._config.temperature
        if self._config.top_p is not None:
            payload["top_p"] = self._config.top_p

        response = requests.post(
            f"{self._config.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "doc-page-extractor-vendor-sample/1.0",
            },
            json=payload,
            timeout=180,
        )
        if response.status_code >= 400:
            _raise_vendor_error(response)

        data = response.json()
        usage = data.get("usage") or {}
        if context is not None:
            context.input_tokens += int(usage.get("prompt_tokens") or 0)
            context.output_tokens += int(usage.get("completion_tokens") or 0)

        choices = data.get("choices") or []
        if not choices:
            return ""
        return str((choices[0].get("message") or {}).get("content") or "")


def _data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _raise_vendor_error(response: requests.Response) -> None:
    try:
        body = response.json()
    except ValueError:
        body = response.text
    raise SystemExit(
        f"Vendor request failed with HTTP {response.status_code}: "
        f"{_preview_text(json.dumps(body, ensure_ascii=False) if not isinstance(body, str) else body)}"
    )


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def _optional_float(name: str) -> float | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a float, got {value!r}") from exc


if __name__ == "__main__":
    main()
