#!/usr/bin/env python3
"""Run OCR samples through unified adapters."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image

from doc_page_extractor import (
    DeepSeekOCR2VendorAdapter,
    DeepSeekOCR2VendorConfig,
    DeepSeekOCRVendorAdapter,
    DeepSeekOCRVendorConfig,
    ExtractionContext,
    UnlimitedOCRAdapter,
    UnlimitedOCRConfig,
    create_ocr_page_extractor,
    create_page_extractor_with_adapter,
)

_DEFAULT_IMAGE = Path("tests/images/friendly-title.png")
_DEFAULT_SIZE = "gundam"


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    _load_dotenv(project_root / ".env")
    image_path = _resolve_path(project_root, args.image)

    if args.adapter == "deepseek-ocr-vendor":
        result = _run_deepseek_ocr_vendor(image_path, args.size, args.limit)
        _print_result(result)
        return

    if args.adapter == "deepseek-ocr2-vendor":
        result = _run_deepseek_ocr2_vendor(image_path, args.size, args.limit)
        _print_result(result)
        return

    if args.adapter == "deepseek-ocr-local":
        result = _run_deepseek_ocr_local(project_root, image_path, args.size, args.limit)
        _print_result(result)
        return

    if args.adapter == "deepseek-ocr2-local":
        result = _run_deepseek_ocr2_local(project_root, image_path, args.size, args.limit)
        _print_result(result)
        return

    if args.adapter == "unlimited-ocr":
        result = _run_unlimited_ocr(image_path, args.size, args.limit)
        _print_result(result)
        return

    if args.adapter == "all":
        deepseek_ocr_result = _run_deepseek_ocr_vendor(
            image_path, args.size, args.limit
        )
        deepseek_ocr2_result = _run_deepseek_ocr2_vendor(
            image_path, args.size, args.limit
        )
        deepseek_ocr_local_result = _run_deepseek_ocr_local(
            project_root, image_path, args.size, args.limit
        )
        deepseek_ocr2_local_result = _run_deepseek_ocr2_local(
            project_root, image_path, args.size, args.limit
        )
        unlimited_ocr_result = _run_unlimited_ocr(image_path, args.size, args.limit)
        print(json.dumps({
            "adapter": "all",
            "image": str(image_path),
            "deepseek_ocr_vendor": deepseek_ocr_result,
            "deepseek_ocr2_vendor": deepseek_ocr2_result,
            "deepseek_ocr_local": deepseek_ocr_local_result,
            "deepseek_ocr2_local": deepseek_ocr2_local_result,
            "unlimited_ocr": unlimited_ocr_result,
        }, ensure_ascii=False, indent=2))
        return

    raise SystemExit(f"Unsupported adapter: {args.adapter}")


def _run_deepseek_ocr_vendor(
    image_path: Path, size: str, limit: int
) -> dict[str, Any]:
    config = DeepSeekOCRVendorConfig.from_env()
    extractor = create_page_extractor_with_adapter(DeepSeekOCRVendorAdapter(config))
    return _run_extractor("deepseek-ocr-vendor", extractor, image_path, size, limit)


def _run_deepseek_ocr2_vendor(
    image_path: Path, size: str, limit: int
) -> dict[str, Any]:
    config = DeepSeekOCR2VendorConfig.from_env()
    extractor = create_page_extractor_with_adapter(DeepSeekOCR2VendorAdapter(config))
    return _run_extractor("deepseek-ocr2-vendor", extractor, image_path, size, limit)


def _run_deepseek_ocr_local(
    project_root: Path, image_path: Path, size: str, limit: int
) -> dict[str, Any]:
    extractor = create_ocr_page_extractor(
        ocr_model="deepseek-ocr",
        model_path=project_root / "models-cache",
        local_only=True,
    )
    return _run_extractor("deepseek-ocr-local", extractor, image_path, size, limit)


def _run_deepseek_ocr2_local(
    project_root: Path, image_path: Path, size: str, limit: int
) -> dict[str, Any]:
    extractor = create_ocr_page_extractor(
        ocr_model="deepseek-ocr2",
        model_path=project_root / "models-cache",
        local_only=True,
    )
    return _run_extractor("deepseek-ocr2-local", extractor, image_path, size, limit)


def _run_unlimited_ocr(image_path: Path, size: str, limit: int) -> dict[str, Any]:
    config = UnlimitedOCRConfig.from_env()
    extractor = create_page_extractor_with_adapter(UnlimitedOCRAdapter(config))
    return _run_extractor("unlimited-ocr", extractor, image_path, size, limit)


def _run_extractor(adapter_name: str, extractor, image_path: Path, size: str, limit: int) -> dict[str, Any]:
    context = ExtractionContext(check_aborted=lambda: False)
    started = time.monotonic()
    layouts: list[Any] = []
    for _, stage_layouts in extractor.extract(
        image=Image.open(image_path),
        size=size,  # type: ignore[arg-type]
        stages=1,
        context=context,
    ):
        layouts = stage_layouts
    elapsed = round(time.monotonic() - started, 3)
    payload = {
        "adapter": adapter_name,
        "image": str(image_path),
        "elapsed_seconds": elapsed,
        "layout_count": len(layouts),
        "tokens": {
            "input": context.input_tokens,
            "output": context.output_tokens,
        },
        "layouts": [
            {
                "index": index + 1,
                "ref": layout.ref,
                "kind": layout.kind.value,
                "det": layout.det,
                "type": layout.type,
                "text": _preview_text(layout.text),
            }
            for index, layout in enumerate(layouts[:limit])
        ],
    }
    if layouts:
        payload["first_layout"] = _layout_summary(layouts[0])
    return payload


def _layout_summary(layout: Any) -> dict[str, Any]:
    return {
        "ref": layout.ref,
        "kind": layout.kind.value,
        "det": layout.det,
        "type": layout.type,
        "text": _preview_text(layout.text),
    }


def _print_result(result: dict[str, Any]) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified OCR samples through adapters.")
    parser.add_argument(
        "--adapter",
        choices=(
            "deepseek-ocr-vendor",
            "deepseek-ocr2-vendor",
            "deepseek-ocr-local",
            "deepseek-ocr2-local",
            "unlimited-ocr",
            "all",
        ),
        required=True,
        help="OCR adapter to run.",
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
        return
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
    return normalized[:120] + ("..." if len(normalized) > 120 else "")


if __name__ == "__main__":
    main()
