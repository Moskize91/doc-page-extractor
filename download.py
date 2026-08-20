from __future__ import annotations

import argparse
import time
from pathlib import Path

from doc_page_extractor import create_ocr_page_extractor


_REVISION = "9f30c71f441d010e5429c532364a86705536c53a"


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).parent
    model_path = args.model_path
    if not model_path.is_absolute():
        model_path = project_root / model_path
    extractor = create_ocr_page_extractor(
        ocr_model=args.ocr_model,
        model_path=model_path,
        local_only=False,
    )
    begin_at = time.time()
    extractor.download_models(args.revision or _default_revision(args.ocr_model))
    print(f"Models downloaded cost {time.time() - begin_at:.2f} seconds.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a local OCR model.")
    parser.add_argument(
        "--ocr-model",
        choices=("deepseek-ocr", "deepseek-ocr2"),
        default="deepseek-ocr",
        help="OCR model to download. Default: deepseek-ocr",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models-cache"),
        help="Hugging Face cache directory. Default: models-cache",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help=(
            "Model revision to download. Default: pinned DeepSeek-OCR 1.0 hash "
            f"for deepseek-ocr, main for deepseek-ocr2."
        ),
    )
    return parser.parse_args()


def _default_revision(ocr_model: str) -> str | None:
    if ocr_model == "deepseek-ocr":
        return _REVISION
    return None


if __name__ == "__main__":
    main()
