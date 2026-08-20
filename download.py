from __future__ import annotations

import argparse
import time
from pathlib import Path

from doc_page_extractor import (
    create_deepseek_ocr_page_extractor,
    create_unlimited_ocr_page_extractor,
)


_REVISIONS = {
    "deepseek-ocr": "9f30c71f441d010e5429c532364a86705536c53a",
    "deepseek-ocr2": "aaa02f3811945a91062062994c5c4a3f4c0af2b0",
    "unlimited-ocr": "07dea832e22aefee32ad281d4b80551282e1c168",
}


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).parent
    model_path = args.model_path
    if not model_path.is_absolute():
        model_path = project_root / model_path
    extractor = _create_extractor(args.ocr_model, model_path)
    begin_at = time.time()
    extractor.download_ocr_model(args.revision or _REVISIONS[args.ocr_model])
    print(f"Models downloaded cost {time.time() - begin_at:.2f} seconds.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a local OCR model.")
    parser.add_argument(
        "--ocr-model",
        choices=("deepseek-ocr", "deepseek-ocr2", "unlimited-ocr"),
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
            "Model revision to download. Default: pinned revision for the "
            "selected OCR model."
        ),
    )
    return parser.parse_args()


def _create_extractor(ocr_model: str, model_path: Path):
    if ocr_model == "unlimited-ocr":
        return create_unlimited_ocr_page_extractor(
            model_path=model_path,
            local_only=False,
        )
    return create_deepseek_ocr_page_extractor(
        ocr_model=ocr_model,  # type: ignore[arg-type]
        model_path=model_path,
        local_only=False,
    )


if __name__ == "__main__":
    main()
