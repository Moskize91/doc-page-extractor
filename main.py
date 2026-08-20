from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image

from doc_page_extractor import ExtractionContext, create_ocr_page_extractor, plot

_IMAGE_STEM = "friendly-title"
_ABORT_TIMEOUT = 9999.0  # seconds


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).parent
    image_dir_path = project_root / "tests" / "images"
    image_name = f"{args.image_stem}.png"
    extractor = create_ocr_page_extractor(
        ocr_model=args.ocr_model,
        model_path=args.model_path,
        local_only=args.local_only,
    )
    begin_at = time.time()
    extractor.load_models()
    print(f"Models loaded in {time.time() - begin_at:.2f} seconds.")

    def check_aborted() -> bool:
        if time.time() - begin_at > _ABORT_TIMEOUT:
            print("Aborted extraction due to timeout.")
            return True
        return False

    plot_dir = project_root / "plot"
    plot_dir.mkdir(exist_ok=True)
    name_stem = Path(image_name).stem
    name_suffix = Path(image_name).suffix
    context = ExtractionContext(check_aborted=check_aborted)

    print("Starting extraction...")
    for i, (image, layouts) in enumerate(
        extractor.extract(
            image=Image.open(image_dir_path / image_name),
            size=args.size,
            stages=args.stages,
            context=context,
        )
    ):
        print("Layouts:")
        for layout in layouts:
            print(f"  Ref: {layout.ref}, Det: {layout.det}, Text: {layout.text}")
        image = plot(image.copy(), layouts)
        output_path = plot_dir / f"{name_stem}_{i}{name_suffix}"
        image.save(output_path)

    print(f"Extraction cost {time.time() - begin_at:.2f} seconds.")
    print(
        f"Input tokens: {context.input_tokens}; Output tokens: {context.output_tokens}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local OCR extraction.")
    parser.add_argument(
        "--ocr-model",
        choices=("deepseek-ocr", "deepseek-ocr2"),
        default="deepseek-ocr",
        help="Local OCR model to use. Default: deepseek-ocr",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models-cache"),
        help="Hugging Face cache directory. Default: models-cache",
    )
    parser.add_argument(
        "--local-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load from the local Hugging Face cache only. Default: true",
    )
    parser.add_argument(
        "--image-stem",
        default=_IMAGE_STEM,
        help=f"Test image stem under tests/images. Default: {_IMAGE_STEM}",
    )
    parser.add_argument(
        "--size",
        choices=("tiny", "small", "base", "large", "gundam"),
        default="gundam",
        help="DeepSeek-OCR size preset. Default: gundam",
    )
    parser.add_argument(
        "--stages",
        type=int,
        default=2,
        help="Extraction stages. Default: 2",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
