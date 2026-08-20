import sys
import tempfile
import warnings
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Iterable

from .adapters.unlimited import UnlimitedOCRAdapter, UnlimitedOCRConfig
from .adapters.deepseek import (
    DeepSeekOCR2VendorAdapter,
    DeepSeekOCR2VendorConfig,
    DeepSeekModelOCRAdapter,
    DeepSeekOCRVendorAdapter,
    DeepSeekOCRVendorConfig,
)
from .types import (
    DeepSeekOCRSize,
    OCRModel,
    OCRModelName,
    ExtractionContext,
    Layout,
    OCRAdapter,
    OCRPageResult,
    PageExtractor,
)
from .structure import build_structured_page

if TYPE_CHECKING:
    from PIL import Image

_DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."


def create_page_extractor(
    model_path: PathLike | str | None = None,
    local_only: bool = False,
    enable_devices_numbers: Iterable[int] | None = None,
) -> PageExtractor:
    return create_ocr_page_extractor(
        ocr_model="deepseek-ocr",
        model_path=model_path,
        local_only=local_only,
        enable_devices_numbers=enable_devices_numbers,
    )


def create_ocr_page_extractor(
    ocr_model: OCRModelName = "deepseek-ocr",
    model_path: PathLike | str | None = None,
    local_only: bool = False,
    enable_devices_numbers: Iterable[int] | None = None,
) -> PageExtractor:
    if ocr_model == "deepseek-ocr":
        from .model import DeepSeekOCRHugginfaceModel

        model = DeepSeekOCRHugginfaceModel(
            model_path=Path(model_path) if model_path else None,
            local_only=local_only,
            enable_devices_numbers=enable_devices_numbers,
        )
    elif ocr_model == "deepseek-ocr2":
        from .model import DeepSeekOCR2HugginfaceModel

        model = DeepSeekOCR2HugginfaceModel(
            model_path=Path(model_path) if model_path else None,
            local_only=local_only,
            enable_devices_numbers=enable_devices_numbers,
        )
    else:
        raise ValueError(f"Unsupported OCR model: {ocr_model}")

    return _PageExtractorImpls(DeepSeekModelOCRAdapter(model, source=ocr_model))


def create_page_extractor_with_model(model: OCRModel) -> PageExtractor:
    if not isinstance(model, OCRModel):
        raise TypeError("model must implement OCRModel protocol")
    return _PageExtractorImpls(DeepSeekModelOCRAdapter(model))


def create_page_extractor_with_adapter(adapter: OCRAdapter) -> PageExtractor:
    if not hasattr(adapter, "supports_multi_stage"):
        setattr(adapter, "supports_multi_stage", True)
    if not isinstance(adapter, OCRAdapter):
        raise TypeError("adapter must implement OCRAdapter protocol")
    return _PageExtractorImpls(adapter)


def create_deepseek_ocr_vendor_page_extractor(
    config: DeepSeekOCRVendorConfig,
) -> PageExtractor:
    return _PageExtractorImpls(DeepSeekOCRVendorAdapter(config))


def create_deepseek_ocr2_vendor_page_extractor(
    config: DeepSeekOCR2VendorConfig,
) -> PageExtractor:
    return _PageExtractorImpls(DeepSeekOCR2VendorAdapter(config))


def create_unlimited_ocr_page_extractor(
    config: UnlimitedOCRConfig,
) -> PageExtractor:
    return _PageExtractorImpls(UnlimitedOCRAdapter(config))


class _PageExtractorImpls:
    def __init__(self, adapter: OCRAdapter) -> None:
        self._adapter: OCRAdapter = adapter

    def download_models(self, revision: str | None = None) -> None:
        downloader = getattr(self._adapter, "download", None)
        if downloader is not None:
            downloader(revision)

    def load_models(self) -> None:
        loader = getattr(self._adapter, "load", None)
        if loader is not None:
            loader()

    def extract(
        self,
        image: "Image.Image",
        size: DeepSeekOCRSize,
        stages: int = 1,
        context: ExtractionContext | None = None,
        device_number: int | None = None,
    ) -> Generator[tuple["Image.Image", list[Layout]], None, None]:
        for stage_image, page_result in self.extract_page_results(
            image=image,
            size=size,
            stages=stages,
            context=context,
            device_number=device_number,
        ):
            yield stage_image, page_result.layouts

    def extract_page_results(
        self,
        image: "Image.Image",
        size: DeepSeekOCRSize,
        stages: int = 1,
        context: ExtractionContext | None = None,
        device_number: int | None = None,
    ) -> Generator[tuple["Image.Image", OCRPageResult], None, None]:
        assert stages >= 1, "stages must be at least 1"
        if stages > 1 and not getattr(self._adapter, "supports_multi_stage", True):
            warnings.warn(
                "This OCR adapter does not support multi-stage redaction; "
                "using a single extraction stage.",
                RuntimeWarning,
                stacklevel=2,
            )
            stages = 1

        fill_color: tuple[int, int, int] | None = None
        output_path: Path | None = None
        temp_dir: tempfile.TemporaryDirectory | None = None

        if context and context.output_dir_path:
            output_path = Path(context.output_dir_path)
        else:
            temp_dir = tempfile.TemporaryDirectory()
            output_path = Path(temp_dir.name)

        try:
            for i in range(stages):
                image_path = output_path / f"raw-{i+1}.png"
                image.save(image_path, "PNG")
                prepared_image_path, scale_x, scale_y = _prepare_adapter_image(
                    image=image,
                    image_path=image_path,
                    output_path=output_path,
                    max_image_side=getattr(self._adapter, "max_image_side", None),
                )
                try:
                    page_result = self._adapter.extract_page(
                        prompt=_DEFAULT_PROMPT,
                        image_path=prepared_image_path,
                        output_path=output_path,
                        size=size,
                        context=context,
                        device_number=device_number,
                    )
                finally:
                    if prepared_image_path != image_path:
                        prepared_image_path.unlink(missing_ok=True)
                    image_path.unlink(missing_ok=True)

                layouts = page_result.layouts
                if scale_x != 1.0 or scale_y != 1.0:
                    _scale_layout_coordinates(layouts, scale_x, scale_y)
                    if page_result.structured is not None:
                        page_result.structured = build_structured_page(layouts)
                if page_result.structured is None:
                    page_result.structured = build_structured_page(layouts)
                yield image, page_result

                if i < stages - 1:
                    from .redacter import background_color, redact

                    if fill_color is None:
                        fill_color = background_color(image)
                    image = redact(
                        image=image.copy(),
                        fill_color=fill_color,
                        rectangles=self._redact_rectangles(
                            image=image,
                            dets=(layout.det for layout in layouts),
                        ),
                    )
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    def _redact_rectangles(
        self, image: "Image.Image", dets: Iterable[tuple[int, int, int, int]]
    ):
        # 将页面上 2/3 全部涂抹，并沿着 2/3 线向下涂抹到每一个识别为文字区块的底部
        # 这种方法旨在涂抹掉尽可能多的不是页脚的区域，以排除诸如页眉之类干扰识别页脚的内容
        rate = float(2 / 3)
        width, height = image.size
        y_cutted = round(height * rate)
        yield (0, 0, width, y_cutted)
        yield from self._redact_button_rectangles(y_cutted, dets)

    def _redact_button_rectangles(
        self, y_cutted: int, dets: Iterable[tuple[int, int, int, int]]
    ):
        parts: list[tuple[int, int, int]] = []  # x1, x2, height
        for det in dets:
            x1, _, x2, y2 = det
            height = y2 - y_cutted
            if height > 0:
                parts.append((x1, x2, height))

        parts.sort()
        forbidden: int = -sys.maxsize

        for i, (x1, x2, height) in enumerate(parts):
            left = max(x1, forbidden)
            right = x2
            for j in range(i + 1, len(parts)):
                nx1, _, nheight = parts[j]
                if nheight > height:
                    right = min(right, nx1)
            if left < right:
                yield (left, y_cutted, right, y_cutted + height)
                forbidden = right


def _prepare_adapter_image(
    image: "Image.Image",
    image_path: Path,
    output_path: Path,
    max_image_side: int | None,
) -> tuple[Path, float, float]:
    if max_image_side is None:
        return image_path, 1.0, 1.0

    width, height = image.size
    max_side = max(width, height)
    if max_side <= max_image_side:
        return image_path, 1.0, 1.0

    ratio = max_image_side / max_side
    resized_width = max(1, round(width * ratio))
    resized_height = max(1, round(height * ratio))
    resized_path = output_path / f"{image_path.stem}-resized.png"
    resized = image.resize((resized_width, resized_height))
    resized.save(resized_path, "PNG")

    return resized_path, width / resized_width, height / resized_height


def _scale_layout_coordinates(
    layouts: list[Layout], scale_x: float, scale_y: float
) -> None:
    for layout in layouts:
        x1, y1, x2, y2 = layout.det
        layout.det = (
            round(x1 * scale_x),
            round(y1 * scale_y),
            round(x2 * scale_x),
            round(y2 * scale_y),
        )
        if layout.polygon is not None:
            layout.polygon = [
                (round(x * scale_x), round(y * scale_y)) for x, y in layout.polygon
            ]
