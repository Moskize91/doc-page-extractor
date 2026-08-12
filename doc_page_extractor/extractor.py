import sys
import tempfile
from os import PathLike
from pathlib import Path
from typing import Generator, Iterable

from PIL import Image

from .adapters.baidu import BaiduCloudOCRAdapter, BaiduCloudOCRConfig
from .adapters.deepseek import (
    DeepSeekLocalOCRAdapter,
    DeepSeekModelOCRAdapter,
    DeepSeekVendorOCRAdapter,
    DeepSeekVendorOCRConfig,
)
from .redacter import background_color, redact
from .types import (
    DeepSeekOCRModel,
    DeepSeekOCRSize,
    ExtractionContext,
    Layout,
    OCRAdapter,
    PageExtractor,
)

_DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."


def create_page_extractor(
    model_path: PathLike | str | None = None,
    local_only: bool = False,
    enable_devices_numbers: Iterable[int] | None = None,
) -> PageExtractor:
    return _PageExtractorImpls(
        DeepSeekLocalOCRAdapter(
            model_path=model_path,
            local_only=local_only,
            enable_devices_numbers=enable_devices_numbers,
        )
    )


def create_page_extractor_with_model(model: DeepSeekOCRModel) -> PageExtractor:
    if not isinstance(model, DeepSeekOCRModel):
        raise TypeError("model must implement DeepSeekOCRModel protocol")
    return _PageExtractorImpls(DeepSeekModelOCRAdapter(model))


def create_page_extractor_with_adapter(adapter: OCRAdapter) -> PageExtractor:
    if not isinstance(adapter, OCRAdapter):
        raise TypeError("adapter must implement OCRAdapter protocol")
    return _PageExtractorImpls(adapter)


def create_deepseek_vendor_page_extractor(
    config: DeepSeekVendorOCRConfig,
) -> PageExtractor:
    return _PageExtractorImpls(DeepSeekVendorOCRAdapter(config))


def create_baidu_page_extractor(config: BaiduCloudOCRConfig) -> PageExtractor:
    return _PageExtractorImpls(BaiduCloudOCRAdapter(config))


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
        image: Image.Image,
        size: DeepSeekOCRSize,
        stages: int = 1,
        context: ExtractionContext | None = None,
        device_number: int | None = None,
    ) -> Generator[tuple[Image.Image, list[Layout]], None, None]:
        assert stages >= 1, "stages must be at least 1"

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
                try:
                    page_result = self._adapter.extract_page(
                        prompt=_DEFAULT_PROMPT,
                        image_path=image_path,
                        output_path=output_path,
                        size=size,
                        context=context,
                        device_number=device_number,
                    )
                finally:
                    image_path.unlink(missing_ok=True)

                layouts = page_result.layouts
                yield image, layouts

                if i < stages - 1:
                    if fill_color is None:
                        fill_color = background_color(image)
                    image = redact(
                        image=image.copy(),
                        fill_color=fill_color,
                        rectangles=self._redect_rectangles(
                            image=image,
                            dets=(layout.det for layout in layouts),
                        ),
                    )
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    def _redect_rectangles(
        self, image: Image.Image, dets: Iterable[tuple[int, int, int, int]]
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
