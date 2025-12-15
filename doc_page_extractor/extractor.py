import tempfile
import sys

from os import PathLike
from pathlib import Path
from typing import cast, Generator, Iterable
from PIL import Image

from .model import DeepSeekOCRHugginfaceModel
from .parser import ParsedItemKind, parse_ocr_response
from .redacter import background_color, redact
from .types import Layout, PageExtractor, ExtractionContext, DeepSeekOCRModel, DeepSeekOCRSize


def create_page_extractor(
    model_path: PathLike | str | None = None,
    local_only: bool = False,
    enable_devices_numbers: Iterable[int] | None = None,
) -> PageExtractor:
    model: DeepSeekOCRHugginfaceModel = DeepSeekOCRHugginfaceModel(
        model_path=Path(model_path) if model_path else None,
        local_only=local_only,
        enable_devices_numbers=enable_devices_numbers,
    )
    return _PageExtractorImpls(model)


def create_page_extractor_with_model(model: DeepSeekOCRModel) -> PageExtractor:
    if not isinstance(model, DeepSeekOCRModel):
        raise TypeError("model must implement DeepSeekOCRModel protocol")
    return _PageExtractorImpls(model)


class _PageExtractorImpls:
    def __init__(self, model: DeepSeekOCRModel) -> None:
        self._model: DeepSeekOCRModel = model

    def download_models(self, revision: str | None = None) -> None:
        self._model.download(revision)

    def load_models(self) -> None:
        self._model.load()

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
                    response = self._model.generate(
                        prompt="<image>\n<|grounding|>Convert the document to markdown.",
                        image_path=image_path,
                        output_path=output_path,
                        size=size,
                        context=context,
                        device_number=device_number,
                    )
                finally:
                    image_path.unlink(missing_ok=True)

                layouts = [
                    Layout(ref, det, text)
                    for ref, det, text in self._parse_response(image, response)
                ]
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

    def _parse_response(self, image: Image.Image, response: str) -> Generator[tuple[str, tuple[int, int, int, int], str | None], None, None]:
        width, height = image.size
        det: tuple[int, int, int, int] | None = None
        ref: str | None = None

        for kind, content in parse_ocr_response(response, width, height):
            if kind == ParsedItemKind.TEXT:
                if det is not None and ref is not None:
                    yield ref, det, cast(str, content)
                    det = None
                    ref = None
            if det is not None and ref is not None:
                yield ref, det, None
                det = None
                ref = None
            elif kind == ParsedItemKind.DET:
                det = cast(tuple[int, int, int, int], content)
            elif kind == ParsedItemKind.REF:
                ref = cast(str, content)
        if det is not None and ref is not None:
            yield ref, det, None

    def _redect_rectangles(self, image: Image.Image, dets: Iterable[tuple[int, int, int, int]]):
        # 将页面上 2/3 全部涂抹，并沿着 2/3 线向下涂抹到每一个识别为文字区块的底部
        # 这种方法旨在涂抹掉尽可能多的不是页脚的区域，以排除诸如页眉之类干扰识别页脚的内容
        rate = float(2/3)
        width, height = image.size
        y_cutted = round(height * rate)
        yield (0, 0, width, y_cutted)
        yield from self._redact_button_rectangles(y_cutted, dets)

    def _redact_button_rectangles(self, y_cutted: int, dets: Iterable[tuple[int, int, int, int]]):
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
