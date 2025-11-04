import tempfile

from dataclasses import dataclass
from typing import Generator, cast
from PIL import Image

from .model import DeepSeekOCRModel, DeepSeekOCRSize
from .parser import parse_ocr_response, ParsedItemKind
from .redacter import redact, background_color


@dataclass
class Layout:
    ref: str
    det: tuple[int, int, int, int]
    text: str | None

class PageExtractor:
    def __init__(self) -> None:
        self._model: DeepSeekOCRModel = DeepSeekOCRModel()

    def extract(self, image: Image.Image, size: DeepSeekOCRSize, stages: int = 1) -> Generator[tuple[Image.Image, list[Layout]], None, None]:
        assert stages >= 1, "stages must be at least 1"
        with tempfile.TemporaryDirectory() as temp_path:
            fill_color: tuple[int, int, int] | None = None
            for i in range(stages):
                response = self._model.generate(
                    image=image,
                    prompt="<image>\n<|grounding|>Convert the document to markdown.",
                    temp_path=temp_path,
                    size=size,
                )
                layouts: list[Layout] = []
                for det, ref, text in self._parse_response(image, response):
                    layouts.append(Layout(det, ref, text))
                yield image, layouts
                if i < stages - 1:
                    if fill_color is None:
                        fill_color = background_color(image)
                    image = redact(
                        image=image.copy(), 
                        fill_color=fill_color,
                        rectangles=(layout.det for layout in layouts),
                    )

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