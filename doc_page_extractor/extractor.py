from dataclasses import dataclass
from typing import Generator, cast
from PIL import Image

from .model import DeepSeekOCRModel, DeepSeekOCRSize
from .parser import parse_ocr_response, ParsedItemKind

@dataclass
class Layout:
    ref: str
    det: tuple[int, int, int, int]
    text: str | None

class PageExtractor:
    def __init__(self) -> None:
        self._model: DeepSeekOCRModel = DeepSeekOCRModel()

    def extract(self, image: Image.Image, size: DeepSeekOCRSize, stages: int = 1):
        prompt = "<image>\n<|grounding|>Convert the document to markdown."
        for _ in range(stages):
            response = self._model.generate(
                image=image,
                prompt=prompt,
                size=size,
            )
            layouts: list[Layout] = []
            for det, ref, text in self._parse_response(image, response):
                layouts.append(Layout(det, ref, text))
            yield layouts

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