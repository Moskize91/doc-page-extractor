import tempfile
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Generator, cast

from PIL import Image

from .abort import AbortContext
from .check_env import check_env
from .model import DeepSeekOCRModel, DeepSeekOCRSize
from .parser import ParsedItemKind, parse_ocr_response
from .redacter import background_color, redact


@dataclass
class Layout:
    ref: str
    det: tuple[int, int, int, int]
    text: str | None


class PageExtractor:
    def __init__(
        self,
        model_path: PathLike | None = None,
        local_only: bool = False,
    ) -> None:
        self._model: DeepSeekOCRModel = DeepSeekOCRModel(
            model_path=Path(model_path) if model_path else None,
            local_only=local_only,
        )

    def download_models(self) -> None:
        self._model.download()

    def load_models(self) -> None:
        self._model.load()

    def extract(
        self,
        image: Image.Image,
        size: DeepSeekOCRSize,
        stages: int = 1,
        aborted_context: AbortContext | None = None,
    ) -> Generator[tuple[Image.Image, list[Layout]], None, None]:
        check_env()
        assert stages >= 1, "stages must be at least 1"
        with tempfile.TemporaryDirectory() as temp_path:
            fill_color: tuple[int, int, int] | None = None
            for i in range(stages):
                response = self._model.generate(
                    image=image,
                    prompt="<image>\n<|grounding|>Convert the document to markdown.",
                    temp_path=temp_path,
                    size=size,
                    aborted_context=aborted_context,
                )
                layouts: list[Layout] = []
                for ref, det, text in self._parse_response(image, response):
                    layouts.append(Layout(ref, det, text))
                yield image, layouts
                if i < stages - 1:
                    if fill_color is None:
                        fill_color = background_color(image)
                    image = redact(
                        image=image.copy(),
                        fill_color=fill_color,
                        rectangles=(layout.det for layout in layouts),
                    )

    def _parse_response(
        self, image: Image.Image, response: str
    ) -> Generator[tuple[str, tuple[int, int, int, int], str | None], None, None]:
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
