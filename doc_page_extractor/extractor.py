import tempfile

from os import PathLike
from pathlib import Path
from typing import cast, Generator, Iterable
from PIL import Image

from .model import DeepSeekOCRHugginfaceModel
from .parser import ParsedItemKind, parse_ocr_response
from .redacter import background_color, redact
from .lazy_loader import lazy_load, LazyGetter
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
        image_path: PathLike | str,
        size: DeepSeekOCRSize,
        stages: int = 1,
        context: ExtractionContext | None = None,
        device_number: int | None = None,
    ) -> Generator[LazyGetter[tuple[Image.Image, list[Layout]]], None, None]:
        assert stages >= 1, "stages must be at least 1"

        image_path = Path(image_path)
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
                response = self._model.generate(
                    prompt="<image>\n<|grounding|>Convert the document to markdown.",
                    image_path=image_path,
                    output_path=output_path,
                    size=size,
                    context=context,
                    device_number=device_number,
                )
                extraction_pair = lazy_load(
                    load=lambda ip=image_path, res=response: self._generate_extraction_pair(
                        ip, res),
                )
                yield extraction_pair

                if i < stages - 1:
                    image, layouts = extraction_pair()
                    if fill_color is None:
                        fill_color = background_color(image)
                    image = redact(
                        image=image.copy(),
                        fill_color=fill_color,
                        rectangles=(layout.det for layout in layouts),
                    )
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    def _generate_extraction_pair(self, image_path: Path, response: str) -> tuple[Image.Image, list[Layout]]:
        layouts: list[Layout] = []
        image = Image.open(image_path)
        for ref, det, text in self._parse_response(image, response):
            layouts.append(Layout(ref, det, text))
        return image, layouts

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
