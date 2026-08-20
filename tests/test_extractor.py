import sys
import types
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from doc_page_extractor import ExtractionContext, Layout, OCRPageResult
from doc_page_extractor.extractor import (
    create_ocr_page_extractor,
    create_page_extractor_with_adapter,
)

class _FakeImage:
    size = (100, 100)

    def save(self, path: Path, image_format: str) -> None:
        del image_format
        path.write_bytes(b"fake")


class _FakeOpenedImage:
    size = (100, 100)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback


class _FakeResizedImage:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.saved_paths: list[Path] = []

    def save(self, path: Path, image_format: str) -> None:
        del image_format
        self.saved_paths.append(path)
        path.write_bytes(b"resized")


class _FakeResizableImage(_FakeResizedImage):
    def __init__(self) -> None:
        super().__init__((9000, 4500))
        self.resize_size: tuple[int, int] | None = None

    def resize(self, size: tuple[int, int]) -> _FakeResizedImage:
        self.resize_size = size
        return _FakeResizedImage(size)


class _SingleStageAdapter:
    allows_multi_stage = False

    def __init__(self) -> None:
        self.calls = 0

    def download(self, revision: str | None) -> None:
        del revision

    def load(self) -> None:
        pass

    def extract_page(
        self,
        prompt: str,
        image_path: Path,
        output_path: Path,
        size: str,
        context: ExtractionContext | None,
        device_number: int | None,
    ) -> OCRPageResult:
        del prompt, image_path, output_path, size, context, device_number
        self.calls += 1
        return OCRPageResult(
            layouts=[Layout(det=(0, 0, 10, 10), text="ok")],
            source="single-stage",
        )


class _MaxSideAdapter:
    allows_multi_stage = True
    max_image_side = 8192

    def __init__(self) -> None:
        self.image_path: Path | None = None

    def download(self, revision: str | None) -> None:
        del revision

    def load(self) -> None:
        pass

    def extract_page(
        self,
        prompt: str,
        image_path: Path,
        output_path: Path,
        size: str,
        context: ExtractionContext | None,
        device_number: int | None,
    ) -> OCRPageResult:
        del prompt, output_path, size, context, device_number
        self.image_path = image_path
        return OCRPageResult(
            layouts=[
                Layout(
                    det=(100, 200, 400, 600),
                    text="ok",
                    polygon=[
                        (100, 200),
                        (400, 200),
                        (400, 600),
                        (100, 600),
                    ],
                )
            ],
            source="max-side",
        )


class TestExtractor(unittest.TestCase):
    def test_single_stage_adapter_ignores_multi_stage_request(self):
        adapter = _SingleStageAdapter()
        extractor = create_page_extractor_with_adapter(adapter)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = list(
                extractor.extract_page_results(
                    image=_FakeImage(),  # type: ignore[arg-type]
                    size="tiny",
                    stages=2,
                    context=ExtractionContext(check_aborted=lambda: False),
                )
            )

        self.assertEqual(adapter.calls, 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1].layouts[0].text, "ok")
        self.assertEqual(len(caught), 1)
        self.assertEqual(caught[0].category, RuntimeWarning)

    def test_adapter_protocol_requires_lifecycle_and_stage_capability(self):
        class _IncompleteAdapter:
            def extract_page(self) -> None:
                pass

        with self.assertRaises(TypeError):
            create_page_extractor_with_adapter(_IncompleteAdapter())  # type: ignore[arg-type]

    def test_adapter_max_image_side_resizes_upload_and_maps_coordinates(self):
        adapter = _MaxSideAdapter()
        image = _FakeResizableImage()
        extractor = create_page_extractor_with_adapter(adapter)

        results = list(
            extractor.extract_page_results(
                image=image,  # type: ignore[arg-type]
                size="tiny",
                stages=1,
                context=ExtractionContext(check_aborted=lambda: False),
            )
        )

        self.assertEqual(image.resize_size, (8192, 4096))
        self.assertIsNotNone(adapter.image_path)
        assert adapter.image_path is not None
        self.assertEqual(adapter.image_path.name, "raw-1-resized.png")
        layout = results[0][1].layouts[0]
        self.assertEqual(layout.det, (110, 220, 439, 659))
        self.assertEqual(
            layout.polygon,
            [(110, 220), (439, 220), (439, 659), (110, 659)],
        )

    def test_ocr_page_extractor_selects_requested_model(self):
        class _DeepSeek1Model:
            def __init__(self, *args, **kwargs) -> None:
                self.args = args
                self.kwargs = kwargs

            def download(self, revision: str | None) -> None:
                del revision

            def load(self) -> None:
                pass

            def unload(self) -> None:
                pass

            def generate(self, *args, **kwargs) -> str:
                del args, kwargs
                return (
                    "<|ref|>text<|/ref|>"
                    "<|det|>[[100, 100, 500, 200]]<|/det|>ocr1 text"
                )

        class _DeepSeek2Model(_DeepSeek1Model):
            def generate(self, *args, **kwargs) -> str:
                del args, kwargs
                return "text[[100, 100, 500, 200]]\nocr2 text"

        fake_model_module = types.ModuleType("doc_page_extractor.model")
        fake_model_module.DeepSeekOCRHuggingFaceModel = _DeepSeek1Model
        fake_model_module.DeepSeekOCR2HuggingFaceModel = _DeepSeek2Model
        fake_pil_module = types.ModuleType("PIL")
        fake_pil_module.Image = SimpleNamespace(open=lambda _path: _FakeOpenedImage())
        with patch.dict(
            sys.modules,
            {
                "doc_page_extractor.model": fake_model_module,
                "PIL": fake_pil_module,
            },
        ):
            extractor1 = create_ocr_page_extractor("deepseek-ocr", model_path="models-cache")
            extractor2 = create_ocr_page_extractor("deepseek-ocr2", model_path="models-cache")

            self.assertIsInstance(extractor1._adapter._model, _DeepSeek1Model)  # type: ignore[attr-defined]
            self.assertIsInstance(extractor2._adapter._model, _DeepSeek2Model)  # type: ignore[attr-defined]

            result1 = list(
                extractor1.extract_page_results(
                    image=_FakeImage(),  # type: ignore[arg-type]
                    size="tiny",
                    stages=1,
                    context=ExtractionContext(check_aborted=lambda: False),
                )
            )[0][1]
            result2 = list(
                extractor2.extract_page_results(
                    image=_FakeImage(),  # type: ignore[arg-type]
                    size="tiny",
                    stages=1,
                    context=ExtractionContext(check_aborted=lambda: False),
                )
            )[0][1]

        self.assertEqual(result1.layouts[0].text, "ocr1 text")
        self.assertEqual(result1.layouts[0].det, (10, 10, 50, 20))
        self.assertEqual(result2.layouts[0].text, "ocr2 text")
        self.assertEqual(result2.layouts[0].det, (10, 10, 50, 20))


if __name__ == "__main__":
    unittest.main()
