import unittest
import warnings
from pathlib import Path

from doc_page_extractor import ExtractionContext, Layout, OCRPageResult
from doc_page_extractor.extractor import create_page_extractor_with_adapter


class _FakeImage:
    size = (100, 100)

    def save(self, path: Path, image_format: str) -> None:
        del image_format
        path.write_bytes(b"fake")


class _SingleStageAdapter:
    supports_multi_stage = False

    def __init__(self) -> None:
        self.calls = 0

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
            layouts=[Layout(ref="text", det=(0, 0, 10, 10), text="ok")],
            source="single-stage",
        )


class TestExtractor(unittest.TestCase):
    def test_single_stage_adapter_ignores_multi_stage_request(self):
        adapter = _SingleStageAdapter()
        extractor = create_page_extractor_with_adapter(adapter)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = list(
                extractor.extract(
                    image=_FakeImage(),  # type: ignore[arg-type]
                    size="tiny",
                    stages=2,
                    context=ExtractionContext(check_aborted=lambda: False),
                )
            )

        self.assertEqual(adapter.calls, 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1][0].text, "ok")
        self.assertEqual(len(caught), 1)
        self.assertEqual(caught[0].category, RuntimeWarning)


if __name__ == "__main__":
    unittest.main()
