import json
import tempfile
import unittest

# Public exports use module-level lazy loading, which pylint cannot infer.
# pylint: disable=no-name-in-module
from pathlib import Path
from unittest.mock import patch

import requests  # type: ignore[import-not-found]
from PIL import Image  # type: ignore[import-not-found]

from doc_page_extractor import (
    AbortError,
    ExtractionContext,
    GLMOCRServiceConfig,
    LayoutKind,
    TokenLimitError,
    create_glm_ocr_service_page_extractor,
)
from doc_page_extractor.adapters.glmocr import (  # type: ignore[import-not-found]
    GLMOCRServiceAdapter,
    parse_glm_ocr_layouts,
)


class _Response:
    def __init__(self, status_code: int, value, text: str = "") -> None:
        self.status_code = status_code
        self._value = value
        self.text = text

    def json(self):
        if isinstance(self._value, BaseException):
            raise self._value
        return self._value


def _response_data():
    return {
        "json_result": [
            [
                {
                    "index": 5,
                    "label": "formula",
                    "native_label": "display_formula",
                    "content": "x^2 + y^2",
                    "bbox_2d": [100, 700, 900, 800],
                },
                {
                    "index": 2,
                    "label": "image",
                    "native_label": "image",
                    "content": None,
                    "bbox_2d": [100, 250, 500, 500],
                    "polygon": [[100, 250], [500, 250], [500, 500], [100, 500]],
                },
                {
                    "index": 6,
                    "label": "text",
                    "native_label": "footnote",
                    "content": "① source note",
                    "bbox_2d": [100, 900, 900, 950],
                },
                {
                    "index": 0,
                    "label": "text",
                    "native_label": "doc_title",
                    "content": "A title",
                    "bbox_2d": [0, 0, 1000, 50],
                },
                {
                    "index": 4,
                    "label": "table",
                    "native_label": "table",
                    "content": "<table><tr><td>A</td></tr></table>",
                    "bbox_2d": [100, 525, 900, 650],
                },
                {
                    "index": 1,
                    "label": "text",
                    "native_label": "text",
                    "content": "Body",
                    "bbox_2d": [100, 100, 900, 200],
                },
                {
                    "index": 3,
                    "label": "text",
                    "native_label": "figure_title",
                    "content": "Figure 1",
                    "bbox_2d": [100, 505, 500, 520],
                },
            ]
        ],
        "markdown_result": "A title\n\nBody",
        "usage": {},
    }


class TestGLMOCRServiceAdapter(unittest.TestCase):
    def _image_path(self, directory: str) -> Path:
        path = Path(directory) / "page.png"
        Image.new("RGB", (1000, 2000), "white").save(path)
        return path

    def test_factory_and_config_are_official_lazy_surface(self):
        config = GLMOCRServiceConfig(
            endpoint_url="http://example.test/glmocr/parse",
            api_key="secret",
            timeout_seconds=12,
        )
        extractor = create_glm_ocr_service_page_extractor(config)

        self.assertEqual(config.timeout_seconds, 12)
        self.assertNotIn("secret", repr(config))
        self.assertIsNotNone(extractor)
        self.assertEqual(
            GLMOCRServiceConfig().endpoint_url,
            "http://127.0.0.1:5002/glmocr/parse",
        )
        with self.assertRaises(ValueError):
            GLMOCRServiceConfig(endpoint_url=" ")
        with self.assertRaises(ValueError):
            GLMOCRServiceConfig(timeout_seconds=0)

    @patch("requests.post")
    def test_maps_sdk_order_labels_and_normalized_geometry(self, post):
        post.return_value = _Response(200, _response_data())
        adapter = GLMOCRServiceAdapter(
            GLMOCRServiceConfig(
                endpoint_url="http://127.0.0.1:5002/glmocr/parse",
                api_key="secret",
                timeout_seconds=9,
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            image_path = self._image_path(directory)
            result = adapter.extract_page(
                prompt="ignored",
                image_path=image_path,
                output_path=Path(directory),
                size="tiny",
                context=ExtractionContext(check_aborted=lambda: False),
                device_number=None,
            )

        self.assertEqual(result.source, "glm-ocr-sdk")
        self.assertEqual([layout.type for layout in result.layouts], [
            "doc_title",
            "text",
            "image",
            "figure_title",
            "table",
            "display_formula",
            "footnote",
        ])
        self.assertEqual(result.layouts[0].kind, LayoutKind.TITLE)
        self.assertEqual(result.layouts[0].det, (0, 0, 1000, 100))
        self.assertEqual(result.layouts[1].det, (100, 200, 900, 400))
        self.assertEqual(result.layouts[2].kind, LayoutKind.IMAGE)
        self.assertEqual(result.layouts[2].polygon, [
            (100, 500),
            (500, 500),
            (500, 1000),
            (100, 1000),
        ])
        self.assertEqual(result.layouts[3].kind, LayoutKind.IMAGE_CAPTION)
        self.assertEqual(result.layouts[4].kind, LayoutKind.TABLE)
        self.assertEqual(result.layouts[4].html, "<table><tr><td>A</td></tr></table>")
        self.assertEqual(result.layouts[5].kind, LayoutKind.EQUATION)
        self.assertEqual(result.layouts[6].kind, LayoutKind.FOOTNOTE)
        self.assertIsNotNone(result.structured)
        assert result.structured is not None
        image_block = next(
            block for block in result.structured.blocks if block.kind == LayoutKind.IMAGE
        )
        self.assertEqual(image_block.children[0].kind, LayoutKind.IMAGE_CAPTION)
        self.assertTrue(post.call_args.kwargs["json"]["images"][0].startswith("data:image/png;base64,"))
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(post.call_args.kwargs["timeout"], 9)

    @patch("requests.post")
    def test_markdown_alias_is_also_preserved_in_raw_text(self, post):
        response = _response_data()
        response["markdown_result"] = ""
        response["md_results"] = "fallback Markdown"
        post.return_value = _Response(200, response)
        with tempfile.TemporaryDirectory() as directory:
            result = GLMOCRServiceAdapter(GLMOCRServiceConfig()).extract_page(
                "", self._image_path(directory), Path(directory), "tiny", None, None
            )
        self.assertEqual(result.raw_text, "fallback Markdown")

    def test_captured_sdk_response_preserves_geometry_and_semantics(self):
        fixture = Path(__file__).parent / "fixtures/glmocr/sdk-rich-response.json"
        response = json.loads(fixture.read_text(encoding="utf-8"))
        layouts = parse_glm_ocr_layouts(response, image_size=(1800, 2600))
        self.assertEqual(len(layouts), 23)
        self.assertEqual(layouts[0].det, (113, 39, 769, 73))
        title = next(layout for layout in layouts if layout.kind == LayoutKind.TITLE)
        self.assertEqual(title.text, "GLM-OCR Structured Region Test")
        assert title.raw is not None
        self.assertEqual(title.raw["content"], "# GLM-OCR Structured Region Test")
        kinds = {layout.kind for layout in layouts}
        for kind in (LayoutKind.TITLE, LayoutKind.TEXT, LayoutKind.FOOTNOTE,
                     LayoutKind.TABLE, LayoutKind.IMAGE, LayoutKind.IMAGE_CAPTION,
                     LayoutKind.HEADER, LayoutKind.FOOTER, LayoutKind.PAGE_NUMBER):
            self.assertIn(kind, kinds)
        for layout in layouts:
            left, top, right, bottom = layout.det
            self.assertTrue(0 <= left < right <= 1800)
            self.assertTrue(0 <= top < bottom <= 2600)
        table = next(layout for layout in layouts if layout.kind == LayoutKind.TABLE)
        self.assertTrue(table.html and table.html.startswith("<table"))

    def test_rejects_missing_or_unsafe_geometry(self):
        for bbox in (None, [0, 0, 0, 100], [0, 0, 100, 1001]):
            with self.subTest(bbox=bbox):
                with self.assertRaises(ValueError):
                    parse_glm_ocr_layouts(
                        {"json_result": [[{
                            "index": 0,
                            "label": "text",
                            "content": "x",
                            "bbox_2d": bbox,
                        }]]},
                        image_size=(100, 100),
                    )

    def test_empty_page_is_not_fabricated_from_markdown(self):
        result = parse_glm_ocr_layouts(
            {"json_result": [[]], "markdown_result": ""},
            image_size=(100, 100),
        )
        self.assertEqual(result, [])
        with self.assertRaises(ValueError):
            parse_glm_ocr_layouts(
                {"json_result": None, "markdown_result": "unlocated text"},
                image_size=(100, 100),
            )

    def test_malformed_response_is_not_a_blank_page(self):
        for response in ({}, {"json_result": None}, {"usage": {}}):
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_glm_ocr_layouts(response, image_size=(100, 100))

    def test_unknown_labels_fail_instead_of_silently_losing_content(self):
        with self.assertRaisesRegex(ValueError, "unsupported layout label"):
            parse_glm_ocr_layouts(
                {"json_result": [[{
                    "index": 0, "label": "new_sdk_label", "content": "Important",
                    "bbox_2d": [0, 0, 100, 100],
                }]]},
                image_size=(100, 100),
            )

    @patch("requests.post")
    def test_http_errors_and_timeouts_are_not_successful_empty_pages(self, post):
        adapter = GLMOCRServiceAdapter(GLMOCRServiceConfig())
        with tempfile.TemporaryDirectory() as directory:
            image_path = self._image_path(directory)
            post.return_value = _Response(502, {"error": "down"})
            with self.assertRaises(RuntimeError):
                adapter.extract_page(
                    "", image_path, Path(directory), "tiny", None, None
                )

            post.side_effect = requests.exceptions.Timeout("slow")
            with self.assertRaises(TimeoutError):
                adapter.extract_page(
                    "", image_path, Path(directory), "tiny", None, None
                )

    @patch("requests.post")
    def test_abort_and_token_limit_semantics_are_explicit(self, post):
        adapter = GLMOCRServiceAdapter(GLMOCRServiceConfig())
        with tempfile.TemporaryDirectory() as directory:
            image_path = self._image_path(directory)
            with self.assertRaises(AbortError):
                adapter.extract_page(
                    "",
                    image_path,
                    Path(directory),
                    "tiny",
                    ExtractionContext(check_aborted=lambda: True),
                    None,
                )
            post.assert_not_called()

            with self.assertRaises(NotImplementedError):
                adapter.extract_page(
                    "",
                    image_path,
                    Path(directory),
                    "tiny",
                    ExtractionContext(check_aborted=lambda: False, max_tokens=100),
                    None,
                )
            post.assert_not_called()

            with self.assertRaises(TokenLimitError):
                adapter.extract_page(
                    "",
                    image_path,
                    Path(directory),
                    "tiny",
                    ExtractionContext(check_aborted=lambda: False, max_tokens=0),
                    None,
                )
            post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
