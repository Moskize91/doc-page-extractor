import unittest
from unittest.mock import patch

import requests
from PIL import Image

from doc_page_extractor import (
    ExtractionContext, LayoutKind, MiniCPMOCRVendorConfig, VendorOCRRequestError,
    create_minicpm_ocr_vendor_page_extractor,
)
from doc_page_extractor.adapters.minicpm import parse_minicpm_ocr_layouts
from doc_page_extractor.structure import build_structured_page


def block(label, content="", bbox=(0, 0, 100, 20), **extra):
    return dict(label=label, content=content, bbox=list(bbox), status="ok", **extra)


def parse(blocks, **page):
    return parse_minicpm_ocr_layouts({"ocr": dict(blocks=blocks, **page)}, (100, 100))


class MiniCPMLayoutTests(unittest.TestCase):
    def test_hosted_and_self_hosted_shapes_scale_and_keep_reading_order(self):
        for key in ("ocr", "minicpm_ocr"):
            with self.subTest(key=key):
                layouts = parse_minicpm_ocr_layouts({key: {
                    "image_width": 50, "image_height": 50,
                    "blocks": [block("text", "second", order=2),
                               block("doc_title", "first", bbox=(1, 2, 30, 10), order=1)],
                }}, (100, 100))
                self.assertEqual([x.text for x in layouts], ["first", "second"])
                self.assertEqual(layouts[0].det, (2, 4, 60, 20))
                self.assertEqual(layouts[0].kind, LayoutKind.TITLE)

    def test_failed_block_rejects_even_partially_successful_page(self):
        for label, content in (("text", ""), ("text", "partial"), ("image", "")):
            failed = block(label, content)
            failed["status"] = "failed"
            with self.subTest(label=label, content=content), self.assertRaisesRegex(ValueError, "failed"):
                parse([block("text", "valid"), failed])

    def test_skipped_images_and_recognized_image_text_survive(self):
        image = block("image")
        image["status"] = "skipped_no_vlm"
        seal = block("seal")
        seal["status"] = "skipped_seal"
        layouts = parse([image, block("image", "recognized image text"), seal])
        self.assertEqual([x.kind for x in layouts], [LayoutKind.IMAGE] * 3)
        self.assertEqual(layouts[1].text, "recognized image text")

    def test_formula_number_attaches_to_equation_and_real_page_number_is_ignored(self):
        layouts = parse([block("display_formula", "x=1", (0, 0, 70, 20)),
                         block("formula_number", "(1.1)", (75, 0, 95, 20)),
                         block("number", "12", (40, 90, 60, 100))])
        structured = build_structured_page(layouts)
        self.assertEqual(structured.blocks[0].children[0].text, "(1.1)")
        self.assertEqual([x.text for x in structured.ignored], ["12"])

    def test_unattached_caption_and_equation_number_remain_readable(self):
        structured = build_structured_page(parse([
            block("formula_number", "(1.1)"), block("figure_title", "Figure 1"),
        ]))
        self.assertEqual([x.text for x in structured.blocks], ["(1.1)", "Figure 1"])

    def test_caption_between_same_kind_assets_belongs_to_nearest_asset(self):
        structured = build_structured_page(parse([
            block("image", bbox=(0, 0, 100, 20)),
            block("figure_title", "Figure B", (0, 70, 100, 75)),
            block("image", bbox=(0, 76, 100, 99)),
        ]))
        self.assertEqual(structured.blocks[0].children, [])
        self.assertEqual([c.text for c in structured.blocks[1].children], ["Figure B"])

    def test_caption_does_not_select_a_nearer_asset_in_another_column(self):
        structured = build_structured_page(parse([
            block("image", bbox=(0, 0, 40, 30)),
            block("figure_title", "left image", (0, 40, 40, 45)),
            block("table", "<table></table>", (60, 46, 100, 80)),
        ]))
        self.assertEqual(structured.blocks[0].children[0].text, "left image")
        self.assertEqual(structured.blocks[1].children, [])

    def test_table_captions_do_not_require_language_patterns(self):
        structured = build_structured_page(parse([
            block("figure_title", "Tabelle Eins", (0, 0, 100, 10)),
            block("table", "<table><tr><td>value</td></tr></table>", (0, 11, 100, 40)),
            block("vision_footnote", "table note", (0, 41, 100, 50)),
        ]))
        self.assertEqual(len(structured.blocks), 1)
        self.assertEqual(structured.blocks[0].kind, LayoutKind.TABLE)
        self.assertEqual([x.text for x in structured.blocks[0].children], ["Tabelle Eins", "table note"])

    def test_regions_fallback_and_clipping(self):
        item = block("text", "text")
        del item["bbox"]
        item["regions"] = [[-10, 0, 30, 10], [50, 20, 120, 30]]
        self.assertEqual(parse([item])[0].det, (0, 0, 100, 30))

    def test_invalid_regions_do_not_silently_drop_text_from_a_valid_page(self):
        for bbox in (None, [1, 2, 1, 3], [0, 0, float("nan"), 20]):
            with self.subTest(bbox=bbox), self.assertRaisesRegex(ValueError, "region"):
                parse([block("text", "valid"), dict(label="text", content="lost", bbox=bbox)])

    def test_rotation_is_rejected(self):
        for value in (True, 1, "true"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "orientation"):
                parse([block("text", "text")], rotation={"applied": value})

    def test_plain_vision_endpoint_and_partial_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "document layout blocks"):
            parse_minicpm_ocr_layouts({"ocr": {}}, (100, 100))
        with self.assertRaisesRegex(ValueError, "image_height"):
            parse([block("text", "text")], image_width=50)
        self.assertEqual(parse([]), [])


class MiniCPMAdapterTests(unittest.TestCase):
    @patch("requests.post")
    def test_public_factory_uses_one_stage_and_preserves_usage(self, post):
        post.return_value.json.return_value = {
            "minicpm_ocr": {"blocks": [block("text", "recognized")]},
            "usage": {"prompt_tokens": 12, "completion_tokens": 34},
        }
        post.return_value.status_code = 200
        context = ExtractionContext(check_aborted=lambda: False)
        extractor = create_minicpm_ocr_vendor_page_extractor(
            MiniCPMOCRVendorConfig("https://example.test/v1"),
        )
        with Image.new("RGB", (100, 100), "white") as image:
            with self.assertWarnsRegex(RuntimeWarning, "single extraction stage"):
                results = list(extractor.extract_page_results(image, "base", stages=2, context=context))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1].structured.blocks[0].text, "recognized")
        self.assertEqual((context.input_tokens, context.output_tokens), (12, 34))
        self.assertEqual(post.call_count, 1)
        call = post.call_args
        self.assertEqual(call.args[0], "https://example.test/v1/chat/completions")
        self.assertEqual(call.kwargs["json"]["minicpm_ocr"], {"mode": "ocr_only", "upright": False})
        self.assertNotIn("Authorization", call.kwargs["headers"])

    @patch("requests.post")
    def test_http_failure_keeps_original_exception(self, post):
        error = requests.HTTPError("HTTP 402")
        post.return_value.raise_for_status.side_effect = error
        extractor = create_minicpm_ocr_vendor_page_extractor(MiniCPMOCRVendorConfig("https://example.test"))
        with Image.new("RGB", (100, 100)) as image, self.assertRaises(VendorOCRRequestError) as raised:
            list(extractor.extract_page_results(image, "base"))
        self.assertIs(raised.exception.__cause__, error)

    @patch("requests.post")
    def test_failed_response_uses_vendor_error(self, post):
        failed = block("text", "")
        failed["status"] = "failed"
        post.return_value.json.return_value = {"ocr": {"blocks": [block("text", "ok"), failed]}}
        extractor = create_minicpm_ocr_vendor_page_extractor(MiniCPMOCRVendorConfig("https://example.test"))
        with Image.new("RGB", (100, 100)) as image, self.assertRaises(VendorOCRRequestError):
            list(extractor.extract_page_results(image, "base"))
