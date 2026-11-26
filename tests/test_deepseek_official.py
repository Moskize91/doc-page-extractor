import unittest

from doc_page_extractor.adapters import parse_deepseek_official_ocr_layouts
from doc_page_extractor.adapters.deepseek import parse_deepseek_ocr_layouts


class ImageSize:
    size = (2000, 3000)


class OfficialLayoutTests(unittest.TestCase):
    def test_official_markers_and_label_case_use_existing_geometry_and_kinds(self):
        for label in ("Title", "Text", "Image", "Table", "Equation", "Footnote", "Future_Label"):
            with self.subTest(label=label):
                content = "<table><tr><td>value</td></tr></table>" if label == "Table" else "Keep literal ｜ and box."
                official = f"<｜ref｜>{label}<｜/ref｜><｜box｜>[[100,200,900,400]]<｜/box｜>{content}"
                legacy = f"<|ref|>{label.lower()}<|/ref|><|det|>[[100,200,900,400]]<|/det|>{content}"
                layouts = parse_deepseek_official_ocr_layouts(ImageSize(), official)
                expected = parse_deepseek_ocr_layouts(ImageSize(), legacy)
                self.assertEqual(len(layouts), 1)
                self.assertEqual(layouts[0].kind, expected[0].kind)
                self.assertEqual(layouts[0].text, expected[0].text)
                self.assertEqual(layouts[0].html, expected[0].html)
                self.assertEqual(layouts[0].det, (200, 600, 1800, 1200))
                self.assertEqual(layouts[0].source, "deepseek-ocr-official")

    def test_ascii_markers_and_empty_output_keep_existing_behavior(self):
        self.assertEqual(parse_deepseek_official_ocr_layouts(ImageSize(), ""), [])
        text = "<|ref|>text<|/ref|><|box|>[[0,0,1000,1000]]<|/box|>body"
        self.assertEqual(parse_deepseek_official_ocr_layouts(ImageSize(), text)[0].text, "body")

    def test_legacy_parser_is_unchanged(self):
        text = "<|ref|>Title<|/ref|><|det|>[[0,0,1000,1000]]<|/det|>body"
        self.assertEqual(parse_deepseek_ocr_layouts(ImageSize(), text)[0].type, "Title")
