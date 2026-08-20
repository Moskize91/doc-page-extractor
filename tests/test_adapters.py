import unittest

from doc_page_extractor.adapters.unlimited import (
    parse_unlimited_ocr_layouts,
    parse_unlimited_ocr_local_layouts,
)
from doc_page_extractor.adapters.deepseek import (
    DeepSeekOCR2VendorConfig,
    _vendor_chat_completions_url,
    parse_deepseek_ocr2_layouts,
    parse_deepseek_ocr_layouts,
)
from doc_page_extractor.structure import build_structured_page
from doc_page_extractor.types import LayoutKind


class _StubImage:
    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)


class TestAdapters(unittest.TestCase):
    def test_deepseek_ocr2_vendor_config_accepts_openai_style_settings(self):
        config = DeepSeekOCR2VendorConfig(
            base_url="https://example.test/openai",
            api_key="test-key",
            model="deepseek-ocr2",
        )

        self.assertEqual(config.base_url, "https://example.test/openai")
        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.model, "deepseek-ocr2")

    def test_vendor_chat_completions_url_accepts_both_base_forms(self):
        self.assertEqual(
            _vendor_chat_completions_url("https://example.test/openai"),
            "https://example.test/openai/v1/chat/completions",
        )
        self.assertEqual(
            _vendor_chat_completions_url("https://example.test/openai/v1"),
            "https://example.test/openai/v1/chat/completions",
        )

    def test_deepseek_layouts_from_token_response(self):
        image = _StubImage(1000, 1000)
        response = "<|ref|>标题<|/ref|><|det|>[[100, 200, 300, 400]]<|/det|>正文"

        layouts = parse_deepseek_ocr_layouts(image, response, source="deepseek-ocr-vendor")

        self.assertEqual(len(layouts), 1)
        layout = layouts[0]
        self.assertEqual(layout.det, (100, 200, 300, 400))
        self.assertEqual(layout.text, "正文")
        self.assertEqual(layout.kind, LayoutKind.TEXT)
        self.assertEqual(layout.type, "标题")
        self.assertEqual(layout.source, "deepseek-ocr-vendor")

    def test_deepseek_layouts_from_ocr2_line_blocks(self):
        image = _StubImage(2000, 3000)
        response = (
            "text[[101, 231, 877, 486]]\n"
            "1774 年爆发的北美革命。"
            "\n\n"
            "text[[102, 494, 877, 785]]\n"
            "不过，此举产生的效果适得其反。"
        )

        layouts = parse_deepseek_ocr2_layouts(
            image, response, source="deepseek-ocr2-vendor"
        )

        self.assertEqual(len(layouts), 2)
        self.assertEqual(layouts[0].det, (202, 693, 1754, 1458))
        self.assertEqual(layouts[0].text, "1774 年爆发的北美革命。")
        self.assertEqual(layouts[0].kind, LayoutKind.TEXT)
        self.assertEqual(layouts[0].type, "text")
        self.assertEqual(layouts[1].text, "不过，此举产生的效果适得其反。")

    def test_deepseek_ocr2_layouts_fall_back_to_token_response(self):
        image = _StubImage(1000, 1000)
        response = (
            "<|ref|>text<|/ref|><|det|>[[100, 200, 300, 400]]<|/det|>"
            "ocr2 text"
        )

        layouts = parse_deepseek_ocr2_layouts(
            image, response, source="deepseek-ocr2"
        )

        self.assertEqual(len(layouts), 1)
        self.assertEqual(layouts[0].det, (100, 200, 300, 400))
        self.assertEqual(layouts[0].text, "ocr2 text")
        self.assertEqual(layouts[0].kind, LayoutKind.TEXT)
        self.assertEqual(layouts[0].source, "deepseek-ocr2")

    def test_deepseek_known_refs_are_typed(self):
        image = _StubImage(1000, 1000)
        response = (
            "<|ref|>image<|/ref|><|det|>[[100, 100, 800, 500]]<|/det|>"
            "image-body"
            "<|ref|>image_caption<|/ref|><|det|>[[100, 520, 800, 600]]<|/det|>"
            "图一"
        )

        layouts = parse_deepseek_ocr_layouts(image, response)
        structured = build_structured_page(layouts)

        self.assertEqual(layouts[0].kind, LayoutKind.IMAGE)
        self.assertEqual(layouts[1].kind, LayoutKind.IMAGE_CAPTION)
        self.assertEqual(len(structured.blocks), 1)
        self.assertEqual(structured.blocks[0].kind, LayoutKind.IMAGE)
        self.assertEqual(structured.blocks[0].children[0].kind, LayoutKind.IMAGE_CAPTION)

    def test_deepseek_ocr2_figure_titles_are_caption_kinds(self):
        image = _StubImage(1000, 1000)
        response = (
            "figure_title[[100, 100, 500, 120]]\n"
            "第六表\n\n"
            "table[[100, 130, 500, 300]]\n"
            "<table><tr><td>A</td></tr></table>\n\n"
            "image[[100, 400, 500, 700]]\n\n"
            "figure_title[[100, 710, 500, 740]]\n"
            "Figure 1.6: muscle length"
        )

        layouts = parse_deepseek_ocr2_layouts(image, response)
        structured = build_structured_page(layouts)

        self.assertEqual(layouts[0].kind, LayoutKind.TABLE_CAPTION)
        self.assertEqual(layouts[1].kind, LayoutKind.TABLE)
        self.assertEqual(layouts[1].html, "<table><tr><td>A</td></tr></table>")
        self.assertEqual(layouts[3].kind, LayoutKind.IMAGE_CAPTION)
        self.assertEqual(structured.blocks[0].kind, LayoutKind.TABLE)
        self.assertEqual(structured.blocks[0].children[0].kind, LayoutKind.TABLE_CAPTION)
        self.assertEqual(structured.blocks[1].kind, LayoutKind.IMAGE)
        self.assertEqual(structured.blocks[1].children[0].kind, LayoutKind.IMAGE_CAPTION)

    def test_deepseek_zero_area_layouts_are_ignored(self):
        image = _StubImage(1000, 1000)
        response = (
            "<|ref|>text<|/ref|><|det|>[[0, 0, 0, 0]]<|/det|>0"
            "<|ref|>text<|/ref|><|det|>[[100, 100, 200, 200]]<|/det|>ok"
        )

        layouts = parse_deepseek_ocr_layouts(image, response)

        self.assertEqual(len(layouts), 1)
        self.assertEqual(layouts[0].text, "ok")

    def test_unlimited_ocr_layouts_from_json(self):
        parse_result = {
            "file_name": "friendly-title.png",
            "pages": [
                {
                    "page_num": 0,
                    "layouts": [
                        {
                            "text": "第二章 鸿商巨贾",
                            "position": [161, 167, 357, 53],
                            "polygon": [[161, 167], [518, 167], [518, 220], [161, 220]],
                            "type": "paragraph_title",
                            "table_html": "",
                        },
                        {
                            "text": "1774 年爆发的北美革命",
                            "position": [158, 510, 1202, 558],
                            "polygon": [[158, 510], [1360, 510], [1360, 1068], [158, 1068]],
                            "type": "text",
                        },
                    ],
                }
            ],
        }

        layouts = parse_unlimited_ocr_layouts(parse_result)

        self.assertEqual(len(layouts), 2)
        self.assertEqual(layouts[0].det, (161, 167, 518, 220))
        self.assertEqual(layouts[0].text, "第二章 鸿商巨贾")
        self.assertEqual(layouts[0].kind, LayoutKind.TITLE)
        self.assertEqual(layouts[0].type, "paragraph_title")
        self.assertEqual(layouts[0].source, "unlimited-ocr-vendor")
        self.assertEqual(layouts[1].kind, LayoutKind.TEXT)
        self.assertEqual(layouts[1].det, (158, 510, 1360, 1068))

    def test_unlimited_ocr_richer_types_collapse_to_stable_kinds(self):
        parse_result = {
            "pages": [
                {
                    "layouts": [
                        {
                            "text": "脚注内容",
                            "position": [10, 900, 300, 40],
                            "type": "footnote",
                        },
                        {
                            "text": "正文",
                            "position": [100, 300, 400, 100],
                            "type": "text",
                        },
                        {
                            "text": "第 六 表",
                            "position": [100, 100, 100, 30],
                            "type": "text",
                        },
                        {
                            "text": "<table><tr><td>A</td></tr></table>",
                            "position": [100, 140, 400, 200],
                            "type": "table",
                            "table_html": "<table><tr><td>A</td></tr></table>",
                        },
                        {
                            "text": "1",
                            "position": [500, 980, 10, 10],
                            "type": "number",
                        },
                        {
                            "text": "① element",
                            "position": [100, 1100, 120, 30],
                            "type": "number",
                        },
                    ]
                }
            ]
        }

        layouts = parse_unlimited_ocr_layouts(parse_result)
        structured = build_structured_page(layouts)

        self.assertEqual(layouts[0].kind, LayoutKind.FOOTNOTE)
        self.assertEqual(layouts[1].kind, LayoutKind.TEXT)
        self.assertEqual(layouts[2].kind, LayoutKind.TABLE_CAPTION)
        self.assertEqual(layouts[3].kind, LayoutKind.TABLE)
        self.assertEqual(layouts[4].kind, LayoutKind.PAGE_NUMBER)
        self.assertEqual(layouts[5].kind, LayoutKind.FOOTNOTE)
        self.assertEqual(len(structured.ignored), 1)
        self.assertEqual(structured.ignored[0].kind, LayoutKind.PAGE_NUMBER)
        table_block = next(
            block for block in structured.blocks if block.kind == LayoutKind.TABLE
        )
        self.assertEqual(table_block.children[0].kind, LayoutKind.TABLE_CAPTION)

    def test_unlimited_ocr_local_layouts_from_det_blocks(self):
        image = _StubImage(999, 1998)
        response = (
            "<|det|>title [100, 100, 500, 200]<|/det|>Chapter 1\n"
            "<|det|>text [100, 220, 800, 400]<|/det|>Line one\n"
            "Line two\n"
            "<|det|>table [100, 500, 800, 900]<|/det|>"
            "<table><tr><td>A</td></tr></table>"
            "<|det|>page_footnote [100, 920, 800, 950]<|/det|>① footnote\n"
            "<|det|>page_number [880, 960, 920, 990]<|/det|>12"
        )

        layouts = parse_unlimited_ocr_local_layouts(image, response)
        structured = build_structured_page(layouts)

        self.assertEqual(len(layouts), 5)
        self.assertEqual(layouts[0].kind, LayoutKind.TITLE)
        self.assertEqual(layouts[0].det, (100, 200, 500, 400))
        self.assertEqual(layouts[0].source, "unlimited-ocr")
        self.assertEqual(layouts[1].kind, LayoutKind.TEXT)
        self.assertEqual(layouts[1].text, "Line one\nLine two")
        self.assertEqual(layouts[2].kind, LayoutKind.TABLE)
        self.assertEqual(layouts[2].html, "<table><tr><td>A</td></tr></table>")
        self.assertEqual(layouts[3].kind, LayoutKind.FOOTNOTE)
        self.assertEqual(layouts[4].kind, LayoutKind.PAGE_NUMBER)
        self.assertEqual(structured.blocks[2].kind, LayoutKind.TABLE)
        self.assertEqual(len(structured.ignored), 1)
        self.assertEqual(structured.ignored[0].kind, LayoutKind.PAGE_NUMBER)


if __name__ == "__main__":
    unittest.main()
