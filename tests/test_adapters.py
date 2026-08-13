import unittest

from doc_page_extractor.adapters.baidu import parse_baidu_layouts
from doc_page_extractor.adapters.deepseek import parse_deepseek_layouts
from doc_page_extractor.structure import build_structured_page
from doc_page_extractor.types import LayoutKind


class _StubImage:
    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)


class TestAdapters(unittest.TestCase):
    def test_deepseek_layouts_from_token_response(self):
        image = _StubImage(1000, 1000)
        response = "<|ref|>标题<|/ref|><|det|>[[100, 200, 300, 400]]<|/det|>正文"

        layouts = parse_deepseek_layouts(image, response, source="deepseek-vendor")

        self.assertEqual(len(layouts), 1)
        layout = layouts[0]
        self.assertEqual(layout.ref, "标题")
        self.assertEqual(layout.det, (100, 200, 300, 400))
        self.assertEqual(layout.text, "正文")
        self.assertEqual(layout.kind, LayoutKind.UNKNOWN)
        self.assertEqual(layout.source, "deepseek-vendor")

    def test_deepseek_known_refs_are_typed(self):
        image = _StubImage(1000, 1000)
        response = (
            "<|ref|>image<|/ref|><|det|>[[100, 100, 800, 500]]<|/det|>"
            "image-body"
            "<|ref|>image_caption<|/ref|><|det|>[[100, 520, 800, 600]]<|/det|>"
            "图一"
        )

        layouts = parse_deepseek_layouts(image, response)
        structured = build_structured_page(layouts)

        self.assertEqual(layouts[0].kind, LayoutKind.IMAGE)
        self.assertEqual(layouts[1].kind, LayoutKind.IMAGE_CAPTION)
        self.assertEqual(len(structured.blocks), 1)
        self.assertEqual(structured.blocks[0].kind, LayoutKind.IMAGE)
        self.assertEqual(structured.blocks[0].children[0].kind, LayoutKind.IMAGE_CAPTION)

    def test_baidu_layouts_from_json(self):
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

        layouts = parse_baidu_layouts(parse_result)

        self.assertEqual(len(layouts), 2)
        self.assertEqual(layouts[0].ref, "sub_title")
        self.assertEqual(layouts[0].det, (161, 167, 518, 220))
        self.assertEqual(layouts[0].text, "第二章 鸿商巨贾")
        self.assertEqual(layouts[0].kind, LayoutKind.TITLE)
        self.assertEqual(layouts[0].type, "paragraph_title")
        self.assertEqual(layouts[0].source, "baidu")
        self.assertEqual(layouts[1].ref, "text")
        self.assertEqual(layouts[1].kind, LayoutKind.TEXT)
        self.assertEqual(layouts[1].det, (158, 510, 1360, 1068))

    def test_baidu_richer_types_collapse_to_stable_kinds(self):
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
                            "text": "第六表",
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
                    ]
                }
            ]
        }

        layouts = parse_baidu_layouts(parse_result)
        structured = build_structured_page(layouts)

        self.assertEqual(layouts[0].kind, LayoutKind.FOOTNOTE)
        self.assertEqual(layouts[0].ref, "text")
        self.assertEqual(layouts[1].kind, LayoutKind.TABLE_CAPTION)
        self.assertEqual(layouts[1].ref, "table_caption")
        self.assertEqual(layouts[2].kind, LayoutKind.TABLE)
        self.assertEqual(layouts[3].kind, LayoutKind.PAGE_NUMBER)
        self.assertEqual(len(structured.ignored), 1)
        self.assertEqual(structured.ignored[0].kind, LayoutKind.PAGE_NUMBER)
        self.assertEqual(structured.blocks[1].kind, LayoutKind.TABLE)
        self.assertEqual(structured.blocks[1].children[0].kind, LayoutKind.TABLE_CAPTION)


if __name__ == "__main__":
    unittest.main()
