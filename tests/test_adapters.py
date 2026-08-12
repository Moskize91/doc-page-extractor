import unittest

from doc_page_extractor.adapters.baidu import parse_baidu_layouts
from doc_page_extractor.adapters.deepseek import parse_deepseek_layouts


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
        self.assertEqual(layout.source, "deepseek-vendor")

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
        self.assertEqual(layouts[0].ref, "paragraph_title")
        self.assertEqual(layouts[0].det, (161, 167, 518, 220))
        self.assertEqual(layouts[0].text, "第二章 鸿商巨贾")
        self.assertEqual(layouts[0].type, "paragraph_title")
        self.assertEqual(layouts[0].source, "baidu")
        self.assertEqual(layouts[1].det, (158, 510, 1360, 1068))


if __name__ == "__main__":
    unittest.main()
