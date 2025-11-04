import unittest
from doc_page_extractor.parser import parse_ocr_response, ParsedItemKind


class TestParseOCRResponse(unittest.TestCase):
    
    def test_det_only(self):
        """测试只有 det 标签的情况"""
        response = "<|det|>[[100, 200, 300, 400]]<|/det|>"
        width, height = 1000, 1000
        
        results = list(parse_ocr_response(response, width, height))
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], ParsedItemKind.DET)
        self.assertEqual(results[0][1], (100, 200, 300, 400))
    
    def test_ref_only(self):
        """测试只有 ref 标签的情况"""
        response = "<|ref|>这是引用文本<|/ref|>"
        width, height = 1000, 1000
        
        results = list(parse_ocr_response(response, width, height))
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], ParsedItemKind.REF)
        self.assertEqual(results[0][1], "这是引用文本")
    
    def test_text_only(self):
        """测试只有普通文本的情况"""
        response = "这是普通文本"
        width, height = 1000, 1000
        
        results = list(parse_ocr_response(response, width, height))
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], ParsedItemKind.TEXT)
        self.assertEqual(results[0][1], "这是普通文本")
    
    def test_mixed_content(self):
        """测试混合内容,按顺序输出"""
        response = "开始文本<|ref|>引用内容<|/ref|>中间文本<|det|>[[113, 588, 972, 927]]<|/det|>结束文本"
        width, height = 1000, 1000
        
        results = list(parse_ocr_response(response, width, height))
        
        self.assertEqual(len(results), 5)
        
        # 第一个: 开始文本
        self.assertEqual(results[0][0], ParsedItemKind.TEXT)
        self.assertEqual(results[0][1], "开始文本")
        
        # 第二个: ref 标签
        self.assertEqual(results[1][0], ParsedItemKind.REF)
        self.assertEqual(results[1][1], "引用内容")
        
        # 第三个: 中间文本
        self.assertEqual(results[2][0], ParsedItemKind.TEXT)
        self.assertEqual(results[2][1], "中间文本")
        
        # 第四个: det 标签
        self.assertEqual(results[3][0], ParsedItemKind.DET)
        self.assertEqual(results[3][1], (113, 588, 972, 927))
        
        # 第五个: 结束文本
        self.assertEqual(results[4][0], ParsedItemKind.TEXT)
        self.assertEqual(results[4][1], "结束文本")
    
    def test_multiple_det_tags(self):
        """测试多个 det 标签"""
        response = "<|det|>[[100, 200, 300, 400]]<|/det|><|det|>[[500, 600, 700, 800]]<|/det|>"
        width, height = 1000, 1000
        
        results = list(parse_ocr_response(response, width, height))
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], ParsedItemKind.DET)
        self.assertEqual(results[0][1], (100, 200, 300, 400))
        self.assertEqual(results[1][0], ParsedItemKind.DET)
        self.assertEqual(results[1][1], (500, 600, 700, 800))
    
    def test_coordinate_scaling(self):
        """测试坐标缩放计算"""
        response = "<|det|>[[500, 500, 1000, 1000]]<|/det|>"
        width, height = 2000, 2000
        
        results = list(parse_ocr_response(response, width, height))
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], ParsedItemKind.DET)
        # 500/1000 * 2000 = 1000, 1000/1000 * 2000 = 2000
        self.assertEqual(results[0][1], (1000, 1000, 2000, 2000))
    
    def test_det_with_spaces(self):
        """测试 det 标签中坐标有空格的情况"""
        response = "<|det|>[[100,  200,   300,    400]]<|/det|>"
        width, height = 1000, 1000
        
        results = list(parse_ocr_response(response, width, height))
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], ParsedItemKind.DET)
        self.assertEqual(results[0][1], (100, 200, 300, 400))
    
    def test_empty_response(self):
        """测试空响应"""
        response = ""
        width, height = 1000, 1000
        
        results = list(parse_ocr_response(response, width, height))
        
        self.assertEqual(len(results), 0)
    
    def test_ref_and_det_adjacent(self):
        """测试 ref 和 det 标签相邻的情况"""
        response = "<|ref|>文本<|/ref|><|det|>[[113, 588, 972, 927]]<|/det|>"
        width, height = 1000, 1000
        
        results = list(parse_ocr_response(response, width, height))
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], ParsedItemKind.REF)
        self.assertEqual(results[0][1], "文本")
        self.assertEqual(results[1][0], ParsedItemKind.DET)
        self.assertEqual(results[1][1], (113, 588, 972, 927))
    
    def test_complex_ref_content(self):
        """测试 ref 标签包含特殊字符"""
        response = "<|ref|>包含特殊字符!@#$%和中文<|/ref|>"
        width, height = 1000, 1000
        
        results = list(parse_ocr_response(response, width, height))
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], ParsedItemKind.REF)
        self.assertEqual(results[0][1], "包含特殊字符!@#$%和中文")


if __name__ == '__main__':
    unittest.main()
