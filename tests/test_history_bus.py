import os
import unittest

from PIL import Image
from doc_page_extractor import DocExtractor, Layout, LayoutClass


class TestGroup(unittest.TestCase):
  def test_history_bugs(self):
    model_path = os.path.join(self._project_path(), "model")
    image_path = os.path.join(self._project_path(), "tests", "images", "figure.png")
    os.makedirs(model_path, exist_ok=True)

    extractor = DocExtractor(model_path, "cpu")
    layouts: list[tuple[LayoutClass, list[str]]]

    with Image.open(image_path) as image:
      result = extractor.extract(image, extract_formula=False)
      layouts = [self._format_Layout(layout) for layout in result.layouts]

    self.assertEqual(layouts, [
      (LayoutClass.PLAIN_TEXT, [
        "口的11.8%①。这既是江南农业落后的反映，又是它的原因。当战国以",
        "后黄河流域因铁器牛耕的普及获得基本的开发，农区联结成一大片的",
        "时候，南方农业开发始终没有突破星点状或斑块状分布的格局。由于",
        "地旷人稀，耕作相当粗放，许多水田采取火耕水瓣的方式，旱田则多",
        "行刀耕火种②。司马迁在《史记·货殖列传》中说：“总之，楚越之",
        "地，地厂人希，饭稻囊鱼，或火耕而水瓣，果隋（蕨）赢（螺）蛤，",
        "不待贾而足，地势饶食，无饥谨之患，以故皆偷生，无积聚而多",
        "贫。”这种概括虽然未免太突出了南方经济的落后面，有一定片面性，",
        "但大体还是反映了实际情形的。战国秦汉时期，南方与黄河流域农业",
        "的差距显然拉大了。",
      ]),
      (LayoutClass.FIGURE, []),
      (LayoutClass.FIGURE_CAPTION, [
        "西晋陶水田犁耙模型（广东连县出土）"
      ]),
      (LayoutClass.FIGURE, []),
      (LayoutClass.FIGURE_CAPTION, [
        "南朝陶耙田模型 （广西苍梧倒水出土）"
      ]),
      (LayoutClass.PLAIN_TEXT, [
        "①据赵文林、谢淑君：《中国人口史》（人民出版社1988年）有关资料统计。",
        "②《盐铁论·通有》：“荆扬…………伐木而树谷，焚莱而播粟，火耕而水。”"
      ]),
      (LayoutClass.ABANDON, [
        "136"
      ]),
    ])

  def _format_Layout(self, layout: Layout) -> tuple[LayoutClass, list[str]]:
    return layout.cls, [f.text.strip() for f in layout.fragments]

  def _project_path(self) -> str:
    return os.path.abspath(os.path.join(__file__, "..", ".."))