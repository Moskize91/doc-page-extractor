import unittest

from PIL import Image
from doc_page_extractor.latex import LaTeX


class TestPix2Tex(unittest.TestCase):
  def test_pix_to_tex(self):
    latex = LaTeX("/Users/taozeyu/codes/github.com/moskize91/doc-page-extractor/model/latex")
    image = Image.open('/Users/taozeyu/codes/github.com/moskize91/doc-page-extractor/tests/images/formula.png')
    print(latex.transform(image))