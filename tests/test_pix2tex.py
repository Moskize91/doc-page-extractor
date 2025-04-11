import unittest

from PIL import Image
from pix2tex.cli import LatexOCR


class TestPix2Tex(unittest.TestCase):
  def test_pix_to_tex(self):
    img = Image.open('/Users/taozeyu/codes/github.com/moskize91/doc-page-extractor/tests/images/formula.png')
    model = LatexOCR()
    print(model(img))