import unittest

from doc_page_extractor.unimernet import Unimernet
from tests.utils import path_from_root


class TestGroup(unittest.TestCase):
  def test_formula(self):
    image_path = path_from_root("tests", "images", "formula.png")
    config_path = path_from_root(
      "doc_page_extractor", "data",
      "unimernet", "demo.yaml",
    )
    unimernet = Unimernet(config_path, "cpu")
    latex_code = unimernet.process_image(image_path)

    print("\nComplete")
    print(latex_code)