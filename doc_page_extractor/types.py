from dataclasses import dataclass
from typing import Literal
from enum import Enum
from PIL.Image import Image
from .rectangle import Rectangle

@dataclass
class OCRFragment:
  order: int
  text: str
  rank: float
  rect: Rectangle

class LayoutClass(Enum):
  TITLE = 0
  PLAIN_TEXT = 1
  ABANDON = 2
  FIGURE = 3
  FIGURE_CAPTION = 4
  TABLE = 5
  TABLE_CAPTION = 6
  TABLE_FOOTNOTE = 7
  ISOLATE_FORMULA = 8
  FORMULA_CAPTION = 9

@dataclass
class BaseLayout:
  rect: Rectangle
  fragments: list[OCRFragment]

@dataclass
class PlainLayout(BaseLayout):
  cls: Literal[
    LayoutClass.TITLE,
    LayoutClass.PLAIN_TEXT,
    LayoutClass.ABANDON,
    LayoutClass.FIGURE,
    LayoutClass.FIGURE_CAPTION,
    LayoutClass.TABLE,
    LayoutClass.TABLE_CAPTION,
    LayoutClass.TABLE_FOOTNOTE,
    LayoutClass.FORMULA_CAPTION,
  ]

@dataclass
class FormulaLayout(BaseLayout):
  latex: str | None
  cls: LayoutClass.ISOLATE_FORMULA

Layout = PlainLayout | FormulaLayout

@dataclass
class ExtractedResult:
  rotation: float
  layouts: list[Layout]
  extracted_image: Image
  adjusted_image: Image | None