from dataclasses import dataclass
from typing import Literal
from enum import auto, Enum
from PIL.Image import Image
from .rectangle import Rectangle

@dataclass
class OCRFragment:
  order: int
  text: str
  rank: float
  rect: Rectangle

class LayoutClass(Enum):
  TITLE = auto()
  PLAIN_TEXT = auto()
  ABANDON = auto()
  FIGURE = auto()
  FIGURE_CAPTION = auto()
  TABLE = auto()
  TABLE_CAPTION = auto()
  TABLE_FOOTNOTE = auto()
  ISOLATE_FORMULA = auto()
  FORMULA_CAPTION = auto()

class TableLayoutParsedFormat(Enum):
  LATEX = auto()
  MARKDOWN = auto()
  HTML = auto()

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
    LayoutClass.TABLE_CAPTION,
    LayoutClass.TABLE_FOOTNOTE,
    LayoutClass.FORMULA_CAPTION,
  ]

@dataclass
class TableLayout(BaseLayout):
  parsed: tuple[str, TableLayoutParsedFormat] | None
  cls: LayoutClass.TABLE

@dataclass
class FormulaLayout(BaseLayout):
  latex: str | None
  cls: LayoutClass.ISOLATE_FORMULA

Layout = PlainLayout | TableLayout | FormulaLayout

@dataclass
class ExtractedResult:
  rotation: float
  layouts: list[Layout]
  extracted_image: Image
  adjusted_image: Image | None