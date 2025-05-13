from dataclasses import dataclass
from typing import Literal, Callable, Protocol, runtime_checkable, List
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

class TableLayoutParsedFormat(Enum):
  LATEX = auto()
  MARKDOWN = auto()
  HTML = auto()

@dataclass
class BaseLayout:
  rect: Rectangle
  fragments: List[OCRFragment]

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
class ExtractParams:
  extract_formula: bool
  extract_table_format: TableLayoutParsedFormat | None = None
  ocr_for_each_layouts: bool = False
  adjust_points: bool = False

@dataclass
class ExtractedResult:
  rotation: float
  layouts: List[Layout]
  extracted_image: Image
  adjusted_image: Image | None

GetModelDir = Callable[[], str]


@runtime_checkable
class ModelsDownloader(Protocol):

  def onnx_ocr(self) -> str:
    pass

  def yolo(self) -> str:
    pass

  def layoutreader(self) -> str:
    pass

  def struct_eqtable(self) -> str:
    pass

  def latex(self) -> str:
    pass


