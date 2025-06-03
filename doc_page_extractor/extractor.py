import torch

from os import PathLike
from typing import Literal, Generator
from PIL.Image import Image
from doclayout_yolo import YOLOv10

from .model import Model, HuggingfaceModel
from .ocr import OCR
from .ocr_corrector import correct_fragments
from .raw_optimizer import RawOptimizer
from .rectangle import intersection_area, Rectangle
from .table import Table
from .latex import LaTeX
from .layout_order import LayoutOrder
from .overlap import merge_fragments_as_line, remove_overlap_layouts
from .clipper import clip_from_image
from .types import (
  ExtractedResult,
  OCRFragment,
  Layout,
  LayoutClass,
  PlainLayout,
  TableLayout,
  FormulaLayout,
  TableLayoutParsedFormat
)


class DocExtractor:
  def __init__(
        self,
        model_cache_dir: PathLike | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
        model: Model | None = None,
      ) -> None:

    if model is None:
      if model_cache_dir is None:
        raise ValueError("You must provide a model_cache_dir or a model instance.")
      model = HuggingfaceModel(model_cache_dir)

    if device == "cuda" and not torch.cuda.is_available():
      device = "cpu"
      print("CUDA is not available. Using CPU instead.")

    self._device: Literal["cpu", "cuda"] = device
    self._model: Model = model
    self._yolo: YOLOv10 | None = None
    self._ocr: OCR = OCR(device, model)
    self._table: Table = Table(device, model)
    self._latex: LaTeX = LaTeX(device, model)
    self._layout_order: LayoutOrder = LayoutOrder(device, model)

  def extract(
      self,
      image: Image,
      extract_formula: bool,
      extract_table_format: TableLayoutParsedFormat | None = None,
      ocr_for_each_layouts: bool = False,
      adjust_points: bool = False
    ) -> ExtractedResult:

    raw_optimizer = RawOptimizer(image, adjust_points)
    fragments = list(self._ocr.search_fragments(raw_optimizer.image_np))
    raw_optimizer.receive_raw_fragments(fragments)
    layouts = list(self._yolo_extract_layouts(raw_optimizer.image))
    layouts = self._layouts_matched_by_fragments(fragments, layouts)
    layouts = remove_overlap_layouts(layouts)

    if ocr_for_each_layouts:
      self._correct_fragments_by_ocr_layouts(raw_optimizer.image, layouts)

    layouts = self._layout_order.sort(layouts, raw_optimizer.image.size)
    layouts = [layout for layout in layouts if self._should_keep_layout(layout)]

    self._parse_table_and_formula_layouts(layouts, raw_optimizer, extract_formula=extract_formula, extract_table_format=extract_table_format)

    for layout in layouts:
      layout.fragments = merge_fragments_as_line(layout.fragments)

    raw_optimizer.receive_raw_layouts(layouts)

    return ExtractedResult(
      rotation=raw_optimizer.rotation,
      layouts=layouts,
      extracted_image=image,
      adjusted_image=raw_optimizer.adjusted_image,
    )

  def _yolo_extract_layouts(self, source: Image) -> Generator[Layout, None, None]:
    # about source parameter to see:
    # https://github.com/opendatalab/DocLayout-YOLO/blob/7c4be36bc61f11b67cf4a44ee47f3c41e9800a91/doclayout_yolo/data/build.py#L157-L175
    det_res = self._get_yolo().predict(
      source=source,
      imgsz=1024,
      conf=0.2,
      device=self._device    # Device to use (e.g., "cuda" or "cpu")
    )
    boxes = det_res[0].__dict__["boxes"]

    for cls_id, rect in zip(boxes.cls, boxes.xyxy):
      cls_id = cls_id.item()
      cls=LayoutClass(round(cls_id))

      x1, y1, x2, y2 = rect
      x1 = x1.item()
      y1 = y1.item()
      x2 = x2.item()
      y2 = y2.item()
      rect = Rectangle(
        lt=(x1, y1),
        rt=(x2, y1),
        lb=(x1, y2),
        rb=(x2, y2),
      )
      if rect.is_valid:
        if cls == LayoutClass.TABLE:
          yield TableLayout(cls=cls, rect=rect, fragments=[], parsed=None)
        elif cls == LayoutClass.ISOLATE_FORMULA:
          yield FormulaLayout(cls=cls, rect=rect, fragments=[], latex=None)
        else:
          yield PlainLayout(cls=cls, rect=rect, fragments=[])

  def _layouts_matched_by_fragments(self, fragments: list[OCRFragment], layouts: list[Layout]):
    layouts_group = self._split_layouts_by_group(layouts)
    for fragment in fragments:
      for sub_layouts in layouts_group:
        layout = self._find_matched_layout(fragment, sub_layouts)
        if layout is not None:
          layout.fragments.append(fragment)
          break
    return layouts

  def _correct_fragments_by_ocr_layouts(self, source: Image, layouts: list[Layout]):
    for layout in layouts:
      correct_fragments(self._ocr, source, layout)

  def _parse_table_and_formula_layouts(
      self,
      layouts: list[Layout],
      raw_optimizer: RawOptimizer,
      extract_formula: bool,
      extract_table_format: TableLayoutParsedFormat | None,
  ):
    for layout in layouts:
      if isinstance(layout, FormulaLayout) and extract_formula:
        image = clip_from_image(raw_optimizer.image, layout.rect)
        layout.latex = self._latex.extract(image)
      elif isinstance(layout, TableLayout) and extract_table_format is not None:
        image = clip_from_image(raw_optimizer.image, layout.rect)
        parsed = self._table.predict(image, extract_table_format)
        if parsed is not None:
          layout.parsed = (parsed, extract_table_format)

  def _split_layouts_by_group(self, layouts: list[Layout]):
    texts_layouts: list[Layout] = []
    abandon_layouts: list[Layout] = []

    for layout in layouts:
      cls = layout.cls
      if cls == LayoutClass.TITLE or \
         cls == LayoutClass.PLAIN_TEXT or \
         cls == LayoutClass.FIGURE_CAPTION or \
         cls == LayoutClass.TABLE_CAPTION or \
         cls == LayoutClass.TABLE_FOOTNOTE or \
         cls == LayoutClass.FORMULA_CAPTION:
        texts_layouts.append(layout)
      elif cls == LayoutClass.ABANDON:
        abandon_layouts.append(layout)

    return texts_layouts, abandon_layouts

  def _find_matched_layout(self, fragment: OCRFragment, layouts: list[Layout]) -> Layout | None:
    fragment_area = fragment.rect.area
    primary_layouts: list[(Layout, float)] = []

    if fragment_area == 0.0:
      return None

    for layout in layouts:
      area = intersection_area(fragment.rect, layout.rect)
      if area / fragment_area > 0.85:
        primary_layouts.append((layout, layout.rect.area))

    min_area: float = float("inf")
    min_layout: Layout | None = None

    for layout, area in primary_layouts:
      if area < min_area:
        min_area = area
        min_layout = layout

    return min_layout

  def _get_yolo(self) -> YOLOv10:
    if self._yolo is None:
      model_path = self._model.get_yolo_path()
      self._yolo = YOLOv10(str(model_path))
    return self._yolo

  def _should_keep_layout(self, layout: Layout) -> bool:
    if len(layout.fragments) > 0:
      return True
    cls = layout.cls
    return (
      cls == LayoutClass.FIGURE or
      cls == LayoutClass.TABLE or
      cls == LayoutClass.ISOLATE_FORMULA
    )

