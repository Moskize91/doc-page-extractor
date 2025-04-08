import os

from typing import Literal
from pathlib import Path
from PIL.Image import Image
from transformers import LayoutLMv3ForTokenClassification
from doclayout_yolo import YOLOv10

from .ocr import OCR
from .ocr_corrector import correct_fragments
from .raw_optimizer import RawOptimizer
from .rectangle import intersection_area, Rectangle
from .types import ExtractedResult, OCRFragment, LayoutClass, Layout
from .downloader import download
from .overlap import merge_fragments_as_line, remove_overlap_layouts
from .order import order_layouts_and_fragments, OrderAI
from .utils import ensure_dir


class DocExtractor:
  def __init__(
      self,
      model_dir_path: str,
      device: Literal["cpu", "cuda"] = "cpu",
      ocr_for_each_layouts: bool = True,
      order_by_layoutreader: bool = False,
    ):
    self._model_dir_path: str = model_dir_path
    self._device: Literal["cpu", "cuda"] = device
    self._ocr_for_each_layouts: bool = ocr_for_each_layouts
    self._order_by_layoutreader: bool = order_by_layoutreader
    self._ocr: OCR = OCR(device, model_dir_path)
    self._yolo: YOLOv10 | None = None
    self._layout: LayoutLMv3ForTokenClassification | None = None

  def extract(
      self,
      image: Image,
      adjust_points: bool = False,
    ) -> ExtractedResult:

    raw_optimizer = RawOptimizer(image, adjust_points)
    fragments = list(self._ocr.search_fragments(raw_optimizer.image_np))
    raw_optimizer.receive_raw_fragments(fragments)

    layouts = self._get_layouts(raw_optimizer.image)
    layouts = self._layouts_matched_by_fragments(fragments, layouts)
    layouts = remove_overlap_layouts(layouts)

    if self._ocr_for_each_layouts:
      self._correct_fragments_by_ocr_layouts(raw_optimizer.image, layouts)

    layouts = [layout for layout in layouts if self._should_keep_layout(layout)]
    for layout in layouts:
      layout.fragments = merge_fragments_as_line(layout.fragments)

    order_ai: OrderAI | None = None
    if self._order_by_layoutreader:
      width, height = raw_optimizer.image.size
      order_ai = OrderAI(
        width=width,
        height=height,
        model=self._get_layoutreader(),
      )
    layouts = order_layouts_and_fragments(layouts, order_ai)
    raw_optimizer.receive_raw_layouts(layouts)

    return ExtractedResult(
      rotation=raw_optimizer.rotation,
      layouts=layouts,
      extracted_image=image,
      adjusted_image=raw_optimizer.adjusted_image,
    )

  def _get_layouts(self, source: Image) -> list[Layout]:
    # about source parameter to see:
    # https://github.com/opendatalab/DocLayout-YOLO/blob/7c4be36bc61f11b67cf4a44ee47f3c41e9800a91/doclayout_yolo/data/build.py#L157-L175
    det_res = self._get_yolo().predict(
      source=source,
      imgsz=1024,
      conf=0.2,
      device=self._device    # Device to use (e.g., "cuda" or "cpu")
    )
    boxes = det_res[0].__dict__["boxes"]
    layouts: list[Layout] = []

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
      layouts.append(Layout(cls, rect, []))

    return layouts

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
      yolo_model_url = "https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/resolve/main/models/Layout/YOLO/doclayout_yolo_ft.pt"
      yolo_model_name = "doclayout_yolo_ft.pt"
      yolo_model_path = Path(os.path.join(self._model_dir_path, yolo_model_name))
      if not yolo_model_path.exists():
        download(yolo_model_url, yolo_model_path)
      self._yolo = YOLOv10(str(yolo_model_path))
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

  def _get_layoutreader(self) -> LayoutLMv3ForTokenClassification:
    if self._layout is None:
      cache_dir = ensure_dir(
        os.path.join(self._model_dir_path, "layoutreader"),
      )
      self._layout = LayoutLMv3ForTokenClassification.from_pretrained(
        pretrained_model_name_or_path="hantian/layoutreader",
        cache_dir=cache_dir,
        local_files_only=os.path.exists(os.path.join(cache_dir, "models--hantian--layoutreader")),
      )
    return self._layout
