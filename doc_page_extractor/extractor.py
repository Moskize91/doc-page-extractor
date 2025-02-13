import os
import sys
import numpy as np

from typing import Literal, Generator
from pathlib import Path
from PIL.Image import Image
from transformers import LayoutLMv3ForTokenClassification
from doclayout_yolo import YOLOv10
from paddleocr import PaddleOCR

from .layoutreader import prepare_inputs, boxes2inputs, parse_logits
from .raw_optimizer import RawOptimizer
from .rectangle import intersection_area, Rectangle
from .types import ExtractedResult, OCRFragment, LayoutClass, Layout
from .downloader import download
from .utils import ensure_dir


# https://github.com/PaddlePaddle/PaddleOCR/blob/2c0c4beb0606819735a16083cdebf652939c781a/paddleocr.py#L108-L157
PaddleLang = Literal["ch", "en", "korean", "japan", "chinese_cht", "ta", "te", "ka", "latin", "arabic", "cyrillic", "devanagari"]

class DocExtractor:
  def __init__(
      self,
      model_dir_path: str,
      device: Literal["cpu", "cuda"] = "cpu",
      order_by_layoutreader: bool = True,
    ):
    self._model_dir_path: str = model_dir_path
    self._device: Literal["cpu", "cuda"] = device
    self._order_by_layoutreader: bool = order_by_layoutreader
    self._ocr_and_lan: tuple[PaddleOCR, PaddleLang] | None = None
    self._yolo: YOLOv10 | None = None
    self._layout: LayoutLMv3ForTokenClassification | None = None

  def extract(
      self,
      image: Image,
      lang: PaddleLang,
      adjust_points: bool = False,
    ) -> ExtractedResult:

    raw_optimizer = RawOptimizer(image, adjust_points)
    fragments = list(self._search_orc_fragments(raw_optimizer.image_np, lang))
    raw_optimizer.receive_raw_fragments(fragments)

    if self._order_by_layoutreader:
      width, height = raw_optimizer.image.size
      self._order_fragments(width, height, fragments)

    layouts = self._get_layouts(raw_optimizer.image)
    layouts = self._layouts_matched_by_fragments(fragments, layouts)
    raw_optimizer.receive_raw_layouts(layouts)

    return ExtractedResult(
      rotation=raw_optimizer.rotation,
      layouts=layouts,
      extracted_image=image,
      adjusted_image=raw_optimizer.adjusted_image,
    )

  # https://paddlepaddle.github.io/PaddleOCR/latest/quick_start.html#_2
  def _search_orc_fragments(self, image: np.ndarray, lang: PaddleLang) -> Generator[OCRFragment, None, None]:
    index: int = 0
    # about img parameter to see
    # https://github.com/PaddlePaddle/PaddleOCR/blob/2c0c4beb0606819735a16083cdebf652939c781a/paddleocr.py#L582-L619
    for item in self._get_ocr(lang).ocr(img=image, cls=True):
      for line in item:
        react: list[list[float]] = line[0]
        text, rank = line[1]
        yield OCRFragment(
          order=index,
          text=text,
          rank=rank,
          rect=Rectangle(
            lt=(react[0][0], react[0][1]),
            rt=(react[1][0], react[1][1]),
            rb=(react[2][0], react[2][1]),
            lb=(react[3][0], react[3][1]),
          ),
        )
        index += 1

  def _order_fragments(self, width: int, height: int, fragments: list[OCRFragment]):
    layout_model = self._get_layout()
    boxes: list[list[int]] = []
    steps: float = 1000.0 # max value of layoutreader
    x_rate: float = 1.0
    y_rate: float = 1.0
    x_offset: float = 0.0
    y_offset: float = 0.0
    if width > height:
      y_rate = height / width
      y_offset = (1.0 - y_rate) / 2.0
    else:
      x_rate = width / height
      x_offset = (1.0 - x_rate) / 2.0

    for left, top, right, bottom in self._collect_rate_boxes(fragments):
      boxes.append([
        round((left * x_rate + x_offset) * steps),
        round((top * y_rate + y_offset) * steps),
        round((right * x_rate + x_offset) * steps),
        round((bottom * y_rate + y_offset) * steps),
      ])
    inputs = boxes2inputs(boxes)
    inputs = prepare_inputs(inputs, layout_model)
    logits = layout_model(**inputs).logits.cpu().squeeze(0)
    orders: list[int] = parse_logits(logits, len(boxes))

    for order, fragment in zip(orders, fragments):
      fragment.order = order

  def _get_layouts(self, source: Image) -> list[Layout]:
    # about source parameter to see:
    # https://github.com/opendatalab/DocLayout-YOLO/blob/7c4be36bc61f11b67cf4a44ee47f3c41e9800a91/doclayout_yolo/data/build.py#L157-L175
    det_res = self._get_yolo().predict(
      source=source,
      imgsz=1024,
      conf=0.2,
      device=self._device    # Device to use (e.g., "cuda:0" or "cpu")
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
    for fragment in fragments:
      max_area: float = 0.0
      max_layout_index: int = 0
      for i, layout in enumerate(layouts):
        area = intersection_area(fragment.rect, layout.rect)
        if area > max_area:
          max_area = area
          max_layout_index = i
      layouts[max_layout_index].fragments.append(fragment)

    for layout in layouts:
      layout.fragments.sort(key=lambda x: x.order)

    layouts.sort(key=self._layout_order)

    return layouts

  def _layout_order(self, layout: Layout) -> int:
    fragments = layout.fragments
    if len(fragments) == 0:
      return sys.maxsize
    else:
      return fragments[0].order

  def _get_ocr(self, lang: PaddleLang) -> PaddleOCR:
    if self._ocr_and_lan is not None:
      ocr, origin_lang = self._ocr_and_lan
      if lang == origin_lang:
        return ocr

    ocr = PaddleOCR(
      lang=lang,
      use_angle_cls=True,
      use_gpu=self._device.startswith("cuda"),
      det_model_dir=ensure_dir(
        os.path.join(self._model_dir_path, "paddle", "det"),
      ),
      rec_model_dir=ensure_dir(
        os.path.join(self._model_dir_path, "paddle", "rec"),
      ),
      cls_model_dir=ensure_dir(
        os.path.join(self._model_dir_path, "paddle", "cls"),
      ),
    )
    self._ocr_and_lan = (ocr, lang)
    return ocr

  def _get_yolo(self) -> YOLOv10:
    if self._yolo is None:
      yolo_model_url = "https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/resolve/main/models/Layout/YOLO/doclayout_yolo_ft.pt"
      yolo_model_name = "doclayout_yolo_ft.pt"
      yolo_model_path = Path(os.path.join(self._model_dir_path, yolo_model_name))
      if not yolo_model_path.exists():
        download(yolo_model_url, yolo_model_path)
      self._yolo = YOLOv10(str(yolo_model_path))
    return self._yolo

  def _get_layout(self) -> LayoutLMv3ForTokenClassification:
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

  def _collect_rate_boxes(self, fragments: list[OCRFragment]):
    boxes = self._get_boxes(fragments)
    left = float("inf")
    top = float("inf")
    right = float("-inf")
    bottom = float("-inf")

    for _left, _top, _right, _bottom in boxes:
      left = min(left, _left)
      top = min(top, _top)
      right = max(right, _right)
      bottom = max(bottom, _bottom)

    width = right - left
    height = bottom - top

    for _left, _top, _right, _bottom in boxes:
      yield (
        (_left - left) / width,
        (_top - top) / height,
        (_right - left) / width,
        (_bottom - top) / height,
      )

  def _get_boxes(self, fragments: list[OCRFragment]):
    boxes: list[tuple[float, float, float, float]] = []
    for fragment in fragments:
      left: float = float("inf")
      top: float = float("inf")
      right: float = float("-inf")
      bottom: float = float("-inf")
      for x, y in fragment.rect:
        left = min(left, x)
        top = min(top, y)
        right = max(right, x)
        bottom = max(bottom, y)
      boxes.append((left, top, right, bottom))
    return boxes