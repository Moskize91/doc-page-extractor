import os
import torch

from typing import Generator
from dataclasses import dataclass
from transformers import LayoutLMv3ForTokenClassification

from .types import Layout, LayoutClass
from .layoutreader import prepare_inputs, boxes2inputs, parse_logits
from .utils import ensure_dir


@dataclass
class _BBox:
  layout_index: int
  fragment_index: int
  virtual: bool
  value: tuple[float, float, float, float]

class LayoutOrder:
  def __init__(self, model_path: str):
    self._model_path: str = model_path
    self._model: LayoutLMv3ForTokenClassification | None = None

  def _get_model(self) -> LayoutLMv3ForTokenClassification:
    if self._model is None:
      model_path = ensure_dir(self._model_path)
      self._model = LayoutLMv3ForTokenClassification.from_pretrained(
        pretrained_model_name_or_path="hantian/layoutreader",
        cache_dir=model_path,
        local_files_only=os.path.exists(os.path.join(model_path, "models--hantian--layoutreader")),
      )
    return self._model

  def sort(self, layouts: list[Layout], size: tuple[int, int]) -> list[Layout]:
    width, height = size
    if width == 0 or height == 0:
      return layouts

    layout_orders = self._layout_orders(layouts, width, height)
    if layout_orders is None:
      return layouts

    mean_orders: list[float] = []
    sorted_layouts: list[tuple[int, Layout]] = []
    empty_layouts: list[tuple[int, Layout]] = []

    for i, orders in enumerate(layout_orders):
      layout = layouts[i]
      mean_order = 0.0
      if len(orders) == 0:
        empty_layouts.append((i, layout))
      else:
        sorted_layouts.append((i, layout))
        mean_order = self._median(orders)
        for order, fragment in zip(orders, layout.fragments):
          fragment.order = order
      mean_orders.append(mean_order)

    sorted_layouts.sort(key=lambda x: mean_orders[x[0]])

    # try to maintain the order of empty layouts and other layouts as much as possible
    for i, layout in empty_layouts:
      max_less_index: int = -1
      max_less_layout: Layout | None = None
      max_less_index_in_enumerated: int = -1
      for j, (k, sorted_layout) in enumerate(sorted_layouts):
        if k < i and k > max_less_index:
          max_less_index = k
          max_less_layout = sorted_layout
          max_less_index_in_enumerated = j

      if max_less_layout is None:
        sorted_layouts.insert(0, (i, layout))
      else:
        sorted_layouts.insert(max_less_index_in_enumerated + 1, (i, layout))

    return [layout for _, layout in sorted_layouts]

  def _layout_orders(self, layouts: list[Layout], width: int, height: int) -> list[list[float]] | None:
    line_height = self._line_height(layouts)
    bbox_list: list[_BBox] = []

    for i, layout in enumerate(layouts):
      if layout.cls == LayoutClass.PLAIN_TEXT and \
         len(layout.fragments) > 0:
        for j, fragment in enumerate(layout.fragments):
          bbox_list.append(_BBox(
            layout_index=i,
            fragment_index=j,
            virtual=False,
            value=fragment.rect.wrapper,
          ))
      else:
        bbox_list.extend(
          self._generate_virtual_lines(
            layout=layout,
            layout_index=i,
            line_height=line_height,
            width=width,
            height=height,
          ),
        )

    if len(bbox_list) > 200:
      # https://github.com/opendatalab/MinerU/blob/980f5c8cd70f22f8c0c9b7b40eaff6f4804e6524/magic_pdf/pdf_parse_union_core_v2.py#L522
      return None

    layoutreader_size = 1000.0
    x_scale = layoutreader_size / float(width)
    y_scale = layoutreader_size / float(height)

    for bbox in bbox_list:
      x0, y0, x1, y1 = self._squeeze(bbox.value, width, height)
      x0 = round(x0 * x_scale)
      y0 = round(y0 * y_scale)
      x1 = round(x1 * x_scale)
      y1 = round(y1 * y_scale)
      bbox.value = (x0, y0, x1, y1)

    bbox_list.sort(key=lambda b: b.value)
    model = self._get_model()

    with torch.no_grad():
      inputs = boxes2inputs([list(bbox.value) for bbox in bbox_list])
      inputs = prepare_inputs(inputs, model)
      logits = model(**inputs).logits.cpu().squeeze(0)
      orders = parse_logits(logits, len(bbox_list))

    layout_orders: list[list[float]] = [[] for _ in range(len(layouts))]
    for order, bbox in zip(orders, bbox_list):
      layout_orders[bbox.layout_index].append(order)

    return layout_orders

  def _line_height(self, layouts: list[Layout]) -> float:
    line_height: float = 0.0
    count: int = 0
    for layout in layouts:
      for fragment in layout.fragments:
        _, height = fragment.rect.size
        line_height += height
        count += 1
    if count == 0:
      return 10.0
    return line_height / float(count)

  def _generate_virtual_lines(
        self,
        layout: Layout,
        layout_index: int,
        line_height: float,
        width: int,
        height: int,
      ) -> Generator[_BBox, None, None]:

    # https://github.com/opendatalab/MinerU/blob/980f5c8cd70f22f8c0c9b7b40eaff6f4804e6524/magic_pdf/pdf_parse_union_core_v2.py#L451-L490
    x0, y0, x1, y1 = layout.rect.wrapper
    layout_height = y1 - y0
    layout_weight = x1 - x0
    lines = int(layout_height / line_height)

    if layout_height <= line_height * 2:
      yield _BBox(
        layout_index=layout_index,
        fragment_index=0,
        virtual=True,
        value=(x0, y0, x1, y1),
      )
      return

    elif layout_height <= height * 0.25 or \
         width * 0.5 <= layout_weight or \
         width * 0.25 < layout_weight:
      if layout_weight > width * 0.4:
        lines = 3
      elif layout_weight <= width * 0.25:
        if layout_height / layout_weight > 1.2:  # 细长的不分
          yield _BBox(
            layout_index=layout_index,
            fragment_index=0,
            virtual=True,
            value=(x0, y0, x1, y1),
          )
          return
        else:  # 不细长的还是分成两行
          lines = 2

    line_height = (y1 - y0) / lines
    current_y = y0

    for i in range(lines):
      yield _BBox(
        layout_index=layout_index,
        fragment_index=i,
        virtual=True,
        value=(x0, current_y, x1, current_y + line_height),
      )
      current_y += line_height

  def _median(self, numbers: list[int]) -> float:
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)

    # 判断是奇数还是偶数个元素
    if n % 2 == 1:
      # 奇数情况，直接取中间的数
      return float(sorted_numbers[n // 2])
    else:
      # 偶数情况，取中间两个数的平均值
      mid1 = sorted_numbers[n // 2 - 1]
      mid2 = sorted_numbers[n // 2]
      return float((mid1 + mid2) / 2)

  def _squeeze(self, bbox: _BBox, width: int, height: int) -> _BBox:
    x0, y0, x1, y1 = bbox
    x0 = self._squeeze_value(x0, width)
    x1 = self._squeeze_value(x1, width)
    y0 = self._squeeze_value(y0, height)
    y1 = self._squeeze_value(y1, height)
    return x0, y0, x1, y1

  def _squeeze_value(self, position: float, size: int) -> float:
    if position < 0:
      position = 0.0
    if position > size:
      position = float(size)
    return position