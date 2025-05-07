import torch

from typing import Generator
from dataclasses import dataclass
from transformers import LayoutLMv3ForTokenClassification

from .types import Layout, LayoutClass, GetModelDir
from .layoutreader import prepare_inputs, boxes2inputs, parse_logits


@dataclass
class _BBox:
  layout_index: int
  fragment_index: int
  virtual: bool
  order: int
  value: tuple[float, float, float, float]

class LayoutOrder:
  def __init__(self, get_model_dir: GetModelDir):
    self._model_path: str = get_model_dir()
    self._model: LayoutLMv3ForTokenClassification | None = None

  def _get_model(self) -> LayoutLMv3ForTokenClassification:
    if self._model is None:
      self._model = LayoutLMv3ForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=self._model_path,
        local_files_only=True,
      )
    return self._model

  def sort(self, layouts: list[Layout], size: tuple[int, int]) -> list[Layout]:
    width, height = size
    if width == 0 or height == 0:
      return layouts

    bbox_list = self._order_and_get_bbox_list(
      layouts=layouts,
      width=width,
      height=height,
    )
    if bbox_list is None:
      return layouts

    return self._sort_layouts_and_fragments(layouts, bbox_list)

  def _order_and_get_bbox_list(
      self,
      layouts: list[Layout],
      width: int,
      height: int,
    ) -> list[_BBox] | None:

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
            order=0,
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

    bbox_list.sort(key=lambda b: b.value) # 必须排序，乱序传入 layoutreader 会令它无法识别正确顺序
    model = self._get_model()

    with torch.no_grad():
      inputs = boxes2inputs([list(bbox.value) for bbox in bbox_list])
      inputs = prepare_inputs(inputs, model)
      logits = model(**inputs).logits.cpu().squeeze(0)
      orders = parse_logits(logits, len(bbox_list))

    sorted_bbox_list = [bbox_list[i] for i in orders]
    for i, bbox in enumerate(sorted_bbox_list):
      bbox.order = i

    return sorted_bbox_list

  def _sort_layouts_and_fragments(self, layouts: list[Layout], bbox_list: list[_BBox]):
    layout_bbox_list: list[list[_BBox]] = [[] for _ in range(len(layouts))]
    for bbox in bbox_list:
      layout_bbox_list[bbox.layout_index].append(bbox)

    layouts_with_median_order: list[tuple[Layout, float]] = []
    for layout_index, bbox_list in enumerate(layout_bbox_list):
      layout = layouts[layout_index]
      orders = [b.order for b in bbox_list] # virtual bbox 保证了 orders 不可能为空
      median_order = self._median(orders)
      layouts_with_median_order.append((layout, median_order))

    for layout, bbox_list in zip(layouts, layout_bbox_list):
      for bbox in bbox_list:
        if not bbox.virtual:
          layout.fragments[bbox.fragment_index].order = bbox.order
      if all(not bbox.virtual for bbox in bbox_list):
        layout.fragments.sort(key=lambda f: f.order)

    layouts_with_median_order.sort(key=lambda x: x[1])
    layouts = [layout for layout, _ in layouts_with_median_order]
    next_fragment_order: int = 0

    for layout in layouts:
      for fragment in layout.fragments:
        fragment.order = next_fragment_order
        next_fragment_order += 1

    return layouts

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
        order=0,
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
            order=0,
            value=(x0, y0, x1, y1),
          )
          return
        else:  # 不细长的还是分成两行
          lines = 2

    lines = max(1, lines)
    line_height = (y1 - y0) / lines
    current_y = y0

    for i in range(lines):
      yield _BBox(
        layout_index=layout_index,
        fragment_index=i,
        virtual=True,
        order=0,
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