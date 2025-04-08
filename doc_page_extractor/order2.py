from dataclasses import dataclass
from typing import Iterable
from transformers import LayoutLMv3ForTokenClassification
from .types import Layout, OCRFragment
from .layoutreader import prepare_inputs, boxes2inputs, parse_logits
from .rectangle import Rectangle


@dataclass
class OrderAI:
  width: int
  height: int
  model: LayoutLMv3ForTokenClassification

def order_layouts_and_fragments(layouts: list[Layout], order_ai: OrderAI | None) -> list[Layout]:
  _order_by_primitive_layouts(layouts)
  if order_ai is not None:
    order_fragments_by_ai(layouts, order_ai)
  return _sort_layouts(layouts)

def _order_by_primitive_layouts(layouts: list[Layout]):
  order: int = 0
  for layout in layouts:
    layout.fragments.sort(key=lambda x: _top_line_2y(x.rect))
    for fragment in layout.fragments:
      fragment.order = order
      order += 1

def order_fragments_by_ai(fragments: list[OCRFragment], order_ai: OrderAI):
  print(">>>", len(fragments))
  width = order_ai.width
  height = order_ai.height

  if width == 0 or height == 0:
    return

  layoutreader_model = order_ai.model
  boxes: list[list[int]] = []
  steps: float = 1000.0 # max value of layoutreader
  x_scale = steps / width
  y_scale = steps / height

  for fragment in fragments:
    left, top, right, bottom = fragment.rect.wrapper
    boxes.append([
      round(left * x_scale),
      round(top * y_scale),
      round(right * x_scale),
      round(bottom * y_scale),
    ])

  inputs = boxes2inputs(boxes)
  inputs = prepare_inputs(inputs, layoutreader_model)
  logits = layoutreader_model(**inputs).logits.cpu().squeeze(0)
  orders: list[int] = parse_logits(logits, len(boxes))

  for order, fragment in zip(orders, fragments):
    fragment.order = order

def _sort_layouts(layouts: list[Layout]) -> list[Layout]:
  layouts.sort(key=lambda layout: layout.rect.lt[1] + layout.rect.rt[1])

  sorted_layouts: list[tuple[int, Layout]] = []
  empty_layouts: list[tuple[int, Layout]] = []

  for i, layout in enumerate(layouts):
    if len(layout.fragments) > 0:
      sorted_layouts.append((i, layout))
    else:
      empty_layouts.append((i, layout))

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

def _collect_rate_boxes(fragments: Iterable[OCRFragment]):
  boxes = _get_boxes(fragments)
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

  if width == 0 or height == 0:
    return

  for _left, _top, _right, _bottom in boxes:
    yield (
      (_left - left) / width,
      (_top - top) / height,
      (_right - left) / width,
      (_bottom - top) / height,
    )

def _get_boxes(fragments: Iterable[OCRFragment]):
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

def _iter_fragments(layouts: list[Layout]):
  for layout in layouts:
    yield from layout.fragments

def _top_line_2y(rect: Rectangle):
  return rect.lt[1] + rect.rt[1]