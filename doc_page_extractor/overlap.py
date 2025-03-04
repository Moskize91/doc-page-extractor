from typing import Generator
from shapely.geometry import Polygon
from .types import Layout, OCRFragment
from .rectangle import Rectangle

def remove_overlap_layouts(layouts: list[Layout]) -> list[Layout]:
  includes_min_rate = 0.99
  removed_indexes: set[int] = set()

  for i, layout1 in enumerate(layouts):
    if i in removed_indexes:
      continue

    polygon1 = Polygon(layout1.rect)
    rates: list[float] = []
    includes_layouts: list[Layout] = []
    includes_layout_indexes: list[int] = []

    for j, layout2 in enumerate(layouts):
      if layout1 == layout2 or j in removed_indexes:
        continue
      rate = overlap_rate(
        polygon1=polygon1,
        polygon2=Polygon(layout2.rect),
      )
      if rate > 0.0:
        rates.append(rate)
        includes_layouts.append(layout2)
        includes_layout_indexes.append(j)

    if len(rates) == 0 or not all(x > includes_min_rate for x in rates):
      pass

    elif len(layout1.fragments) == 0:
      removed_indexes.add(i)
    else:
      removed_indexes.update(includes_layout_indexes)
      for layout in includes_layouts:
        layout1.fragments.extend(layout.fragments)

  return [
    layout for i, layout in enumerate(layouts)
    if i not in removed_indexes
  ]

def regroup_lines(origin_fragments: list[OCRFragment]) -> list[OCRFragment]:
  fragments: list[OCRFragment] = []
  for group in _split_fragments_into_groups(origin_fragments):
    if len(group) == 1:
      fragments.append(group[0])
      continue

    min_order: float = float("inf")
    texts: list[str] = []
    text_rate_weights: float = 0.0
    proto_texts_len: int = 0

    x1: float = float("inf")
    y1: float = float("inf")
    x2: float = float("-inf")
    y2: float = float("-inf")

    for fragment in sorted(group, key=lambda x: x.rect.lt[0] + x.rect.lb[0]):
      proto_texts_len += len(fragment.text)
      text_rate_weights += fragment.rank * len(fragment.text)
      texts.append(fragment.text)
      min_order = min(min_order, fragment.order)
      for x, y in fragment.rect:
        x1 = min(x1, x)
        y1 = min(y1, y)
        x2 = max(x2, x)
        y2 = max(y2, y)

    fragments.append(OCRFragment(
      order=min_order,
      text=" ".join(texts),
      rank=text_rate_weights / proto_texts_len,
      rect=Rectangle(
        lt=(x1, y1),
        rt=(x2, y1),
        lb=(x1, y2),
        rb=(x2, y2),
      ),
    ))
  return fragments

def _split_fragments_into_groups(fragments: list[OCRFragment]) -> Generator[list[OCRFragment], None, None]:
  group: list[OCRFragment] = []
  sum_height: float = 0.0
  sum_median: float = 0.0
  max_deviation_rate = 0.35

  for fragment in sorted(fragments, key=lambda x: x.rect.lt[1] + x.rect.rt[1]):
    _, y1, _, y2 = fragment.rect.wrapper
    height = y2 - y1
    median = (y1 + y2) / 2.0

    if len(group) > 0:
      next_mean_median = (sum_median + median) / (len(group) + 1)
      next_mean_height = (sum_height + height) / (len(group) + 1)
      deviation_rate = abs(median - next_mean_median) / next_mean_height

      if deviation_rate > max_deviation_rate:
        yield group
        group = []
        sum_height = 0.0
        sum_median = 0.0

    group.append(fragment)
    sum_height += height
    sum_median += median

  if len(group) > 0:
    yield group

# calculating overlap ratio: The reason why area is not used is
# that most of the measurements are of rectangles representing text lines.
# they are very sensitive to changes in height because they are very thin and long.
# In order to make it equally sensitive to length and width, the ratio of area is not used.
def overlap_rate(polygon1: Polygon, polygon2: Polygon) -> float:
  intersection: Polygon = polygon1.intersection(polygon2)
  if intersection.is_empty:
    return 0.0
  else:
    overlay_width, overlay_height = _polygon_size(intersection)
    polygon2_width, polygon2_height = _polygon_size(polygon2)
    return (overlay_width / polygon2_width + overlay_height / polygon2_height) / 2.0

def _polygon_size(polygon: Polygon) -> tuple[float, float]:
  x1: float = float("inf")
  y1: float = float("inf")
  x2: float = float("-inf")
  y2: float = float("-inf")
  for x, y in polygon.exterior.coords:
    x1 = min(x1, x)
    y1 = min(y1, y)
    x2 = max(x2, x)
    y2 = max(y2, y)
  return x2 - x1, y2 - y1