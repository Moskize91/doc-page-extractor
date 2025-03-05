from typing import Generator
from shapely.geometry import Polygon
from .types import Layout, OCRFragment
from .rectangle import Rectangle


_INCLUDES_MIN_RATE = 0.99

def remove_overlap_layouts(layouts: list[Layout]) -> list[Layout]:
  removed_indexes: set[int] = set()
  rates_matrix = _rate_matrix(layouts)

  for i, layout in enumerate(layouts):
    if i in removed_indexes or all(
      rate == 0.0 or rate > _INCLUDES_MIN_RATE
      for rate in _rates_with_other(rates_matrix, i)
    ):
      continue

    if len(layout.fragments) == 0:
      removed_indexes.add(i)
    else:
      for j in _search_includes_indexes(rates_matrix, i):
        removed_indexes.add(j)
        layout.fragments.extend(layouts[j].fragments)

  return [
    layout for i, layout in enumerate(layouts)
    if i not in removed_indexes
  ]

def _rate_matrix(layouts: list[Layout]) -> list[list[float]]:
  length: int = len(layouts)
  polygons: list[Polygon] = [Polygon(layout.rect) for layout in layouts]
  rate_matrix: list[list[float]] = [[1.0 for _ in range(length)] for _ in range(length)]
  for i in range(length):
    polygon1 = polygons[i]
    rates = rate_matrix[i]
    for j in range(length):
      if i != j:
        polygon2 = polygons[j]
        rates[j] = overlap_rate(polygon1, polygon2)
  return rate_matrix

def _rates_with_other(rates_matrix: list[list[float]], index: int):
  for i, rate in enumerate(rates_matrix[index]):
    if i != index:
      yield rate

def _search_includes_indexes(layout_matrix: list[list[float]], index: int):
  for i, rate in enumerate(layout_matrix[index]):
    if i != index and rate >= _INCLUDES_MIN_RATE:
      yield i

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