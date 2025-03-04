import os
import re

from shapely.geometry import Polygon

def ensure_dir(path: str) -> str:
  path = os.path.abspath(path)
  os.makedirs(path, exist_ok=True)
  return path

def is_space_text(text: str) -> bool:
  return re.match(r"^\s*$", text)

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