import os
import re

from shapely.geometry import Polygon
from .rectangle import Rectangle

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
def overlap_rate(rect1: Rectangle, rect2: Rectangle) -> float:
  polygon1 = Polygon(rect1)
  polygon2 = Polygon(rect2)
  intersection: Polygon = polygon1.intersection(polygon2)
  if intersection.is_empty:
    return 0.0

  x1: float = float("inf")
  y1: float = float("inf")
  x2: float = float("-inf")
  y2: float = float("-inf")
  for x, y in intersection.exterior.coords:
    x1 = min(x1, x)
    y1 = min(y1, y)
    x2 = max(x2, x)
    y2 = max(y2, y)

  overlay_width = x2 - x1
  overlay_height = y2 - y1
  x1, y1, x2, y2 = rect1.wrapper

  return (overlay_width / (x2 - x1) + overlay_height / (y2 - y1)) / 2.0