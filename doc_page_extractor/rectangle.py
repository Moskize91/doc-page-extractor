from typing import Generator
from dataclasses import dataclass
from math import sqrt
from shapely.geometry import Polygon


Point = tuple[float, float]

@dataclass
class Rectangle:
  lt: Point
  rt: Point
  lb: Point
  rb: Point

  def __iter__(self) -> Generator[Point, None, None]:
    yield self.lt
    yield self.lb
    yield self.rb
    yield self.rt

  @property
  def is_valid(self) -> bool:
    return Polygon(self).is_valid

  @property
  def segments(self) -> Generator[tuple[Point, Point], None, None]:
    yield (self.lt, self.lb)
    yield (self.lb, self.rb)
    yield (self.rb, self.rt)
    yield (self.rt, self.lt)

  @property
  def area(self) -> float:
    return Polygon(self).area

  @property
  def size(self) -> tuple[float, float]:
    width: float = 0.0
    height: float = 0.0
    for i, (p1, p2) in enumerate(self.segments):
      dx = p1[0] - p2[0]
      dy = p1[1] - p2[1]
      distance = sqrt(dx * dx + dy * dy)
      if i % 2 == 0:
        height += distance
      else:
        width += distance
    return width / 2, height / 2

  @property
  def wrapper(self) -> tuple[float, float, float, float]:
    x1: float = float("inf")
    y1: float = float("inf")
    x2: float = float("-inf")
    y2: float = float("-inf")
    for x, y in self:
      x1 = min(x1, x)
      y1 = min(y1, y)
      x2 = max(x2, x)
      y2 = max(y2, y)
    return x1, y1, x2, y2

def intersection_area(rect1: Rectangle, rect2: Rectangle) -> float:
  poly1 = Polygon(rect1)
  poly2 = Polygon(rect2)
  if not poly1.is_valid or not poly2.is_valid:
    return 0.0
  intersection = poly1.intersection(poly2)
  if intersection.is_empty:
    return 0.0
  return intersection.area