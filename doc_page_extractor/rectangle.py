from typing import Generator
from dataclasses import dataclass
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
  def segments(self) -> Generator[tuple[Point, Point], None, None]:
    yield (self.lt, self.lb)
    yield (self.lb, self.rb)
    yield (self.rb, self.rt)
    yield (self.rt, self.lt)

  @property
  def area(self) -> float:
    return Polygon(self).area

def intersection_area(rect1: Rectangle, rect2: Rectangle) -> float:
  poly1 = Polygon(rect1)
  poly2 = Polygon(rect2)
  intersection = poly1.intersection(poly2)
  if intersection.is_empty:
    return 0.0
  return intersection.area