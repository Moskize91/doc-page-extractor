from math import atan2, pi, sqrt, sin, cos, atan
from .rectangle import Point
from .types import OCRFragment


class RotationAdjuster:
  def __init__(
      self,
      origin_size: tuple[int, int],
      new_size: tuple[int, int],
      rotation: float,
    ):
    self._rotation: float = rotation
    self._center_offset: tuple[float, float] = (
      - origin_size[0] / 2.0,
      - origin_size[1] / 2.0,
    )
    self._new_offset: tuple[float, float] = (
      (new_size[0] - origin_size[0]) / 2.0,
      (new_size[1] - origin_size[1]) / 2.0,
    )

  def adjust(self, point: Point) -> Point:
    x, y = self._center_offset
    x += point[0]
    y += point[1]

    if x != 0.0 or y != 0.0:
      radius = sqrt(x*x + y*y)
      angle = atan(y, x) - self._rotation
      x = radius * cos(angle)
      y = radius * sin(angle)

    x += self._new_offset[0]
    y += self._new_offset[1]

    return x, y

def calculate_rotation(fragments: list[OCRFragment]):
  rotations0: list[float] = []
  rotations1: list[float] = []

  for fragment in fragments:
    result = _rotation_with(fragment)
    if result is not None:
      rotation0, rotation1 = result
      rotations0.append(rotation0)
      rotations1.append(rotation1)

  if len(rotations0) == 0 or len(rotations1) == 0:
    return 0.0

  median0 = _find_median(rotations0)
  median1 = _find_median(rotations1)
  if abs(median0 - 0.5 * pi) < abs(median1 - 0.5 * pi):
    vertical = median0
    horizontal = median1
  else:
    vertical = median1
    horizontal = median0

  if horizontal > 0.5 * pi:
    horizontal -= pi

  return 0.5 * (vertical - 0.5 * pi + horizontal)

def _rotation_with(fragment: OCRFragment):
  rotations: list[float] = []
  for p1, p2 in _iter_fragment_segment(fragment):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0.0 and dy == 0.0:
      return None
    rotation: float = atan2(dy, dx)
    if rotation < 0.0:
      rotation += pi
    rotations.append(rotation)

  rotation0: float = 0.0
  rotation1: float = 0.0
  for i, rotation in enumerate(rotations):
    if i % 2 == 0:
      rotation0 += rotation
    else:
      rotation1 += rotation

  half_count = 2
  rotation0 /= half_count
  rotation1 /= half_count

  return min(rotation0, rotation1), max(rotation0, rotation1)

def _find_median(rotations: list[float]):
  rotations.sort()
  n = len(rotations)

  if n % 2 == 1:
    return rotations[n // 2]
  else:
    mid1 = rotations[n // 2 - 1]
    mid2 = rotations[n // 2]
    return (mid1 + mid2) / 2

def _iter_fragment_segment(fragment: OCRFragment):
  rect = fragment.rect
  yield (rect.lt, rect.lb)
  yield (rect.lb, rect.rb)
  yield (rect.rb, rect.rt)
  yield (rect.rt, rect.lt)