from math import pi, atan2, sqrt, sin, cos
from .types import OCRFragment
from .rectangle import Point, Rectangle


class RotationAdjuster:
  def __init__(
      self,
      origin_size: tuple[int, int],
      new_size: tuple[int, int],
      rotation: float,
      to_origin_coordinate: bool,
    ):
    from_size: tuple[int, int]
    to_size: tuple[int, int]
    if to_origin_coordinate:
      from_size = new_size
      to_size = origin_size
    else:
      from_size = origin_size
      to_size = new_size
      rotation = -rotation

    self._rotation: float = rotation
    self._center_offset: tuple[float, float] = (
      - from_size[0] / 2.0,
      - from_size[1] / 2.0,
    )
    self._new_offset: tuple[float, float] = (
      to_size[0] / 2.0,
      to_size[1] / 2.0,
    )

  def adjust(self, point: Point) -> Point:
    x, y = point
    x += self._center_offset[0]
    y += self._center_offset[1]

    if x != 0.0 or y != 0.0:
      radius = sqrt(x*x + y*y)
      angle = atan2(y, x) + self._rotation
      x = radius * cos(angle)
      y = radius * sin(angle)

    x += self._new_offset[0]
    y += self._new_offset[1]

    return x, y

# to [0, pi)
def normal_vertical_rotation(rotation: float) -> float:
  while rotation >= 2 * pi:
    rotation -= 2 * pi
  while rotation <= - 2 * pi:
    rotation += 2 * pi
  if rotation < 0.0:
    rotation += pi
  return rotation

def calculate_rotation(fragments: list[OCRFragment]):
  horizontal_rotations: list[float] = []
  vertical_rotations: list[float] = []

  for fragment in fragments:
    result = _rotation_with(fragment.rect)
    if result is not None:
      horizontal_rotations.extend(result[0])
      vertical_rotations.extend(result[1])

  if len(horizontal_rotations) == 0 or len(vertical_rotations) == 0:
    return 0.0

  horizontal_rotations = _normal_horizontal_rotations(horizontal_rotations)
  horizontal_rotation = _find_median(horizontal_rotations)
  vertical_rotation = _find_median(vertical_rotations)

  return (vertical_rotation - 0.5 * pi + horizontal_rotation) / 2.0

# @return horizontal [-pi/2, pi/2), vertical [0, pi)
def calculate_rotation_with_rect(rect: Rectangle) -> tuple[float, float]:
  result = _rotation_with(rect)
  if result is None:
    return 0.0, 0.5 * pi

  horizontal_rotations, vertical_rotations = result
  horizontal_rotations = _normal_horizontal_rotations(horizontal_rotations)
  horizontal_rotation = _find_mean(horizontal_rotations)
  vertical_rotation = _find_mean(vertical_rotations)

  return horizontal_rotation, vertical_rotation

def _rotation_with(rect: Rectangle):
  rotations0: list[float] = []
  rotations1: list[float] = []

  for i, (p1, p2) in enumerate(rect.segments):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0.0 and dy == 0.0:
      return None
    rotation: float = atan2(dy, dx)
    if rotation < 0.0:
      rotation += pi
    if i % 2 == 0:
      rotations0.append(rotation)
    else:
      rotations1.append(rotation)

  if _is_vertical(rotations0):
    return rotations1, rotations0
  else:
    return rotations0, rotations1

# [0, pi) --> [-pi/2, pi/2)
def _normal_horizontal_rotations(rotations: list[float]) -> list[float]:
  for i, rotation in enumerate(rotations):
    if rotation > 0.5 * pi:
      rotations[i] = rotation - pi
  return rotations

def _find_median(rotations: list[float]):
  rotations.sort()
  n = len(rotations)

  if n % 2 == 1:
    return rotations[n // 2]
  else:
    mid1 = rotations[n // 2 - 1]
    mid2 = rotations[n // 2]
    return (mid1 + mid2) / 2

def _find_mean(rotations: list[float]) -> float:
  if len(rotations) == 0:
    return 0.0
  return sum(rotations) / len(rotations)

# rotation is in [0, pi)
def _is_vertical(rotations: list[float]):
  horizontal_count: int = 0
  vertical_count: int = 0
  horizontal_delta: float = 0.0
  vertical_delta: float = 0.0

  for rotation in rotations:
    if rotation < 0.25 * pi: # [0, pi/4)
      horizontal_count += 1
      horizontal_delta += rotation
    elif rotation < 0.75 * pi: # [pi/4, 3pi/4)
      vertical_count += 1
      vertical_delta += abs(rotation - 0.5 * pi)
    else: # [3pi/4, pi)
      horizontal_count += 1
      horizontal_delta += pi - rotation

  if vertical_count == horizontal_delta:
    return vertical_delta < horizontal_delta
  else:
    return vertical_count > horizontal_count
