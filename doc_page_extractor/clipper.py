import numpy as np

from math import ceil, sin, cos, sqrt
from PIL.Image import Image, Transform
from .types import Layout, ExtractedResult
from .rectangle import Point, Rectangle
from .rotation import calculate_rotation_with_rect


def clip(extracted_result: ExtractedResult, layout: Layout) -> Image:
  image: Image
  if extracted_result.adjusted_image is None:
    image = extracted_result.extracted_image
  else:
    image = extracted_result.adjusted_image

  return clip_from_image(image, layout.rect)

def clip_from_image(image: Image, rect: Rectangle) -> Image:
  horizontal_rotation, vertical_rotation = calculate_rotation_with_rect(rect)
  image = image.copy()
  matrix_move = np.array(_get_move_matrix(rect.lt[0], rect.lt[1])).reshape(3, 3)
  matrix_rotate = np.array(_get_rotate_matrix(-horizontal_rotation)).reshape(3, 3)
  matrix = np.dot(matrix_move, matrix_rotate)
  size = _size_with_rect(rect)
  affine_matrix = _to_pillow_matrix(matrix)

  return image.transform(size, Transform.AFFINE, affine_matrix)

def _size_with_rect(rect: Rectangle):
  n = 4 / 2
  width: float = 0.0
  height: float = 0.0
  for i, (p1, p2) in enumerate(rect.segments):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    distance = sqrt(dx*dx + dy*dy)
    if i % 2 == 0:
      height += distance
    else:
      width += distance
  return ceil(width / n), ceil(height / n)

def _warp_rectangle(origin_rect: Rectangle, matrix: np.array):
  left: float = float("inf")
  right: float = float("-inf")
  top: float = float("inf")
  bottom: float = float("-inf")
  for point in origin_rect:
    x, y = _transform_point(matrix, point)
    print("point:", point, "transformed:", (x, y))
    left = min(left, x)
    right = max(right, x)
    top = min(top, y)
    bottom = max(bottom, y)

  width: int = ceil(right - left)
  height: int = ceil(bottom - top)

  return (left, top), (width, height)

def _to_pillow_matrix(matrix: np.array):
  return (
    matrix[0][0], matrix[0][1], matrix[0][2],
    matrix[1][0], matrix[1][1], matrix[1][2],
  )

def _transform_point(matrix: np.array, point: Point) -> Point:
  matrix_vector = np.array((point[0], point[1], 1.0)).reshape(3, 1)
  matrix_result = np.dot(matrix, matrix_vector)
  x = matrix_result[0][0]
  y = matrix_result[1][0]
  z = matrix_result[2][0]
  return x / z, y / z

def _get_move_matrix(dx: float, dy: float):
  return (
    1.0, 0.0, dx,
    0.0, 1.0, dy,
    0.0, 0.0, 1.0,
  )

def _get_rotate_matrix(rotation: float):
  return (
    cos(rotation),  sin(rotation),  0.0,
    -sin(rotation), cos(rotation),  0.0,
    0.0,            0.0,            1.0
  )