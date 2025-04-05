import numpy as np

from math import pi, ceil, sin, cos, sqrt
from PIL.Image import Image, Transform
from .types import Layout, ExtractedResult
from .rectangle import Rectangle
from .rotation import calculate_rotation_with_rect, normal_vertical_rotation


def clip(
    extracted_result: ExtractedResult,
    layout: Layout,
    wrapped_width: float = 0.0,
    wrapped_height: float = 0.0,
  ) -> Image:
  image: Image
  if extracted_result.adjusted_image is None:
    image = extracted_result.extracted_image
  else:
    image = extracted_result.adjusted_image

  return clip_from_image(
    image, layout.rect,
    wrapped_width, wrapped_height,
  )

def clip_from_image(
    image: Image,
    rect: Rectangle,
    wrapped_width: float = 0.0,
    wrapped_height: float = 0.0,
  ) -> Image:
  horizontal_rotation, vertical_rotation = calculate_rotation_with_rect(rect)
  image = image.copy()
  matrix_move = np.array(_get_move_matrix(rect.lt[0], rect.lt[1])).reshape(3, 3)
  matrix_rotate = np.array(_get_rotate_matrix(-horizontal_rotation)).reshape(3, 3)
  matrix = np.dot(matrix_move, matrix_rotate)

  y_axis_rotation = normal_vertical_rotation(vertical_rotation - horizontal_rotation)

  if abs(y_axis_rotation - 0.25 * pi) > 0.0:
    x = cos(y_axis_rotation)
    y = sin(y_axis_rotation)
    matrix_shear = np.array(_get_shear_matrix(x, y)).reshape(3, 3)
    matrix = np.dot(matrix, matrix_shear)

  width, height, max_width, max_height = _size_and_wrapper(rect)
  max_width += wrapped_width
  max_height += wrapped_height

  if max_width != width or max_height != height:
    dx = (max_width - width) / 2.0
    dy = (max_height - height) / 2.0
    matrix_move = np.array(_get_move_matrix(-dx, -dy)).reshape(3, 3)
    matrix = np.dot(matrix, matrix_move)

  return image.transform(
    size=(ceil(max_width), ceil(max_height)),
    method=Transform.AFFINE,
    data=_to_pillow_matrix(matrix),
  )

def _size_and_wrapper(rect: Rectangle):
  widths: list[float] = []
  heights: list[float] = []

  for i, (p1, p2) in enumerate(rect.segments):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    distance = sqrt(dx*dx + dy*dy)
    if i % 2 == 0:
      heights.append(distance)
    else:
      widths.append(distance)

  if len(widths) == 0 and len(heights) == 0:
    return 0.0, 0.0, 0.0, 0.0

  width: float = sum(widths) / len(widths)
  height: float = sum(heights) / len(heights)
  max_width: float = width
  max_height: float = height

  for width in widths:
    if width > max_width:
      max_width = width

  for height in heights:
    if height > max_height:
      max_height = height

  return width, height, max_width, max_height

def _to_pillow_matrix(matrix: np.array):
  return (
    matrix[0][0], matrix[0][1], matrix[0][2],
    matrix[1][0], matrix[1][1], matrix[1][2],
  )

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

def _get_shear_matrix(x0: float, y0: float):
  return (
    1.0, 0.0, 0.0,
    x0,  y0,  0.0,
    0.0, 0.0, 1.0,
  )