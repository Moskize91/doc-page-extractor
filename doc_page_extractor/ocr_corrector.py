import numpy as np

from typing import Iterable
from shapely.geometry import Polygon
from PIL.Image import new, Image, Resampling
from .types import Layout, OCRFragment
from .ocr import OCR
from .overlap import overlap_rate
from .rectangle import Point, Rectangle


_MIN_RATE = 0.5

def correct_fragments(ocr: OCR, source: Image, layout: Layout):
  x1, y1, x2, y2 = layout.rect.wrapper
  image: Image = source.crop((
    round(x1), round(y1),
    round(x2), round(y2),
  ))
  image, dx, dy, scale = _adjust_image(image)
  image_np = np.array(image)
  ocr_fragments = list(ocr.search_fragments(image_np))
  corrected_fragments: list[OCRFragment] = []

  for fragment in ocr_fragments:
    _apply_fragment(fragment.rect, layout, dx, dy, scale)

  matched_fragments, not_matched_fragments = _match_fragments(
    zone_rect=layout.rect,
    fragments1=layout.fragments,
    fragments2=ocr_fragments,
  )
  for fragment1, fragment2 in matched_fragments:
    if fragment1.rank > fragment2.rank:
      corrected_fragments.append(fragment1)
    else:
      corrected_fragments.append(fragment2)

  corrected_fragments.extend(not_matched_fragments)
  layout.fragments = corrected_fragments

def _adjust_image(image: Image) -> tuple[Image, int, int, float]:
  # after testing, adding white borders to images can reduce
  # the possibility of some text not being recognized
  border_size: int = 50
  adjusted_size: int = 1024 - 2 * border_size
  width, height = image.size
  core_width = float(max(adjusted_size, width))
  core_height = float(max(adjusted_size, height))

  scale_x = core_width / width
  scale_y = core_height / height
  scale = min(scale_x, scale_y)
  adjusted_width = width * scale
  adjusted_height = height * scale

  dx = (core_width - adjusted_width) / 2.0
  dy = (core_height - adjusted_height) / 2.0
  dx = round(dx) + border_size
  dy = round(dy) + border_size

  if scale != 1.0:
    width = round(width * scale)
    height = round(height * scale)
    image = image.resize((width, height), Resampling.BICUBIC)

  width = round(core_width) + 2 * border_size
  height = round(core_height) + 2 * border_size
  new_image = new("RGB", (width, height), (255, 255, 255))
  new_image.paste(image, (dx, dy))

  return new_image, dx, dy, scale

def _apply_fragment(rect: Rectangle, layout: Layout, dx: int, dy: int, scale: float):
  rect.lt = _apply_point(rect.lt, layout, dx, dy, scale)
  rect.lb = _apply_point(rect.lb, layout, dx, dy, scale)
  rect.rb = _apply_point(rect.rb, layout, dx, dy, scale)
  rect.rt = _apply_point(rect.rt, layout, dx, dy, scale)

def _apply_point(point: Point, layout: Layout, dx: int, dy: int, scale: float) -> Point:
  x, y = point
  x = (x - dx) / scale + layout.rect.lt[0]
  y = (y - dy) / scale + layout.rect.lt[1]
  return x, y

def _match_fragments(
    zone_rect: Rectangle,
    fragments1: Iterable[OCRFragment],
    fragments2: Iterable[OCRFragment],
  ) -> tuple[list[tuple[OCRFragment, OCRFragment]], list[OCRFragment]]:

  zone_polygon = Polygon(zone_rect)
  fragments2: list[OCRFragment] = list(fragments2)
  matched_fragments: list[tuple[OCRFragment, OCRFragment]] = []
  not_matched_fragments: list[OCRFragment] = []

  for fragment1 in fragments1:
    polygon1 = Polygon(fragment1.rect)
    polygon1 = zone_polygon.intersection(polygon1)
    if polygon1.is_empty:
      continue

    beast_j = -1
    beast_rate = 0.0

    for j, fragment2 in enumerate(fragments2):
      polygon2 = Polygon(fragment2.rect)
      rate = overlap_rate(polygon1, polygon2)
      if rate < _MIN_RATE:
        continue

      if rate > beast_rate:
        beast_j = j
        beast_rate = rate

    if beast_j != -1:
      matched_fragments.append((
        fragment1,
        fragments2[beast_j],
      ))
      del fragments2[beast_j]
    else:
      not_matched_fragments.append(fragment1)

  not_matched_fragments.extend(fragments2)
  return matched_fragments, not_matched_fragments