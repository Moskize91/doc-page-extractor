from typing import Iterable
from shapely.geometry import Polygon
from .types import Layout, OCRFragment
from .rectangle import Rectangle


_MIN_RATE = 0.5

def correct_ocr(
    layout: Layout,
    layout_fragments: Iterable[OCRFragment],
    cropped_fragments: Iterable[OCRFragment],
  ) -> list[OCRFragment]:

  corrected_fragments: list[OCRFragment] = []
  matched_fragments, not_matched_fragments = _match_fragments(
    zone_rect=layout.rect,
    fragments1=[_relative_layout(layout, f, False) for f in layout_fragments],
    fragments2=cropped_fragments,
  )
  for fragment1, fragment2 in matched_fragments:
    if fragment1.rank > fragment2.rank:
      corrected_fragments.append(fragment1)
    else:
      corrected_fragments.append(fragment2)

  corrected_fragments.extend(not_matched_fragments)
  corrected_fragments = [_relative_layout(layout, f, True) for f in corrected_fragments]

  return corrected_fragments

def _relative_layout(layout: Layout, fragment: OCRFragment, addition: bool) -> OCRFragment:
  dx, dy = layout.rect.lt
  if not addition:
    dx, dy = -dx, -dy

  rect = fragment.rect
  rect = Rectangle(
    lt=(rect.lt[0] + dx, rect.lt[1] + dy),
    rt=(rect.rt[0] + dx, rect.rt[1] + dy),
    lb=(rect.lb[0] + dx, rect.lb[1] + dy),
    rb=(rect.rb[0] + dx, rect.rb[1] + dy),
  )
  return OCRFragment(
    order=fragment.order,
    text=fragment.text,
    rank=fragment.rank,
    rect=rect,
  )

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
      intersection = polygon2.intersection(polygon1)
      intersection_area = 0.0
      if not intersection.is_empty:
        intersection_area = intersection.area
      rate = intersection_area / polygon1.area
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