import numpy as np

from dataclasses import dataclass
from PIL.Image import Image
from math import pi
from .types import OCRFragment, Layout
from .rotation import calculate_rotation, RotationAdjuster
from .rectangle import Rectangle


_TINY_ROTATION = 0.005 # below this angle, we consider the text is horizontal


@dataclass
class _RotationContext:
  to_origin: RotationAdjuster
  to_new: RotationAdjuster
  fragment_origin_rectangles: list[Rectangle]

class RawOptimizer:
  def __init__(
      self,
      raw: Image,
      adjust_points: bool,
    ):
    self._raw: Image = raw
    self._image: Image = raw
    self._adjust_points: bool = adjust_points
    self._fragments: list[OCRFragment]
    self._rotation: float = 0.0
    self._rotation_context: _RotationContext | None  = None

  @property
  def image(self) -> Image:
    return self._image

  @property
  def adjusted_image(self) -> Image | None:
    if self._adjust_points and self._image != self._raw:
      return self._image

  @property
  def rotation(self) -> float:
    return self._rotation

  @property
  def image_np(self) -> np.ndarray:
    return np.array(self._raw)

  def receive_raw_fragments(self, fragments: list[OCRFragment]):
    self._fragments = fragments
    self._rotation = calculate_rotation(fragments)

    if abs(self._rotation) < _TINY_ROTATION:
      return

    origin_size = self._raw.size
    self._image = self._raw.rotate(
      angle=self._rotation * 180 / pi,
      fillcolor=(255, 255, 255),
      expand=True,
    )
    self._rotation_context = _RotationContext(
      fragment_origin_rectangles=[f.rect for f in fragments],
      to_origin=RotationAdjuster(
        origin_size=origin_size,
        new_size=self._image.size,
        rotation=self._rotation,
        to_origin_coordinate=True,
      ),
      to_new=RotationAdjuster(
        origin_size=origin_size,
        new_size=self._image.size,
        rotation=self._rotation,
        to_origin_coordinate=False,
      ),
    )
    adjuster = self._rotation_context.to_new

    for fragment in fragments:
      rect = fragment.rect
      fragment.rect = Rectangle(
        lt=adjuster.adjust(rect.lt),
        rt=adjuster.adjust(rect.rt),
        lb=adjuster.adjust(rect.lb),
        rb=adjuster.adjust(rect.rb),
      )

  def receive_raw_layouts(self, layouts: list[Layout]):
    if self._adjust_points or self._rotation_context is None:
      return

    for fragment, origin_rect in zip(self._fragments, self._rotation_context.fragment_origin_rectangles):
      fragment.rect = origin_rect

    adjuster = self._rotation_context.to_origin

    for layout in layouts:
      layout.rect = Rectangle(
        lt=adjuster.adjust(layout.rect.lt),
        rt=adjuster.adjust(layout.rect.rt),
        lb=adjuster.adjust(layout.rect.lb),
        rb=adjuster.adjust(layout.rect.rb),
      )
