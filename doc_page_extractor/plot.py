from typing import Iterable
from PIL import ImageDraw
from PIL.ImageFont import load_default, FreeTypeFont
from PIL.Image import Image
from .types import Layout, LayoutClass
from .rectangle import Point

_FRAGMENT_COLOR = (0x49, 0xCF, 0xCB) # Light Green
_Color = tuple[int, int, int]

def plot(image: Image, layouts: Iterable[Layout]) -> None:
  layout_font = load_default(size=35)
  fragment_font = load_default(size=25)
  draw = ImageDraw.Draw(image, mode="RGBA")

  def _draw_number(position: Point, number: int, font: FreeTypeFont, bold: bool, color: _Color) -> None:
    nonlocal draw
    x, y = position
    text = str(object=number)
    width = len(text) * font.size
    offset = round(font.size * 0.15)

    for dx, dy in _generate_delta(bold):
      draw.text(
        xy=(x + dx - width - offset, y + dy),
        text=text,
        font=font,
        fill=color,
      )

  for layout in layouts:
    draw.polygon(
      xy=[p for p in layout.rect],
      outline=_layout_color(layout),
      width=5,
    )

  for layout in layouts:
    for fragment in layout.fragments:
      draw.polygon(
        xy=[p for p in fragment.rect],
        outline=_FRAGMENT_COLOR,
        width=3,
      )
      _draw_number(
        position=fragment.rect.lt,
        number=fragment.order + 1,
        font=fragment_font,
        bold=False,
        color=_FRAGMENT_COLOR,
      )

  for i, layout in enumerate(layouts):
    _draw_number(
      position=layout.rect.lt,
      number=i + 1,
      font=layout_font,
      bold=True,
      color=_layout_color(layout),
    )

def _generate_delta(bold: bool):
  if bold:
    for dx in range(-1, 2):
      for dy in range(-1, 2):
        yield dx, dy
  else:
    yield 0, 0

def _layout_color(layout: Layout) -> _Color:
  cls = layout.cls
  if cls == LayoutClass.TITLE:
    return (0x0A, 0x12, 0x2C) # Dark
  elif cls == LayoutClass.PLAIN_TEXT:
    return (0x3C, 0x67, 0x90) # Blue
  elif cls == LayoutClass.ABANDON:
    return (0xC0, 0xBB, 0xA9) # Gray
  elif cls == LayoutClass.FIGURE:
    return (0x5B, 0x91, 0x3C) # Dark Green
  elif cls == LayoutClass.FIGURE_CAPTION:
    return (0x77, 0xB3, 0x54) # Green
  elif cls == LayoutClass.TABLE:
    return (0x44, 0x17, 0x52) # Dark Purple
  elif cls == LayoutClass.TABLE_CAPTION:
    return (0x81, 0x75, 0xA0) # Purple
  elif cls == LayoutClass.TABLE_FOOTNOTE:
    return (0xEF, 0xB6, 0xC9) # Pink Purple
  elif cls == LayoutClass.ISOLATE_FORMULA:
    return (0xFA, 0x38, 0x27) # Red
  elif cls == LayoutClass.FORMULA_CAPTION:
    return (0xFF, 0x9D, 0x24) # Orange