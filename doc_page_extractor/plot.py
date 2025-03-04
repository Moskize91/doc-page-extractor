from typing import Iterable
from PIL import ImageDraw
from PIL.Image import Image
from .types import Layout, LayoutClass

_FRAGMENT_COLOR = (0x49, 0xCF, 0xCB) # Light Green

def plot(image: Image, layouts: Iterable[Layout]):
  draw = ImageDraw.Draw(image, mode="RGBA")
  for layout in layouts:
    draw.polygon([p for p in layout.rect], outline=_layout_color(layout), width=5)

  for layout in layouts:
    for fragments in layout.fragments:
      draw.polygon([p for p in fragments.rect], outline=_FRAGMENT_COLOR, width=3)

def _layout_color(layout: Layout) -> tuple[int, int, int]:
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