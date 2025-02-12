from typing import Iterable
from PIL import ImageDraw
from PIL.Image import Image
from .types import Layout

def plot(image: Image, layouts: Iterable[Layout]):
  draw = ImageDraw.Draw(image, mode="RGBA")
  for layout in layouts:
    draw.polygon([p for p in layout.rect], outline=(255, 0, 0), width=3)
    for fragments in layout.fragments:
      draw.polygon([p for p in fragments.rect], outline=(0, 255, 0), width=1)