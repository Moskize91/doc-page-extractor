from typing import Iterable
from PIL import ImageDraw
from PIL.ImageFile import ImageFile
from .types import Layout

def plot(image: ImageFile, layouts: Iterable[Layout]):
  draw = ImageDraw.Draw(image, mode="RGBA")
  for layout in layouts:
    draw.polygon([p for p in layout.rect], outline=(255, 0, 0), width=3)
    for fragments in layout.fragments:
      draw.polygon([p for p in fragments.rect], outline=(0, 255, 0), width=1)