import os
import re

from math import ceil
from PIL.Image import Image
from PIL.ImageOps import expand


def ensure_dir(path: str) -> str:
  path = os.path.abspath(path)
  os.makedirs(path, exist_ok=True)
  return path

def is_space_text(text: str) -> bool:
  return re.match(r"^\s*$", text)

def expand_image(image: Image, percent: float):
  width, height = image.size
  border_width = ceil(width * percent)
  border_height = ceil(height * percent)
  fill_color: tuple[int, ...]

  if image.mode == "RGBA":
    fill_color = (255, 255, 255, 255)
  else:
    fill_color = (255, 255, 255)

  return expand(
    image=image,
    border=(border_width, border_height),
    fill=fill_color,
  )