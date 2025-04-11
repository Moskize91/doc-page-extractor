import os
import requests

from munch import Munch
from math import ceil
from pix2tex.cli import LatexOCR
from PIL.Image import Image
from PIL.ImageOps import expand


class LaTeX:
  def __init__(self, model_path: str):
    self._model_path: str = model_path
    self._model: LatexOCR | None = None

  def extract(self, image: Image) -> str | None:
    image = self._expand_image(image, 0.1) # 添加边缘提高识别准确率
    return self._get_model()(image)

  def _get_model(self) -> LatexOCR:
    if self._model is None:
      if not os.path.exists(self._model_path):
        self._download_model()

      self._model = LatexOCR(Munch({
        "config": os.path.join("settings", "config.yaml"),
        "checkpoint": os.path.join(self._model_path, "weights.pth"),
        "no_cuda": True,
        "no_resize": False,
      }))
    return self._model

  # from https://github.com/lukas-blecher/LaTeX-OCR/blob/5c1ac929bd19a7ecf86d5fb8d94771c8969fcb80/pix2tex/model/checkpoints/get_latest_checkpoint.py#L37-L45
  def _download_model(self):
    os.makedirs(self._model_path, exist_ok=True)
    tag = "v0.0.1"
    files: list[tuple[str, str]] = (
      ("weights.pth", f"https://github.com/lukas-blecher/LaTeX-OCR/releases/download/{tag}/weights.pth"),
      ("image_resizer.pth", f"https://github.com/lukas-blecher/LaTeX-OCR/releases/download/{tag}/image_resizer.pth")
    )
    for file_name, url in files:
      file_path = os.path.join(self._model_path, file_name)
      try:
        with open(file_path, "wb") as file:
          response = requests.get(url, stream=True, timeout=15)
          response.raise_for_status()
          for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # 过滤掉保持连接的新块
              file.write(chunk)
          file.flush()

      except BaseException as e:
        if os.path.exists(file_path):
          os.remove(file_path)
        raise e

  def _expand_image(self, image: Image, percent: float):
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