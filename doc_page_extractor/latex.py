import os
import torch

from munch import Munch
from pix2tex.cli import LatexOCR
from PIL.Image import Image
from .utils import expand_image
from .types import GetModelDir

class LaTeX:
  def __init__(self, get_model_dir: GetModelDir):
    self._model_path: str = get_model_dir()
    self._model: LatexOCR | None = None

  def extract(self, image: Image) -> str | None:
    image = expand_image(image, 0.1) # 添加边缘提高识别准确率
    model = self._get_model()
    with torch.no_grad():
      return model(image)

  def _get_model(self) -> LatexOCR:
    if self._model is None:
      self._model = LatexOCR(Munch({
        "config": os.path.join("settings", "config.yaml"),
        "checkpoint": os.path.join(self._model_path, "checkpoints", "weights.pth"),
        "no_cuda": True,
        "no_resize": False,
      }))
    return self._model
