import os
import torch

from munch import Munch
from pix2tex.cli import LatexOCR
from PIL.Image import Image
from typing import Literal
from .utils import expand_image
from .model import Model


class LaTeX:
  def __init__(self, device: Literal["cpu", "cuda"], model: Model) -> None:
    self._model: Model = model
    self._latex_model: LatexOCR | None = None
    self._device: Literal["cpu", "cuda"] = device

  def extract(self, image: Image) -> str | None:
    image = expand_image(image, 0.1) # 添加边缘提高识别准确率
    model = self._get_model()
    with torch.no_grad():
      return model(image)

  def _get_model(self) -> LatexOCR:
    if self._latex_model is None:
      model_path = self._model.get_latex_path()
      self._latex_model = LatexOCR(Munch({
        "config": os.path.join("settings", "config.yaml"),
        "checkpoint": os.path.join(model_path, "checkpoints", "weights.pth"),
        "no_cuda": self._device == "cpu",
        "no_resize": False,
      }))
    return self._latex_model
