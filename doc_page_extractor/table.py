import os
import torch

from typing import Literal, Any
from PIL.Image import Image
from .types import TableLayoutParsedFormat, GetModelDir
from .utils import expand_image


OutputFormat = Literal["latex", "markdown", "html"]

class Table:
  def __init__(
      self,
      device: Literal["cpu", "cuda"],
      get_model_dir: GetModelDir,
    ):
    self._model: Any | None = None
    self._model_path: str = get_model_dir()
    self._ban: bool = False
    if device == "cpu" or not torch.cuda.is_available():
      self._ban = True

  def predict(self, image: Image, format: TableLayoutParsedFormat) -> str | None:
    if self._ban:
      print("CUDA is not available. You cannot parse table from image.")
      return None

    output_format: str
    if format == TableLayoutParsedFormat.LATEX:
      output_format = "latex"
    elif format == TableLayoutParsedFormat.MARKDOWN:
      output_format = "markdown"
    elif format == TableLayoutParsedFormat.HTML:
      output_format = "html"
    else:
      raise ValueError(f"Table format {format} is not supported.")

    image = expand_image(image, 0.1)
    model = self._get_model()

    with torch.no_grad():
      results = model([image], output_format=output_format)

    if len(results) == 0:
      return None

    return results[0]

  def _get_model(self):
    if self._model is None:
      local_files_only: bool
      if os.path.exists(self._model_path):
        local_files_only = True
      else:
        local_files_only = False
        os.makedirs(self._model_path)

      from .struct_eqtable import build_model
      model = build_model(
        model_ckpt=self._model_path,
        max_new_tokens=1024,
        max_time=30,
        lmdeploy=False,
        flash_attn=True,
        batch_size=1,
        local_files_only=local_files_only,
      )
      self._model = model.cuda()
    return self._model