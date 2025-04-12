import torch

from typing import Literal, Any
from PIL.Image import Image
from .types import TableLayoutParsedFormat
from .utils import expand_image


OutputFormat = Literal["latex", "markdown", "html"]

class Table:
  def __init__(self, device: Literal["cpu", "cuda"]):
    self._model: Any | None = None
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
    results = self._get_model()([image], output_format=output_format)

    if len(results) == 0:
      return None

    return results[0]

  def _get_model(self):
    if self._model is None:
      from .struct_eqtable import build_model
      model = build_model(
        model_ckpt="U4R/StructTable-InternVL2-1B",
        max_new_tokens=1024,
        max_time=30,
        lmdeploy=False,
        flash_attn=True,
        batch_size=1,
      )
      self._model = model.cuda()
    return self._model