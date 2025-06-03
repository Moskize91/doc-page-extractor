import torch

from typing import Literal, Any
from PIL.Image import Image
from .types import TableLayoutParsedFormat
from .model import Model
from .utils import expand_image


OutputFormat = Literal["latex", "markdown", "html"]

class Table:
  def __init__(self, device: Literal["cpu", "cuda"], model: Model) -> None:
    self._model: Model = model
    self._table_model: Any | None = None
    self._ban: bool = False
    if device == "cpu":
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

  def _get_model(self) -> Any:
    if self._table_model is None:
      from .struct_eqtable import build_model
      model_path = self._model.get_struct_eqtable_path()
      table_model = build_model(
        model_ckpt=model_path,
        max_new_tokens=1024,
        max_time=30,
        lmdeploy=False,
        flash_attn=True,
        batch_size=1,
        local_files_only=False,
      )
      self._table_model = table_model.cuda()
    return self._table_model