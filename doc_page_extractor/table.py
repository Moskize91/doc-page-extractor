from typing import Literal
from PIL import Image
from .struct_eqtable import build_model


OutputFormat = Literal["latex", "markdown", "html"]

class TableParsingStructEqTable:
  def __init__(self):
    self._model = build_model(
      model_ckpt="U4R/StructTable-InternVL2-1B",
      max_new_tokens=1024,
      max_time=30,
      lmdeploy=False,
      flash_attn=True,
      batch_size=1,
    )

  def predict(self, images: list[str], output_format: OutputFormat = "latex"):
    load_images = [Image.open(image_path) for image_path in images]
    if output_format not in ("latex", "markdown", "html"):
      raise ValueError(f"Output format {output_format} is not supported.")

    results = self._model(
      load_images,
      output_format=output_format
    )
    return results
