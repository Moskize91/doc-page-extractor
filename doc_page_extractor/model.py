from os import PathLike
from typing import runtime_checkable, Protocol
from pathlib import Path
from threading import Lock
from huggingface_hub import hf_hub_download, snapshot_download, try_to_load_from_cache


@runtime_checkable
class Model(Protocol):
  def get_onnx_ocr_path(self) -> Path:
    pass

  def get_yolo_path(self) -> Path:
    pass

  def get_layoutreader_path(self) -> Path:
    pass

  def get_struct_eqtable_path(self) -> Path:
    pass

  def get_latex_path(self) -> Path:
    pass

class HuggingfaceModel(Model):
  def __init__(self, model_cache_dir: PathLike):
    super().__init__()
    self._lock: Lock = Lock()
    self._model_cache_dir: Path = Path(model_cache_dir)

  def get_onnx_ocr_path(self) -> Path:
    return self._get_model_path(
      repo_id="moskize/OnnxOCR",
      filename=None,
      repo_type=None,
      is_snapshot=True
    )

  def get_yolo_path(self) -> Path:
    return self._get_model_path(
      repo_id="opendatalab/PDF-Extract-Kit-1.0",
      filename="models/Layout/YOLO/doclayout_yolo_ft.pt",
      repo_type=None,
      is_snapshot=False,
    )

  def get_layoutreader_path(self) -> Path:
    return self._get_model_path(
      repo_id="hantian/layoutreader",
      filename=None,
      repo_type=None,
      is_snapshot=True,
    )

  def get_struct_eqtable_path(self) -> Path:
    return self._get_model_path(
      repo_id="U4R/StructTable-InternVL2-1B",
      filename="model.safetensors",
      repo_type=None,
      is_snapshot=True,
    )

  def get_latex_path(self) -> Path:
    return self._get_model_path(
      repo_id="lukbl/LaTeX-OCR",
      filename="checkpoints/weights.pth",
      repo_type="space",
      is_snapshot=True,
    )

  def _get_model_path(
        self,
        repo_id: str,
        filename: str | None,
        repo_type: str | None,
        is_snapshot: bool,
      ) -> Path:
    with self._lock:
      cache_filename = "README.md"
      if filename is not None:
        cache_filename = filename
      model_path = try_to_load_from_cache(
        repo_id=repo_id,
        filename=cache_filename,
        repo_type=repo_type,
        cache_dir=self._model_cache_dir
      )
      if isinstance(model_path, str):
        if filename is None:
          model_path = Path(model_path).parent

      elif is_snapshot:
        model_path = snapshot_download(
          cache_dir=self._model_cache_dir,
          repo_id=repo_id,
        )
      else:
        model_path = hf_hub_download(
          cache_dir=self._model_cache_dir,
          repo_id=repo_id,
          filename=filename,
        )
      return Path(model_path)