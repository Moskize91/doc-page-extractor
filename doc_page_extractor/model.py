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
      filename="README.md",
      repo_type=None,
      is_snapshot=True,
      wanna_dir_path=True,
    )

  def get_yolo_path(self) -> Path:
    return self._get_model_path(
      repo_id="opendatalab/PDF-Extract-Kit-1.0",
      filename="models/Layout/YOLO/doclayout_yolo_ft.pt",
      repo_type=None,
      is_snapshot=False,
      wanna_dir_path=False,
    )

  def get_layoutreader_path(self) -> Path:
    return self._get_model_path(
      repo_id="hantian/layoutreader",
      filename="model.safetensors",
      repo_type=None,
      is_snapshot=True,
      wanna_dir_path=True,
    )

  def get_struct_eqtable_path(self) -> Path:
    return self._get_model_path(
      repo_id="U4R/StructTable-InternVL2-1B",
      filename="model.safetensors",
      repo_type=None,
      is_snapshot=True,
      wanna_dir_path=True,
    )

  def get_latex_path(self) -> Path:
    return self._get_model_path(
      repo_id="lukbl/LaTeX-OCR",
      filename="checkpoints/weights.pth",
      repo_type="space",
      is_snapshot=True,
      wanna_dir_path=True,
    )

  def _get_model_path(
        self,
        repo_id: str,
        filename: str,
        repo_type: str | None,
        is_snapshot: bool,
        wanna_dir_path: bool,
      ) -> Path:

    with self._lock:
      model_path = try_to_load_from_cache(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        cache_dir=self._model_cache_dir
      )
      if isinstance(model_path, str):
        model_path = Path(model_path)
        if wanna_dir_path:
          for _ in Path(filename).parts:
            model_path = model_path.parent

      else:
        if is_snapshot:
          model_path = snapshot_download(
            cache_dir=self._model_cache_dir,
            repo_id=repo_id,
            repo_type=repo_type,
          )
        else:
          model_path = hf_hub_download(
            cache_dir=self._model_cache_dir,
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
          )
        model_path = Path(model_path)

      return model_path