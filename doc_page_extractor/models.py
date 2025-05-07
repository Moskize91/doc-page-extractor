import os

from logging import Logger
from huggingface_hub import hf_hub_download, snapshot_download, try_to_load_from_cache
from .types import ModelsDownloader

class HuggingfaceModelsDownloader(ModelsDownloader):
  def __init__(
      self,
      logger: Logger,
      model_dir_path: str | None
    ):
    self._logger = logger
    self._model_dir_path: str | None = model_dir_path

  def onnx_ocr(self) -> str:
    repo_path = try_to_load_from_cache(
      repo_id="moskize/OnnxOCR",
      filename="README.md",
      cache_dir=self._model_dir_path
    )
    if isinstance(repo_path, str):
      return os.path.dirname(repo_path)
    else:
      self._logger.info("Downloading OCR model...")
      return snapshot_download(
        cache_dir=self._model_dir_path,
        repo_id="moskize/OnnxOCR",
      )

  def yolo(self) -> str:
    yolo_file_path = try_to_load_from_cache(
      repo_id="opendatalab/PDF-Extract-Kit-1.0",
      filename="models/Layout/YOLO/doclayout_yolo_ft.pt",
      cache_dir=self._model_dir_path
    )
    if isinstance(yolo_file_path, str):
      return yolo_file_path
    else:
      self._logger.info("Downloading YOLO model...")
      return hf_hub_download(
        cache_dir=self._model_dir_path,
        repo_id="opendatalab/PDF-Extract-Kit-1.0",
        filename="models/Layout/YOLO/doclayout_yolo_ft.pt",
      )

  def layoutreader(self) -> str:
    repo_path = try_to_load_from_cache(
      repo_id="hantian/layoutreader",
      filename="model.safetensors",
      cache_dir=self._model_dir_path
    )
    if isinstance(repo_path, str):
      return os.path.dirname(repo_path)
    else:
      self._logger.info("Downloading LayoutReader model...")
      return snapshot_download(
        cache_dir=self._model_dir_path,
        repo_id="hantian/layoutreader",
      )

  def struct_eqtable(self) -> str:
    repo_path = try_to_load_from_cache(
      repo_id="U4R/StructTable-InternVL2-1B",
      filename="model.safetensors",
      cache_dir=self._model_dir_path
    )
    if isinstance(repo_path, str):
      return os.path.dirname(repo_path)
    else:
      self._logger.info("Downloading StructEqTable model...")
      return snapshot_download(
        cache_dir=self._model_dir_path,
        repo_id="U4R/StructTable-InternVL2-1B",
      )

  def latex(self):
    repo_path = try_to_load_from_cache(
      repo_id="lukbl/LaTeX-OCR",
      filename="checkpoints/weights.pth",
      repo_type="space",
      cache_dir=self._model_dir_path
    )
    if isinstance(repo_path, str):
      return os.path.dirname(os.path.dirname(repo_path))
    else:
      self._logger.info("Downloading LaTeX model...")
      return snapshot_download(
        cache_dir=self._model_dir_path,
        repo_type="space",
        repo_id="lukbl/LaTeX-OCR",
      )
