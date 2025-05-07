from huggingface_hub import hf_hub_download, snapshot_download, try_to_load_from_cache
import os

from .types import ModelsDownloader

class HugginfaceModelsDownloader(ModelsDownloader):
  def __init__(self, model_dir_path: str | None):
    self._model_dir_path: str | None = model_dir_path

  def onnx_ocr(self) -> str:
    repo_path = try_to_load_from_cache(repo_id="moskize/OnnxOCR", filename="README.md")
    if isinstance(repo_path, str):
      return os.path.dirname(repo_path)
    else:
        print("Downloading OCR model...")
        return snapshot_download(
          cache_dir=self._model_dir_path,
          repo_id="moskize/OnnxOCR",
        )

  def yolo(self) -> str:
    yolo_file_path = try_to_load_from_cache(repo_id="opendatalab/PDF-Extract-Kit-1.0", filename="models/Layout/YOLO/doclayout_yolo_ft.pt")
    if isinstance(yolo_file_path, str):
      return yolo_file_path
    else:
      print("Downloading YOLO model...")
      return hf_hub_download(
        cache_dir=self._model_dir_path,
        repo_id="opendatalab/PDF-Extract-Kit-1.0",
        filename="models/Layout/YOLO/doclayout_yolo_ft.pt",
      )

  def layoutreader(self) -> str:
    repo_path = try_to_load_from_cache(repo_id="hantian/layoutreader", filename="model.safetensors")
    if isinstance(repo_path, str):
      return os.path.dirname(repo_path)
    else:
      print("Downloading LayoutReader model...")
      return snapshot_download(
        cache_dir=self._model_dir_path,
        repo_id="hantian/layoutreader",
      )

  def struct_eqtable(self) -> str:
    repo_path = try_to_load_from_cache(repo_id="U4R/StructTable-InternVL2-1B", filename="model.safetensors")
    if isinstance(repo_path, str):
      return os.path.dirname(repo_path)
    else:
      print("Downloading StructEqTable model...")
      return snapshot_download(
        cache_dir=self._model_dir_path,
        repo_id="U4R/StructTable-InternVL2-1B",
      )

  def latex(self):
    repo_path = try_to_load_from_cache(repo_id="lukbl/LaTeX-OCR", filename="checkpoints/weights.pth", repo_type="space")
    if isinstance(repo_path, str):
      return os.path.dirname(os.path.dirname(repo_path))
    else:
      print("Downloading LaTeX model...")
      return snapshot_download(
        cache_dir=self._model_dir_path,
        repo_type="space",
        repo_id="lukbl/LaTeX-OCR",
      )
