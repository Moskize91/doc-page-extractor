import os
import numpy as np

from typing import Literal, Any
from paddleocr import PaddleOCR
from .utils import ensure_dir


# https://github.com/PaddlePaddle/PaddleOCR/blob/2c0c4beb0606819735a16083cdebf652939c781a/paddleocr.py#L108-L157
PaddleLang = Literal["ch", "en", "korean", "japan", "chinese_cht", "ta", "te", "ka", "latin", "arabic", "cyrillic", "devanagari"]

# https://paddlepaddle.github.io/PaddleOCR/latest/quick_start.html#_2
class OCR:
  def __init__(self, device: Literal["cpu", "cuda"], model_dir_path: str):
    self._device: Literal["cpu", "cuda"] = device
    self._model_dir_path: str = model_dir_path
    self._ocr_and_lan: tuple[PaddleOCR, PaddleLang] | None = None

  def do(self, lang: PaddleLang, image: np.ndarray) -> list[Any]:
    # about img parameter to see
    # https://github.com/PaddlePaddle/PaddleOCR/blob/2c0c4beb0606819735a16083cdebf652939c781a/paddleocr.py#L582-L619
    return self._get_ocr(lang).ocr(img=image, cls=True)

  def _get_ocr(self, lang: PaddleLang) -> PaddleOCR:
    if self._ocr_and_lan is not None:
      ocr, origin_lang = self._ocr_and_lan
      if lang == origin_lang:
        return ocr

    ocr = PaddleOCR(
      lang=lang,
      use_angle_cls=True,
      use_gpu=self._device.startswith("cuda"),
      det_model_dir=ensure_dir(
        os.path.join(self._model_dir_path, "det"),
      ),
      rec_model_dir=ensure_dir(
        os.path.join(self._model_dir_path, "rec"),
      ),
      cls_model_dir=ensure_dir(
        os.path.join(self._model_dir_path, "cls"),
      ),
    )
    self._ocr_and_lan = (ocr, lang)
    return ocr