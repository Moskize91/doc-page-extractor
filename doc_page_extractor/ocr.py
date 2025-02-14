import os
import numpy as np
import cv2

from typing import Literal, Any
from paddleocr import PaddleOCR
from .utils import ensure_dir


# https://github.com/PaddlePaddle/PaddleOCR/blob/2c0c4beb0606819735a16083cdebf652939c781a/paddleocr.py#L108-L157
PaddleLang = Literal["ch", "en", "korean", "japan", "chinese_cht", "ta", "te", "ka", "latin", "arabic", "cyrillic", "devanagari"]

# https://paddlepaddle.github.io/PaddleOCR/latest/quick_start.html#_2
class OCR:
  def __init__(
      self,
      device: Literal["cpu", "cuda"],
      model_dir_path: str,
      bin: bool = True,
      inv: bool = False,
    ):
    self._device: Literal["cpu", "cuda"] = device
    self._model_dir_path: str = model_dir_path
    self._ocr_and_lan: tuple[PaddleOCR, PaddleLang] | None = None
    self._bin: bool = bin
    self._inv: bool = inv

  def do(self, lang: PaddleLang, image: np.ndarray) -> list[Any]:
    ocr = self._get_ocr(lang)
    image = self._preprocess_image(image)
    # about img parameter to see
    # https://github.com/PaddlePaddle/PaddleOCR/blob/2c0c4beb0606819735a16083cdebf652939c781a/paddleocr.py#L582-L619
    return ocr.ocr(img=image, cls=True)

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

  def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
    image = self._alpha_to_color(image, (255, 255, 255))
    if self._inv:
      image = cv2.bitwise_not(image)
    if self._bin:
      image = self._binarize_img(image)
    return image

  def _alpha_to_color(self, image: np.ndarray, alpha_color: tuple[float, float, float]) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] == 4:
      B, G, R, A = cv2.split(image)
      alpha = A / 255

      R = (alpha_color[0] * (1 - alpha) + R * alpha).astype(np.uint8)
      G = (alpha_color[1] * (1 - alpha) + G * alpha).astype(np.uint8)
      B = (alpha_color[2] * (1 - alpha) + B * alpha).astype(np.uint8)

      image = cv2.merge((B, G, R))

    return image

  def _binarize_img(self, image: np.ndarray):
    if len(image.shape) == 3 and image.shape[2] == 3:
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # conversion to grayscale image
      # use cv2 threshold binarization
      _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return image
