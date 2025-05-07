import numpy as np
import cv2
import os

from typing import Literal, Generator
from dataclasses import dataclass
from .onnxocr import TextSystem
from .types import GetModelDir, OCRFragment
from .rectangle import Rectangle
from .utils import is_space_text


_MODELS = (
  ("ppocrv4", "rec", "rec.onnx"),
  ("ppocrv4", "cls", "cls.onnx"),
  ("ppocrv4", "det", "det.onnx"),
  ("ch_ppocr_server_v2.0", "ppocr_keys_v1.txt"),
)

@dataclass
class _OONXParams:
  use_angle_cls: bool
  use_gpu: bool
  rec_image_shape: tuple[int, int, int]
  cls_image_shape: tuple[int, int, int]
  cls_batch_num: int
  cls_thresh: float
  label_list: list[str]

  det_algorithm: str
  det_limit_side_len: int
  det_limit_type: str
  det_db_thresh: float
  det_db_box_thresh: float
  det_db_unclip_ratio: float
  use_dilation: bool
  det_db_score_mode: str
  det_box_type: str
  rec_batch_num: int
  drop_score: float
  save_crop_res: bool
  rec_algorithm: str
  use_space_char: bool
  rec_model_dir: str
  cls_model_dir: str
  det_model_dir: str
  rec_char_dict_path: str




class OCR:
  def __init__(
      self,
      device: Literal["cpu", "cuda"],
      get_model_dir: GetModelDir,
    ):
    self._device: Literal["cpu", "cuda"] = device
    self._get_model_dir: GetModelDir = get_model_dir
    self._text_system: TextSystem | None = None

  def search_fragments(self, image: np.ndarray) -> Generator[OCRFragment, None, None]:
    for box, res in self._ocr(image):
      text, rank = res
      if is_space_text(text):
        continue

      rect = Rectangle(
        lt=(box[0][0], box[0][1]),
        rt=(box[1][0], box[1][1]),
        rb=(box[2][0], box[2][1]),
        lb=(box[3][0], box[3][1]),
      )
      if not rect.is_valid or rect.area == 0.0:
        continue

      yield OCRFragment(
        order=0,
        text=text,
        rank=rank,
        rect=rect,
      )

  def _ocr(self, image: np.ndarray) -> Generator[tuple[list[list[float]], tuple[str, float]], None, None]:
    text_system = self._get_text_system()
    image = self._preprocess_image(image)
    dt_boxes, rec_res = text_system(image)

    for box, res in zip(dt_boxes, rec_res):
      yield box.tolist(), res

  def make_model_paths(self) -> list[str]:
    model_paths = []
    model_dir = self._get_model_dir()
    for model_path in _MODELS:
      file_name = os.path.join(*model_path)
      model_paths.append(os.path.join(model_dir, file_name))
    return model_paths

  def _get_text_system(self) -> TextSystem:
    if self._text_system is None:
      model_paths = self.make_model_paths()
      self._text_system = TextSystem(_OONXParams(
        use_angle_cls=True,
        use_gpu=(self._device != "cpu"),
        rec_image_shape=(3, 48, 320),
        cls_image_shape=(3, 48, 192),
        cls_batch_num=6,
        cls_thresh=0.9,
        label_list=["0", "180"],
        det_algorithm="DB",
        det_limit_side_len=960,
        det_limit_type="max",
        det_db_thresh=0.3,
        det_db_box_thresh=0.6,
        det_db_unclip_ratio=1.5,
        use_dilation=False,
        det_db_score_mode="fast",
        det_box_type="quad",
        rec_batch_num=6,
        drop_score=0.5,
        save_crop_res=False,
        rec_algorithm="SVTR_LCNet",
        use_space_char=True,
        rec_model_dir=model_paths[0],
        cls_model_dir=model_paths[1],
        det_model_dir=model_paths[2],
        rec_char_dict_path=model_paths[3],
      ))

    return self._text_system

  def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
    image = self._alpha_to_color(image, (255, 255, 255))
    # image = cv2.bitwise_not(image) # inv
    # image = self._binarize_img(image) # bin
    image = cv2.normalize(
      src=image,
      dst=np.zeros((image.shape[0], image.shape[1])),
      alpha=0,
      beta=255,
      norm_type=cv2.NORM_MINMAX,
    )
    image = cv2.fastNlMeansDenoisingColored(
      src=image,
      dst=None,
      h=10,
      hColor=10,
      templateWindowSize=7,
      searchWindowSize=15,
    )
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # image to gray
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
