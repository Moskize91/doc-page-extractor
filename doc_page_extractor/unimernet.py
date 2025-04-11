import argparse
import numpy as np
import unimernet.tasks as tasks

from typing import Literal
from PIL import Image
from unimernet.common.config import Config
from unimernet.processors import load_processor


class Unimernet:
  def __init__(self, cfg_path: str, device: Literal["cpu", "cuda"]):
    self._cfg_path: str = cfg_path
    self._device: Literal["cpu", "cuda"] = device
    self._model, self._vis_processor = self._load_model_and_processor()

  def _load_model_and_processor(self):
    args = argparse.Namespace(cfg_path=self._cfg_path, options=None)
    cfg = Config(args)
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg).to(self._device)
    vis_processor = load_processor(
      name="formula_image_eval",
      cfg=cfg.config.datasets.formula_rec_eval.vis_processor.eval,
    )
    return model, vis_processor

  def process_image(self, image_path):
    try:
      raw_image = Image.open(image_path)
    except IOError:
      print(f"Error: Unable to open image at {image_path}")
      return
    # Convert PIL Image to OpenCV format
    open_cv_image = np.array(raw_image)
    # Convert RGB to BGR
    if len(open_cv_image.shape) == 3:
      # Convert RGB to BGR
      open_cv_image = open_cv_image[:, :, ::-1].copy()
    # Display the image using cv2

    image = self._vis_processor(raw_image).unsqueeze(0).to(self._device)
    output = self._model.generate({"image": image})
    pred = output["pred_str"][0]

    return pred