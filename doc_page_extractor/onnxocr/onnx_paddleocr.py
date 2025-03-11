import os

from .predict_system import TextSystem


class ONNXPaddleOcr(TextSystem):
  def __init__(self, model_dir_path: str):
    super().__init__({
      "use_angle_cls": True,
      "use_gpu": False,
      "rec_image_shape": "3, 48, 320",
      "cls_image_shape": "3, 48, 192",
      "cls_batch_num": 6,
      "cls_thresh": 0.9,
      "label_list": ["0", "180"],
      "det_algorithm": "DB",
      "det_limit_side_len": 960,
      "det_limit_type": "max",
      "det_db_thresh": 0.3,
      "det_db_box_thresh": 0.6,
      "det_db_unclip_ratio": 1.5,
      "use_dilation": False,
      "det_db_score_mode": "fast",
      "det_box_type": "quad",
      "rec_batch_num": 6,
      "drop_score": 0.5,
      "save_crop_res": False,
      "rec_algorithm": "SVTR_LCNet",
      "use_space_char": True,
      "rec_model_dir": os.path.join(model_dir_path, "ppocrv4", "rec", "rec.onnx"),
      "cls_model_dir": os.path.join(model_dir_path, "ppocrv4", "cls", "cls.onnx"),
      "det_model_dir": os.path.join(model_dir_path, "ppocrv4", "det", "det.onnx"),
      "rec_char_dict_path": os.path.join(model_dir_path, "ch_ppocr_server_v2.0", "ppocr_keys_v1.txt"),
    })

  def ocr(self, img, det=True, rec=True, cls=True):
    if det and rec:
      ocr_res = []
      dt_boxes, rec_res = self.__call__(img, cls)
      tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
      ocr_res.append(tmp_res)
      return ocr_res
    elif det and not rec:
      ocr_res = []
      dt_boxes = self.text_detector(img)
      tmp_res = [box.tolist() for box in dt_boxes]
      ocr_res.append(tmp_res)
      return ocr_res
    else:
      ocr_res = []
      cls_res = []

      if not isinstance(img, list):
        img = [img]
      if self.use_angle_cls and cls:
        img, cls_res_tmp = self.text_classifier(img)
        if not rec:
          cls_res.append(cls_res_tmp)
      rec_res = self.text_recognizer(img)
      ocr_res.append(rec_res)

      if not rec:
        return cls_res
      return ocr_res

