import numpy as np
import cv2

def get_rotate_crop_image(img, points):
  """
  img_height, img_width = img.shape[0:2]
  left = int(np.min(points[:, 0]))
  right = int(np.max(points[:, 0]))
  top = int(np.min(points[:, 1]))
  bottom = int(np.max(points[:, 1]))
  img_crop = img[top:bottom, left:right, :].copy()
  points[:, 0] = points[:, 0] - left
  points[:, 1] = points[:, 1] - top
  """
  assert len(points) == 4, "shape of points must be 4*2"
  img_crop_width = int(
    max(
      np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
    )
  )
  img_crop_height = int(
    max(
      np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
    )
  )
  pts_std = np.float32(
    [
      [0, 0],
      [img_crop_width, 0],
      [img_crop_width, img_crop_height],
      [0, img_crop_height],
    ]
  )
  M = cv2.getPerspectiveTransform(points, pts_std)
  dst_img = cv2.warpPerspective(
    img,
    M,
    (img_crop_width, img_crop_height),
    borderMode=cv2.BORDER_REPLICATE,
    flags=cv2.INTER_CUBIC,
  )
  dst_img_height, dst_img_width = dst_img.shape[0:2]
  if dst_img_height * 1.0 / dst_img_width >= 1.5:
    dst_img = np.rot90(dst_img)
  return dst_img


def get_minarea_rect_crop(img, points):
  bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
  points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

  index_a, index_b, index_c, index_d = 0, 1, 2, 3
  if points[1][1] > points[0][1]:
    index_a = 0
    index_d = 1
  else:
    index_a = 1
    index_d = 0
  if points[3][1] > points[2][1]:
    index_b = 2
    index_c = 3
  else:
    index_b = 3
    index_c = 2

  box = [points[index_a], points[index_b], points[index_c], points[index_d]]
  crop_img = get_rotate_crop_image(img, np.array(box))
  return crop_img

def str2bool(v):
  return v.lower() in ("true", "t", "1")