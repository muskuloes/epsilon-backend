import sys
import cv2
import itertools
import keras
import numpy as np
import requests
import tensorflow as tf

from model.mrcnn.config import Config
import model.mrcnn.model as modellib

MODEL_DIR = "model/mrcnn/model"


class FashionConfig(Config):
    """Configuration for training on the toy shapes dataset.
   Derives from the base Config class and overrides values specific
   to the toy shapes dataset.
   """

    # Give the configuration a recognizable name
    NAME = "fashion"

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Number of classes (including background)
    NUM_CLASSES = 46 + 1  # iMaterialist Dataset has 46 classes


config = FashionConfig()


class InferenceConfig(FashionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(
    mode="inference", config=inference_config, model_dir=MODEL_DIR
)

# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = "model/weights.h5"

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


def refine_masks(masks, rois):
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m] == True)
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois


def to_rle(bits):
    rle = []
    pos = 0
    for bit, group in itertools.groupby(bits):
        group_list = list(group)
        if bit:
            rle.extend([pos, sum(group_list)])
        pos += len(group_list)
    return rle


def rle_decode(rle_str, mask_shape, mask_dtype=np.uint8):
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(np.prod(mask_shape), dtype=mask_dtype)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(mask_shape[::-1]).T


def rle_to_string(runs):
    return " ".join(str(x) for x in runs)


def detect(file):
    img_array = np.array(bytearray(file), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)
    results = model.detect([img])
    predictions = results[0]
    for k, v in predictions.items():
        if k == "masks":
            masks = predictions["masks"][:, :, 0].ravel(order="F")
            predictions["masks"] = rle_to_string(to_rle(masks))
        else:
            predictions[k] = v.tolist()
    return predictions


#  img = cv2.imread("model/test.jpg")
#  results = model.detect([img])
#  r = results[0]
#  mask = r["masks"][:, :, 0].ravel(order="F")
#  rle = to_rle(mask)
#  r["masks"] = rle
#  for k, v in r.items():
#  if k == "masks":
#  continue
#  r[k] = v.tolist()
#  print(r)
