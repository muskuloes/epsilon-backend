import sys
import cv2
import itertools
import json
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
model_path = "model/weights2.h5"

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

with open("model/label_descriptions.json", "r") as file:
    label_desc = json.load(file)
categories = label_desc["categories"]


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


def image_resize(image, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    width = None
    height = None
    (h, w) = image.shape[:2]
    if max(h, w) > 1024:
        if h > w:
            height = 1024
        else:
            width = 1024

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def detect(file):
    img_array = np.array(bytearray(file), dtype=np.uint8)
    img = cv2.cvtColor(cv2.imdecode(img_array, -1), cv2.COLOR_BGR2RGB)

    results = model.detect([img])
    predictions = results[0]
    predictions["bbox"] = [
        cv2.boundingRect(np.matrix(predictions["masks"][:, :, j] * 1, dtype=np.uint8))
        for j in range(predictions["masks"].shape[-1])
    ]
    for k, v in predictions.items():
        if k == "masks":
            if predictions[k].shape[-1] > 0:
                masks = predictions[k][:, :, 0].ravel(order="F")
                predictions[k] = rle_to_string(to_rle(masks))
            else:
                predictions[k] = ""

        elif k == "class_ids":
            predictions[k] = [
                c["name"]
                for class_id in predictions["class_ids"]
                for c in categories
                if c["id"] == class_id - 1
            ]
        elif k == "bbox":
            continue
        else:
            predictions[k] = v.tolist()
    return predictions


#  img = cv2.cvtColor(image_resize(cv2.imread("model/test.jpg")), cv2.COLOR_BGR2RGB)
#  results = model.detect([img])
#  r = results[0]
#  mask = r["masks"][:, :, 0].ravel(order="F")
#  rle = to_rle(mask)
#  for j in range(r["masks"].shape[-1]):
#  print(
#  "\nbounding box: {}\n".format(
#  cv2.boundingRect(np.matrix(r["masks"][:, :, j] * 1, dtype=np.uint8))
#  )
#  )
#  r["masks"] = rle
#  for k, v in r.items():
#  if k == "masks":
#  continue
#  elif k == "class_ids":
#  r["class_ids"] = [
#  t["name"]
#  for class_id in r["class_ids"]
#  for t in categories
#  if t["id"] == class_id - 1
#  ]
#  else:
#  r[k] = v.tolist()
#  print(r)
