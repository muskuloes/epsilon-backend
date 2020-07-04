import sys
import cv2
import numpy as np
import tensorflow as tf
import keras
import requests
from celery.signals import worker_process_init

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


def detect(file):
    img_array = np.array(bytearray(file), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)
    return model.detect([img])
