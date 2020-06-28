import sys
import cv2
import tensorflow as tf

# Import Mask RCNN
sys.path.append("unicorn-fashionguide/experimental-models/mask-rcnn")
from mrcnn.config import Config
import mrcnn.model as modellib

MODEL_DIR = "./model"

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
config.display()

class InferenceConfig(FashionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = "model/model.h5"

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

original_image = cv2.imread("test.jpg")
results = model.detect([original_image])
print(model.metrics_names)
