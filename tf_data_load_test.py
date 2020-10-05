import tensorflow as tf
import numpy as np
from glob import glob
from PIL import Image

train_image_list = []
train_label_list = []

test_image_list = []
test_label_list = []

# Train dataset
train_path = "dataset/train/*/*.jpg"
train_data = glob(train_path)

# Test dataset
test_path = "dataset/test/*/*.jpg"
test_data = glob(test_path)

# Get label from data path
def get_label(path):
    return path.split("\\")[-2]

# Read image from data path
def read_image(path):
    image = np.array(Image.open(path))
    return image.reshape(image.shape[0], image.shape[1], 1)

train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
test_dataset = tf.data.Dataset.from_tensor_slices((test_image_list, test_label_list))