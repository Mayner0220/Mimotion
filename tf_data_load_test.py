import tensorflow as tf
import numpy as np
from glob import glob
from PIL import Image

batch_size = 64
data_height = 48
data_width = 48
channel_n = 1

train_image_list = []
train_label_list = []
train_label_name_list = []

test_image_list = []
test_label_list = []
test_label_name_list = []

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

def dataset_map(dataset, image_list, label_list):
    dataset = dataset.map(lambda image_list, label_list: tuple(
        tf.py_function([image_list, label_list], [tf.int32, tf.uint8])
    ))

    return dataset

for path in train_data:
    train_label_name_list.append(get_label(path))

for path in test_data:
    test_label_name_list.append(get_label(path))

train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
test_dataset = tf.data.Dataset.from_tensor_slices((test_image_list, test_label_list))

map_train_dataset = dataset_map(train_dataset, read_image(train_data), get_label(train_label_name_list))
map_test_dataset = dataset_map(test_dataset, read_image(test_data), get_label(test_label_name_list))