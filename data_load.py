import os
from glob import glob
from PIL import Image
import numpy as np

train_path = "dataset/train/*/*.jpg"
train_data = glob(train_path)
# train_image = np.array(Image.open(train_data))

test_path = "dataset/test/*/*.jpg"
test_data = glob(test_path)
# test_image = np.array(Image.open(test_data))

# Get label from data path
def get_label(path):
    return path.split("\\")[-2]

def read_image(path):
    image = np.array(Image.open(path))
    return image.reshape(image.shape[0], image.shape[1], 1)

label_name_list = []

for path in train_data:
    label_name_list.append(get_label(path))

unique_label_names = np.unique(label_name_list)

def onehot_encode_label(path):
    onehot_label = unique_label_names == get_label(path)
    onehot_label = onehot_label.astype(np.uint8)
    return onehot_label
