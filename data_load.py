# Reference: https://medium.com/trackin-datalabs/data-input-%EB%A7%8C%EB%93%A4%EA%B8%B0-74bb5c1ce52f
import os
from glob import glob
from PIL import Image
import numpy as np

# Hyper Parameter
batch_size = 64
data_height = 48
data_width = 48
channel_n = 1

num_classes = 7

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

batch_image = np.zeros((batch_size, data_height, data_width, channel_n))
batch_label = np.zeros((batch_size, num_classes))

for n, path in enumerate(train_data[:batch_size]):
    image = read_image(path)
    onehot_label = onehot_encode_label(path)
    batch_image[n, :, :, :] = image
    batch_label[n, :] = onehot_label

print(batch_image.shape, batch_label.shape)