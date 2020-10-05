# Reference: https://medium.com/trackin-datalabs/data-input-%EB%A7%8C%EB%93%A4%EA%B8%B0-74bb5c1ce52f
from glob import glob
from PIL import Image
import numpy as np

# Hyper Parameter
batch_size = 64
data_height = 48
data_width = 48
channel_n = 1

# Number of Classes
num_classes = 7

# Label name list
train_label_name_list = []
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


for path in train_data:
    train_label_name_list.append(get_label(path))

for path in test_data:
    test_label_name_list.append(get_label(path))

unique_label_names = np.unique(train_label_name_list)


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
test1 = Image.open(train_data[100])
test2 = Image.open(train_data[101])
test1.show()
test2.show()