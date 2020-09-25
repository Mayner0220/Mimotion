import os
from glob import glob
from PIL import Image
import numpy as np

train_path = "dataset/train/*/*.jpg"
train_data = glob(train_path)
train_image = np.array(Image.open(train_data))

test_path = "dataset/test/*/*.jpg"
test_data = glob(test_path)
test_image = np.array(Image.open(test_data))

# Get label from data path
def get_label(path):
    return path.split("\\")[-2]