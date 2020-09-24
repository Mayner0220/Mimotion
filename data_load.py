import os
from glob import glob

# Dataset Path
train_path = "dataset/train/*/*.jpg"
test_path = "dataset/test/*/*.jpg"

# Get data's path list
train_data = glob(train_path)
test_path = glob(test_path)

# Get label from data path
def get_label(path):
    return path.split("\\")[-2]