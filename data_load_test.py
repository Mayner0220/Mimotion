import os
from glob import glob

from PIL import Image
import numpy as np

path = "dataset/train/*/*.jpg"
# print(glob("dataset/train/*/*.jpg"))
print(glob(path))