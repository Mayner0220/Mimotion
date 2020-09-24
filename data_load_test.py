import os
from glob import glob

from PIL import Image
import numpy as np

print(glob(r'C:\U*'))
path = "dataset/train/*/*.jpg"
data = glob(r"dataset/train/*/*.jpg")
print(data)