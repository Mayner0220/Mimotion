# import libraries
import sys
import pickle
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt

class FER2013Data(object):
    def __init__(self, batch_size, tfrecords_path="./tfrecords"):
        self.batch_size =  batch_size
        self.tfrecords_path = tfrecords_path
        self.image_height = 48
        self.image_width = 48
        self.image_num_channels = 3
        self.train_data_size = 28709
        self.test_data_size = 22968

        # Create datasets
        self.train_dataset = self.create_dataset(is_train_or_test="train")
        self.test_dataset = self.create_dataset(is_train_or_test="test")

    def create_dataset(self, is_train_or_test="train"):
        if is_train_or_test is "train":
            tfrecords_files = glob(self.tfrecords_path + "/train/*")
        elif is_train_or_test is "test":
            tfrecords_files = glob(self.tfrecords_path + "/test/*")

        # Read dataset from TFRecords
        dataset = tf.data.TFRecordDataset(tfrecords_files)

        dataset = dataset.map(self._parse_and_decode, num_parallel_calls=self.batch_size)

        if is_train_or_test is "train":
            dataset = dataset.shuffle(10000).repeat().batch(self.batch_size)
        else:
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(self.batch_size)

            return dataset

    def _parse_and_decode(self, serialized_example):
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                "image": tf.io.FixedLenFeature([], tf.string),
                "label": tf.io.FixedLenFeature([], tf.int64)
            }
        )

        image = tf.io.decode_raw(features["image"], tf.uint8)
        image = tf.cast(image, tf.float32)

        image.set_shape(self.image_num_channels * self.image_height * self.image_width)

        image = tf.cast(
            tf.transpose(tf.reshape(
                image, [self.image_num_channels, self.image_height, self.image_width])
            [1, 2, 0]), tf.float32)

        label = tf.cast(features["label"], tf.int32)
        label = tf.one_hot(label, 7)

        return image, label

def main(argv):
    if argv is not None:
        print("argv: {}".format(argv))

        batch_size = 32

        tfrecords_path = "./tfrecords"
        FER2013_data = FER2013Data(batch_size, tfrecords_path)
        tds = FER2013_data.train_dataset

        print(tds)

        for images, classes in tds.take(4):
            print('shape %', images.shape)
            print('shape %', classes.shape)

if __name__ == '__main__':
  main(sys.argv)