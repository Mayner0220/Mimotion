import os
import sys
import glob
import pickle
import shutil
import argparse
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_pickle_from_file(filename):
    with tf.io.gfile.GFile(filename, "rb") as f:
        if sys.version_info >= (3, 0):
            data_dict = pickle.load(f, encoding="bytes")
        else:
            data_dict = pickle.load(f)


    return data_dict

def convert_to_tfrecord(input_files, output_file):
    print("Generating %s" % output_file)

    with tf.io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
            data_dict = read_pickle_from_file(input_file)
            data = data_dict[b"data"]
            labels = data_dict[b"labels"]
            num_entries_in_batch = len(labels)