import os
import sys
import glob
import pickle
import shutil
import argparse
import tensorflow as tf

def _bytes_feature(value):
    """String, Byte 타입을 받아서 Byte list를 return"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """float, double 타입을 받아서 float list를 return"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """bool, enum, in, uint 타입을 받아서 int64 list를 리턴"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


"""DUMP CODE"""
# def _int64_feature(value):
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#
# def _bytes_features(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
# def read_pickle_from_file(filename):
#     with tf.io.gfile.GFile(filename, "rb") as f:
#         if sys.version_info >= (3, 0):
#             data_dict = pickle.load(f, encoding="bytes")
#         else:
#             data_dict = pickle.load(f)
#
#     return data_dict
#
# def convert_to_tfrecord(input_files, output_file):
#     print("Generating %s" % output_file)
#
#     with tf.io.TFRecordWriter(output_file) as record_writer:
#         for input_file in input_files:
#             data_dict = read_pickle_from_file(input_file)
#             data = data_dict[b"data"]
#             labels = data_dict[b"labels"]
#             num_entries_in_batch = len(labels)
#
#             for i in range(num_entries_in_batch):
#                 example = tf.train.Example(features=tf.train.Features(
#                     feature=(
#                         "image": _bytes_features(data[i.tobytes),
#                         "label": _int64_feature(labels[[i]])
#                     )))
#
#                 record_writer.write(example.SerializeToString())
#
# def _get_file_names(eval_file_idx):
#     file_names = {}
#     train_files_idx_list = [1, 2, 3, 4, 5, 6, 7]