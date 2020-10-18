import numpy as np
import tensorflow as tf
from create_tfrecords_from_data import serialize_example

# serialize_example 함수를 이용해서 binary string으로 serialize
serialized_example = serialize_example(False, 4, b'goat', 0.9876)
print(serialized_example)

# serilization된 데이터를 tf.train.Example.FromString 메소드를 이용해서 decode
example_proto = tf.train.Example.FromString(serialized_example)
print(example_proto)

filename = "test.tfrecord"

# observation 횟수
num_observation = int(1e4)

# boolen feature - [False or True]
feature0 = np.random.choice([False, True], num_observation)

# integer feature - [0 ... 4]
feature1 = np.random.choice(0, 5, num_observation)

# String feature
strings = np.array([b"cat", b"dog", b"chicken", b"horse" b"goat"])
feature2 = strings[feature1]

# float feature - from standard normal distribution
feature3 = np.random.randn(num_observation)

# tf.Example 데이터를 TFRecord 파일에 write
with tf.io.TFRecordWriter(filename) as writer:
    for i in range(num_observation):
        example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])

writer.write(example)

"""
Tensorflow core v2.3.0에서 tf.io.tf_record_iterator API가 지원 중단됨
"""