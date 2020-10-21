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

def serialize_example(feature0, feature1, feature2, feature3):
    """파일에 write 하기 위한 tf.Example messange 생성"""

    # Key: feature name
    # Value: tf.Example에 적합한 타입
    feature = {
        "feature0": _int64_feature(feature0),
        "feature1": _int64_feature(feature1),
        "feature2": _bytes_feature(feature2),
        "feature3": _float_feature(feature3)
    }

    # tf.train.Example을 통해 Feature message를 생성
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()