from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from create_tfrecords_from_data import _int64_feature, _bytes_feature

cat_in_snow  = tf.keras.utils.get_file('320px-Felis_catus-cat_on_snow.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
williamsburg_bridge = tf.keras.utils.get_file('194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')

cat_in_snow_image = Image.open(cat_in_snow)
plt.imshow(cat_in_snow_image)
plt.show()

williamsburg_bridge_image = Image.open(williamsburg_bridge)
plt.imshow(williamsburg_bridge_image, cmap="gray")
plt.show()

image_labels = {
    cat_in_snow : 0,
    williamsburg_bridge : 1,
}

image_string = open(cat_in_snow, "rb").read()
label = image_labels[cat_in_snow]

# sess = tf.compat.v1.InteractiveSession()

def image_example(image_string, label):
    with tf.compat.v1.Session() as sess:
        image_shape = sess.run(tf.image.decode_jpeg(image_string)).shape

        feature = {
            "height": _int64_feature(image_shape[0]),
            "width": _int64_feature(image_shape[1]),
            "depth": _int64_feature(image_shape[2]),
            "label": _int64_feature(label),
            "image_raw": _bytes_feature(image_string)
        }

    return tf.train.Example(feature=tf.train.Feature(feature=feature))

for line in str(image_example(image_string, label)).split("\n")[:15]:
    print(line)

print("...")