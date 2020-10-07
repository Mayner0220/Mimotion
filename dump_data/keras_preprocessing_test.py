import tensorflow as tf
import matplotlib.pyplot as plt

# Hyperparameter
batch_size = 64
img_height = 48
img_width = 48
channel_n = 1

# Train & Validation dataset dir path
train_data_dir = "./dataset/train"
validation_data_dir = "./dataset/train"

# Get train dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="training",
    seed=2020,
    image_size=(img_height, img_height),
    batch_size=batch_size
)

# Get validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    validation_data_dir,
    validation_split=0.2,
    subset="validation",
    seed=2020,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")