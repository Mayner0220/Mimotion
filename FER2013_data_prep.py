from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "./dataset/train"
test_dir = "./dataset/test"

num_train = 28709
num_val = 7178
batch_size = 64
epoch = 50

generator = ImageDataGenerator(rescale=1./255)

train_generator = generator.flow_from_directory(
    train_dir,
    target_size=(48,48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical"
)

test_generator = generator.flow_from_directory(
    test_dir,
    target_size=(48,48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical"
)