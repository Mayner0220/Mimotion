import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

epoch = 50
num_val = 7178
batch_size = 64
num_train = 28709

train_dir = "./dataset/train"
test_dir = "./dataset/test"

Train_sad = len(os.walk(train_dir + "/sad").__next__()[2])
Train_fear = len(os.walk(train_dir + "/fear").__next__()[2])
Train_angry = len(os.walk(train_dir + "/angry").__next__()[2])
Train_happy = len(os.walk(train_dir + "/happy").__next__()[2])
Train_neutral = len(os.walk(train_dir + "/neutral").__next__()[2])
Train_disgust = len(os.walk(train_dir + "/disgust").__next__()[2])
Train_surprise = len(os.walk(train_dir + "/surprise").__next__()[2])

Test_sad = len(os.walk(test_dir + "/sad").__next__()[2])
Test_fear = len(os.walk(test_dir + "/fear").__next__()[2])
Test_angry = len(os.walk(test_dir + "/angry").__next__()[2])
Test_happy = len(os.walk(test_dir + "/happy").__next__()[2])
Test_neutral = len(os.walk(test_dir + "/neutral").__next__()[2])
Test_disgust = len(os.walk(test_dir + "/disgust").__next__()[2])
Test_surprise = len(os.walk(test_dir + "/surprise").__next__()[2])


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