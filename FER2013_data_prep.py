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

validation_generator = generator.flow_from_directory(
    test_dir,
    target_size=(48,48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical"
)

print("\n=======================[Train Dataset Information]=======================")
print("train: sad -", Train_sad)
print("train: fear -", Train_fear)
print("train: angry -", Train_angry)
print("train: happy -", Train_happy)
print("train: neutral -", Train_neutral)
print("train: disgust -", Train_disgust)
print("train: surprise -", Train_surprise)

print("\n=======================[Test Dataset Information]=======================")
print("test: sad -", Test_sad)
print("test: fear -", Test_fear)
print("test: angry -", Test_angry)
print("test: happy -", Test_happy)
print("test: neutral -", Test_neutral)
print("test: disgust -", Test_disgust)
print("test: surprise -", Test_surprise)