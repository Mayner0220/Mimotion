import cv2
import argparse
import numpy as np
from model import Model
from plot_history import plot_model
from tensorflow.keras.optimizers import Adam
from FER2013_data_prep import train_generator, validation_generator

epoch = 50
num_val = 7178
batch_size = 64
num_train = 28709

model = Model()

# Temporarily disable the argparse library for fast debugging
ap = argparse.ArgumentParser("Choose mode")
ap.add_argument("--mode", help="train/display")
mode = ap.parse_args().mode

# mode = input("[Mode]\n>> ")
# print("Mode:", mode)

# Train same model or try other models
if mode == "train":
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.0001, decay=1e-6), metrics=["accuracy"])
    model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size
        #callbacks=callback
    )

    model.save_weights("weight/model.h5")

    plot_model(model_info)

# emotions will be displayed on your face from the webcam feed
elif mode == "display":
    model.load_weights("weight/model.h5")

    cv2.ocl.setUseOpenCL(False)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)

        if not ret:
            break

        facecasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()