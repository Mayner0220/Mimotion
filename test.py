import face_recognition as FR
import cv2

image = FR.load_image_file("./testImg.png")
face_location = FR.face_locations(image, model="cnn")