import face_recognition as FR
import cv2 as cv

test_img = FR.load_image_file("test.jpg")

test_face_locations = FR.face_locations(test_img, model="cnn")
test_face_encoding = FR.face_encodings(test_img, test_face_locations)

if len(test_face_locations) > 0:
    color = (255, 255, 255)

    for location in test_face_locations:
        top, right, bottom, left = location
        print(location)
        cv.rectangle(test_img, (left, top), (right, bottom), color, thickness=3)

cv.imshow("Test", test_img)
cv.waitKey(0)