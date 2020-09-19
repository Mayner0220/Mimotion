import face_recognition as FR
import cv2
import face_recognition_models

# face_recognition library: https://github.com/ageitgey/face_recognition
# Reference: https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py

# Get a reference to webcam #0
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Initialize variables
face_locations = []
face_encodings = []
process_this_frame = True

while True:
    # Grab a single frame of capture
    ret, frame = capture.read()

    # Resize frame of capture to 1/4 size
    # For faster face recognition prcoessing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color to RGB color
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        # Find all faces and face encodings in the current frame of video
        # It will be developed to determine whether the CNN model will be used or not,
        # depending on the user's use of CUDA.

        # CUDA unavailable
        # face_locations = FR.face_locations(rgb_small_frame)

        # CUDA available
        face_locations = FR.face_locations(rgb_small_frame, model="cnn")
        face_encodings = FR.face_encodings(rgb_small_frame, face_locations)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left) in face_locations:
        # Scale back up face location
        # The frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        # Rectangle's color: teal - RGB(128, 128, 000)
        cv2.rectangle(frame, (left, top), (right, bottom), (128, 128, 0), 2)

    # Display the result image
    cv2.imshow("Face Recognition", frame)

    # Push the 'q' key on keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release handle to the webcam
capture.release()
cv2.destroyAllWindows()