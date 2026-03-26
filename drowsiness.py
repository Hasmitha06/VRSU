import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import pygame

# Initialize alarm
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

# Eye Aspect Ratio function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Constants
EYE_AR_THRESH = 0.24
EYE_AR_CONSEC_FRAMES = 15

COUNTER = 0
ALARM_ON = False

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)

        leftEye = [(shape.part(i).x, shape.part(i).y) for i in LEFT_EYE]
        rightEye = [(shape.part(i).x, shape.part(i).y) for i in RIGHT_EYE]

        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        print("EAR:", ear)

        # Draw eyes
        cv2.polylines(frame, [np.array(leftEye)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(rightEye)], True, (0, 255, 0), 1)

        # Drowsiness logic
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    pygame.mixer.music.play(-1)
                    ALARM_ON = True

                cv2.putText(frame, "DROWSINESS ALERT!", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            COUNTER = 0
            if ALARM_ON:
                pygame.mixer.music.stop()
                ALARM_ON = False

        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Drowsiness Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()