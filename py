import cv2
import mediapipe as mp
import numpy as np

# Initialize the hand tracking module

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

def count_fingers(landmarks, handedness):

    fingers = 0

    if handedness == "Right":
        if landmarks[4].x < landmarks[3].x: # thumb movement
            fingers += 1
    else:  # Left hand
        if landmarks[4].x > landmarks[3].x: # thumb movement
            fingers += 1

    tips = [8, 12, 16, 20] # index, middle, ring, pinky
    pips = [6, 10, 14, 18 ]

    for tip, pip in zip(tips, pips):
        if landmarks[tip].y < landmarks[pip].y:
            fingers += 1

    return fingers

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            original_label = handedness.classification[0].label
            if original_label == "Right":
                hand_label = "Left"
            else:
                hand_label = "Right"
            count = count_fingers(hand_landmarks.landmark, original_label)
            cv2.putText(frame, f"{hand_label}: {count}", (int(hand_landmarks.landmark[12].x * frame.shape[1]), int(hand_landmarks.landmark[12].y * frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
