# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
Gesture Calibration Tool
========================
Run this BEFORE gesture_control.py to test and verify
your gestures are being recognized correctly.
No game needed - just checks what each gesture maps to.

Usage:
    python calibrate.py
"""

import cv2
import mediapipe as mp
import collections

CAMERA_INDEX = 0
FLIP_CAMERA  = True

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

def fingers_extended(hand_landmarks, handedness):
    lm = hand_landmarks.landmark
    tips    = [4, 8, 12, 16, 20]
    knuckle = [3, 6, 10, 14, 18]
    extended = []
    is_right = (handedness == "Right")
    if is_right:
        extended.append(lm[tips[0]].x < lm[knuckle[0]].x)
    else:
        extended.append(lm[tips[0]].x > lm[knuckle[0]].x)
    for i in range(1, 5):
        extended.append(lm[tips[i]].y < lm[knuckle[i]].y)
    return extended

def classify_left_gesture(hand_landmarks):
    lm  = hand_landmarks.landmark
    ext = fingers_extended(hand_landmarks, "Left")
    thumb, index, middle, ring, pinky = ext
    if index and middle and ring and pinky:
        return "FORWARD (W)"
    if not index and not middle and not ring and not pinky:
        return "BACKWARD (S)"
    if index and middle and not ring and not pinky:
        return "LEFT (A)"
    if index and middle and ring and not pinky:
        return "RIGHT (D)"
    return "NEUTRAL"

def classify_right_gesture(hand_landmarks):
    ext = fingers_extended(hand_landmarks, "Right")
    thumb, index, middle, ring, pinky = ext
    if index and middle and ring and pinky:
        return "JUMP (Space)"
    if index and thumb and not middle and not ring and not pinky:
        return "SHOOT (F)"
    if not index and not middle and ring and pinky:
        return "CROUCH (Ctrl)"
    if not index and not middle and not ring and not pinky:
        return "DODGE (Shift)"
    if not index and not middle and not ring and pinky:
        return "INTERACT (E)"
    return "NEUTRAL"

def main():
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.6,
    )
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    left_hist  = collections.deque(maxlen=4)
    right_hist = collections.deque(maxlen=4)

    print("Calibration Mode - No keypresses sent. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if FLIP_CAMERA:
            frame = cv2.flip(frame, 1)

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        left_g  = "- no hand -"
        right_g = "- no hand -"

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_info.classification[0].label
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
                if label == "Left":
                    raw = classify_left_gesture(hand_lm)
                    left_hist.append(raw)
                    left_g = collections.Counter(left_hist).most_common(1)[0][0]
                else:
                    raw = classify_right_gesture(hand_lm)
                    right_hist.append(raw)
                    right_g = collections.Counter(right_hist).most_common(1)[0][0]

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.putText(frame, "CALIBRATION MODE - no keys sent",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 180, 220), 2)
        cv2.putText(frame, f"LEFT  HAND:  {left_g}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 100), 2)
        cv2.putText(frame, f"RIGHT HAND:  {right_g}",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 100, 220), 2)

        cv2.imshow("Gesture Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Calibration closed.")

if __name__ == "__main__":
    main()
