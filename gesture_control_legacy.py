# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
Tomb Raider Gesture Control System
====================================
Control Rise of the Tomb Raider using hand gestures via your webcam.

LEFT HAND  → Movement (WASD)
RIGHT HAND → Actions (Jump, Crouch, Shoot, Dodge)

Requirements:
    pip install opencv-python mediapipe pynput numpy

Run BEFORE launching the game, then Alt+Tab into the game.
"""

import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Key, Controller
import time
import collections

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
CAMERA_INDEX      = 0       # Change if webcam not detected
CONFIDENCE        = 0.75    # Hand detection confidence threshold
SMOOTHING_FRAMES  = 4       # Frames to smooth gesture over (reduces flicker)
HOLD_DELAY        = 0.08    # Seconds between repeated key sends
SHOW_LANDMARKS    = True    # Draw hand skeleton on feed
FLIP_CAMERA       = True    # Mirror the webcam feed

# ──────────────────────────────────────────────
# GESTURE → KEY MAPPING
# ──────────────────────────────────────────────
# LEFT HAND gestures  → Movement keys
LEFT_GESTURE_MAP = {
    "FORWARD":   "w",
    "BACKWARD":  "s",
    "LEFT":      "a",
    "RIGHT":     "d",
    "NEUTRAL":   None,
}

# RIGHT HAND gestures → Action keys
RIGHT_GESTURE_MAP = {
    "JUMP":      Key.space,
    "CROUCH":    Key.ctrl_l,
    "SHOOT":     "f",          # Aim/shoot (rebind in game to F)
    "DODGE":     Key.shift_l,  # Roll/sprint
    "INTERACT":  "e",
    "NEUTRAL":   None,
}

# ──────────────────────────────────────────────
# GESTURE RECOGNITION LOGIC
# ──────────────────────────────────────────────

def fingers_extended(hand_landmarks, handedness):
    """
    Returns a list of booleans [thumb, index, middle, ring, pinky]
    indicating which fingers are extended.
    """
    lm = hand_landmarks.landmark
    tips   = [4, 8, 12, 16, 20]   # tip landmark IDs
    knuckle= [3, 6, 10, 14, 18]   # lower knuckle IDs

    extended = []

    # Thumb: compare x-axis (mirrored for left vs right hand)
    is_right = (handedness == "Right")
    if is_right:
        extended.append(lm[tips[0]].x < lm[knuckle[0]].x)
    else:
        extended.append(lm[tips[0]].x > lm[knuckle[0]].x)

    # Other fingers: tip y < knuckle y means extended (up)
    for i in range(1, 5):
        extended.append(lm[tips[i]].y < lm[knuckle[i]].y)

    return extended  # [thumb, index, middle, ring, pinky]


def classify_left_gesture(hand_landmarks):
    """Movement gestures based on palm/wrist direction and finger state."""
    lm = hand_landmarks.landmark
    ext = fingers_extended(hand_landmarks, "Left")
    thumb, index, middle, ring, pinky = ext

    wrist_y  = lm[0].y
    mid_tip_y= lm[12].y

    # FORWARD — open hand, palm facing camera, fingers up
    if index and middle and ring and pinky:
        return "FORWARD"

    # BACKWARD — closed fist
    if not index and not middle and not ring and not pinky:
        return "BACKWARD"

    # LEFT — index + middle up only (peace/scissors sign ✌️)
    if index and middle and not ring and not pinky:
        return "LEFT"

    # RIGHT — index + middle + ring up, pinky down (3 fingers, no pinky)
    if index and middle and ring and not pinky:
        return "RIGHT"

    return "NEUTRAL"


def classify_right_gesture(hand_landmarks):
    """Action gestures for right hand."""
    lm = hand_landmarks.landmark
    ext = fingers_extended(hand_landmarks, "Right")
    thumb, index, middle, ring, pinky = ext

    # JUMP — open hand (all fingers up)
    if index and middle and ring and pinky:
        return "JUMP"

    # SHOOT — gun shape: index + thumb extended, rest curled
    if index and thumb and not middle and not ring and not pinky:
        return "SHOOT"

    # CROUCH — ring + pinky up, index + middle curled (two-finger wave down)
    if not index and not middle and ring and pinky:
        return "CROUCH"

    # DODGE — closed fist (roll/sprint)
    if not index and not middle and not ring and not pinky:
        return "DODGE"

    # INTERACT — pinky only
    if not index and not middle and not ring and pinky:
        return "INTERACT"

    return "NEUTRAL"


# ──────────────────────────────────────────────
# KEY CONTROLLER
# ──────────────────────────────────────────────

class KeyController:
    def __init__(self):
        self.kb = Controller()
        self.active_keys = set()

    def press(self, key):
        if key and key not in self.active_keys:
            self.kb.press(key)
            self.active_keys.add(key)

    def release(self, key):
        if key and key in self.active_keys:
            self.kb.release(key)
            self.active_keys.discard(key)

    def release_all(self):
        for k in list(self.active_keys):
            self.kb.release(k)
        self.active_keys.clear()

    def update(self, desired_keys: set):
        """Press new keys, release keys no longer needed."""
        to_release = self.active_keys - desired_keys
        to_press   = desired_keys - self.active_keys
        for k in to_release:
            self.release(k)
        for k in to_press:
            self.press(k)


# ──────────────────────────────────────────────
# GESTURE SMOOTHER (reduces flicker)
# ──────────────────────────────────────────────

class GestureSmoother:
    def __init__(self, window=SMOOTHING_FRAMES):
        self.window = window
        self.history = collections.deque(maxlen=window)

    def update(self, gesture):
        self.history.append(gesture)
        # Return the most common gesture in the window
        return collections.Counter(self.history).most_common(1)[0][0]


# ──────────────────────────────────────────────
# HUD OVERLAY
# ──────────────────────────────────────────────

# Colors (BGR)
C_GREEN  = (80, 220, 100)
C_RED    = (60, 60, 220)
C_GOLD   = (30, 180, 220)
C_WHITE  = (240, 240, 240)
C_BLACK  = (10, 10, 10)
C_TEAL   = (180, 200, 60)
C_BG     = (20, 20, 20)


def draw_hud(frame, left_gesture, right_gesture, left_key, right_key, fps):
    h, w = frame.shape[:2]

    # Semi-transparent dark bar at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 130), (w, h), C_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "TOMB RAIDER GESTURE CONTROL",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_GOLD, 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}",
                (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_WHITE, 1)

    # Divider
    cv2.line(frame, (0, 38), (w, 38), C_GOLD, 1)

    # LEFT panel
    lx = 10
    ly = h - 110
    cv2.putText(frame, "LEFT HAND - MOVE",
                (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEAL, 1)
    color = C_GREEN if left_gesture != "NEUTRAL" else C_WHITE
    cv2.putText(frame, f"Gesture: {left_gesture}",
                (lx, ly + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    key_str = str(left_key).replace("Key.", "").upper() if left_key else "-"
    cv2.putText(frame, f"Key:     [{key_str}]",
                (lx, ly + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WHITE, 1)

    # Divider center
    cv2.line(frame, (w // 2, h - 125), (w // 2, h - 5), C_GOLD, 1)

    # RIGHT panel
    rx = w // 2 + 10
    cv2.putText(frame, "RIGHT HAND - ACTION",
                (rx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEAL, 1)
    color = C_RED if right_gesture != "NEUTRAL" else C_WHITE
    cv2.putText(frame, f"Gesture: {right_gesture}",
                (rx, ly + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    key_str2 = str(right_key).replace("Key.", "").upper() if right_key else "-"
    cv2.putText(frame, f"Key:     [{key_str2}]",
                (rx, ly + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WHITE, 1)

    # Bottom hint
    cv2.putText(frame, "Press Q to quit | Alt+Tab to switch to game",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    return frame


def draw_gesture_guide(frame):
    """Small reference card in top-right corner."""
    h, w = frame.shape[:2]
    guide = [
        ("LEFT HAND",   "", C_TEAL),
        ("Open palm",   "-> FORWARD (W)",   C_WHITE),
        ("Fist",        "-> BACKWARD (S)",  C_WHITE),
        ("Peace sign",  "-> LEFT (A)",      C_WHITE),
        ("3 fingers",   "-> RIGHT (D)",     C_WHITE),
        ("RIGHT HAND",  "", C_TEAL),
        ("Open palm",   "-> JUMP (Space)",  C_WHITE),
        ("Gun shape",   "-> SHOOT (F)",     C_WHITE),
        ("Ring+Pinky",  "-> CROUCH (Ctrl)", C_WHITE),
        ("Fist",        "-> DODGE (Shift)", C_WHITE),
        ("Pinky only",  "-> INTERACT (E)",  C_WHITE),
    ]
    gx = w - 260
    gy = 50
    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (gx - 8, gy - 18), (w - 5, gy + len(guide) * 17 + 4), C_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, (label, action, color) in enumerate(guide):
        text = f"{label}  {action}" if action else label
        cv2.putText(frame, text,
                    (gx, gy + i * 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return frame


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────

def main():
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=CONFIDENCE,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    key_ctrl       = KeyController()
    left_smoother  = GestureSmoother()
    right_smoother = GestureSmoother()

    prev_time = time.time()

    print("\nTomb Raider Gesture Control -- ACTIVE")
    print("Alt+Tab into the game window.")
    print("Press Q in the webcam window to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Webcam not accessible.")
            break

        if FLIP_CAMERA:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        left_gesture  = "NEUTRAL"
        right_gesture = "NEUTRAL"

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                # After flipping, MediaPipe "Right" = user's right hand
                label = hand_info.classification[0].label

                if SHOW_LANDMARKS:
                    mp_draw.draw_landmarks(
                        frame, hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_style.get_default_hand_landmarks_style(),
                        mp_style.get_default_hand_connections_style(),
                    )

                if label == "Left":   # user's LEFT hand after flip
                    raw = classify_left_gesture(hand_lm)
                    left_gesture = left_smoother.update(raw)
                else:
                    raw = classify_right_gesture(hand_lm)
                    right_gesture = right_smoother.update(raw)

        # Resolve keys to press
        desired = set()
        left_key  = LEFT_GESTURE_MAP.get(left_gesture)
        right_key = RIGHT_GESTURE_MAP.get(right_gesture)
        if left_key:
            desired.add(left_key)
        if right_key:
            desired.add(right_key)

        key_ctrl.update(desired)

        # FPS
        now = time.time()
        fps = 1 / max(now - prev_time, 1e-6)
        prev_time = now

        # Draw UI
        frame = draw_gesture_guide(frame)
        frame = draw_hud(frame, left_gesture, right_gesture, left_key, right_key, fps)

        cv2.imshow("Tomb Raider Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    key_ctrl.release_all()
    cap.release()
    cv2.destroyAllWindows()
    print("Gesture control stopped. All keys released.")


if __name__ == "__main__":
    main()
