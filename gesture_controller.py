"""
=============================================================
  Tomb Raider Gesture Controller  — Clean Edition
  Single hand gestures | Head = camera | Voice = ESC/map/inv
=============================================================
GESTURE MAP:
  Open Palm      ->  Move Forward      (Left stick up)
  Hang Loose     ->  Move Backward     (Left stick down)
  Point Left     ->  Strafe Left       (Left stick left)
  Point Right    ->  Strafe Right      (Left stick right)
  Both Hands Up  ->  Jump              (A button)
  Thumb Up       ->  Crouch/Cover      (B button)
  Peace Sign     ->  Interact          (X button)

CAMERA:
  Turn/tilt head ->  Right stick (pan/tilt)
  Auto-recenters after 2 sec of stillness

VOICE COMMANDS:
  "cancel"       ->  ESC key
  "map"          ->  Back/Select button
  "inventory"    ->  Y button

HOTKEYS (webcam window):
  Q              ->  Quit
  R              ->  Recalibrate head
=============================================================
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker
import time, sys, urllib.request, os, threading, ctypes, ctypes.wintypes
import vgamepad as vg
import speech_recognition as sr
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

HOLD_DURATION = 0.35
RELEASE_DELAY = 0.12

HAND_MODEL_PATH = "hand_landmarker.task"
FACE_MODEL_PATH = "face_landmarker.task"
HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
FACE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

STICK_MAX = 32767
STICK_ZERO = 0
STICK_MIN = -32768

HEAD_X_SENSITIVITY  = 7.0
HEAD_Y_SENSITIVITY  = 5.2
HEAD_DEADZONE_X     = 0.010
HEAD_DEADZONE_Y     = 0.008
AUTOCENTER_DELAY    = 2.0
AUTOCENTER_SPEED    = 0.004


# ── ESC key via SendInput ─────────────────────────────────────────────────────

KEYEVENTF_KEYUP    = 0x0002
KEYEVENTF_SCANCODE = 0x0008
VK_ESCAPE          = 0x1B
SC_ESCAPE          = 0x01

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk",ctypes.c_ushort),("wScan",ctypes.c_ushort),
                ("dwFlags",ctypes.c_ulong),("time",ctypes.c_ulong),
                ("dwExtraInfo",ctypes.POINTER(ctypes.c_ulong))]

class INPUT_UNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT)]

class INPUT(ctypes.Structure):
    _fields_ = [("type",ctypes.c_ulong),("ii",INPUT_UNION)]

def send_esc():
    """Press and release the ESC key via SendInput."""
    def _do():
        extra = ctypes.c_ulong(0)
        for up in [False, True]:
            flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
            ki  = KEYBDINPUT(0, SC_ESCAPE, flags, 0, ctypes.pointer(extra))
            inp = INPUT(1, INPUT_UNION(ki=ki))
            ctypes.windll.user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))
            time.sleep(0.05)
    threading.Thread(target=_do, daemon=True).start()

# ── Virtual gamepad ───────────────────────────────────────────────────────────

gamepad = vg.VX360Gamepad()
left_x = left_y = right_x = right_y = STICK_ZERO

def clamp(v): return max(STICK_MIN, min(STICK_MAX, int(v)))

def push():
    gamepad.left_joystick( x_value=clamp(left_x),  y_value=clamp(left_y))
    gamepad.right_joystick(x_value=clamp(right_x), y_value=clamp(right_y))
    gamepad.update()

def reset_all():
    global left_x, left_y, right_x, right_y
    left_x = left_y = right_x = right_y = STICK_ZERO
    gamepad.left_joystick(x_value=STICK_ZERO, y_value=STICK_ZERO)
    gamepad.right_joystick(x_value=STICK_ZERO, y_value=STICK_ZERO)
    for btn in [vg.XUSB_BUTTON.XUSB_GAMEPAD_A, vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
                vg.XUSB_BUTTON.XUSB_GAMEPAD_X, vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
                vg.XUSB_BUTTON.XUSB_GAMEPAD_START, vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK]:
        gamepad.release_button(button=btn)
    gamepad.update()

def tap_button(btn, duration=0.15):
    def _tap():
        gamepad.press_button(button=btn)
        gamepad.update()
        time.sleep(duration)
        gamepad.release_button(button=btn)
        gamepad.update()
    threading.Thread(target=_tap, daemon=True).start()

def apply_gesture(gesture):
    global left_x, left_y
    left_x = left_y = STICK_ZERO
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)

    if   gesture == "open_palm":   left_y =  STICK_MAX
    elif gesture == "move_backward": left_y =  STICK_MIN
    elif gesture == "point_left":  left_x =  STICK_MIN
    elif gesture == "point_right": left_x =  STICK_MAX
    elif gesture == "jump":        tap_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    elif gesture == "crouch":      gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
    elif gesture == "interact":    gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
    push()

def release_gesture():
    global left_x, left_y
    left_x = left_y = STICK_ZERO
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
    push()

# ── Model download ────────────────────────────────────────────────────────────

def ensure_model(path, url):
    if not os.path.exists(path):
        print(f"  Downloading {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)
        print("  Done.\n")

# ── Gesture detection ─────────────────────────────────────────────────────────

def finger_states(lm):
    tips = [4, 8, 12, 16, 20]
    f = [lm[4].y < lm[2].y]
    for i in range(1, 5):
        f.append(lm[tips[i]].y < lm[tips[i]-1].y)
    return f

def classify_gesture(result):
    if not result or not result.hand_landmarks:
        return "none"
    lms = result.hand_landmarks

    if len(lms) >= 2:
        counts = [sum(finger_states(lm)) for lm in lms]
        if min(counts) >= 3:
            return "jump"

    lm    = lms[0]
    f     = finger_states(lm)
    thumb, index, middle, ring, pinky = f
    total = sum(f)

    if total >= 4:   return "open_palm"

    # Thumb only (up) = crouch
    if not index and not middle and not ring and not pinky:
        return "crouch"

    # Hang loose: thumb + pinky only = move backward
    if thumb and not index and not middle and not ring and pinky:
        return "move_backward"

    if index and middle and not ring and not pinky:  return "interact"
    if index and not middle and not ring and not pinky:
        return "point_left" if (lm[8].x - lm[0].x) > 0 else "point_right"
    return "none"

# ── Head tracking ─────────────────────────────────────────────────────────────

nose_cx = 0.5
nose_cy = 0.5
last_move_time = 0.0

def update_head(face_result, now):
    global right_x, right_y, nose_cx, nose_cy, last_move_time
    if not face_result or not face_result.face_landmarks:
        right_x = right_y = STICK_ZERO
        return STICK_ZERO, STICK_ZERO

    nose = face_result.face_landmarks[0][1]
    dx   = nose.x - nose_cx
    dy   = nose.y - nose_cy

    moved = abs(dx) > HEAD_DEADZONE_X * 1.5 or abs(dy) > HEAD_DEADZONE_Y * 1.5
    if moved:
        last_move_time = now
    elif (now - last_move_time) > AUTOCENTER_DELAY:
        nose_cx += (nose.x - nose_cx) * AUTOCENTER_SPEED
        nose_cy += (nose.y - nose_cy) * AUTOCENTER_SPEED
        dx = nose.x - nose_cx
        dy = nose.y - nose_cy

    if abs(dx) < HEAD_DEADZONE_X: dx = 0.0
    if abs(dy) < HEAD_DEADZONE_Y: dy = 0.0

    rx = clamp( dx * HEAD_X_SENSITIVITY * STICK_MAX)
    ry = clamp(-dy * HEAD_Y_SENSITIVITY * STICK_MAX)
    right_x = rx
    right_y = ry
    return rx, ry

# ── Voice commands ────────────────────────────────────────────────────────────

voice_status   = "Starting..."
last_voice_cmd = ""
voice_lock     = threading.Lock()

def voice_thread():
    global voice_status, last_voice_cmd
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 400
    recognizer.dynamic_energy_threshold = True
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    with voice_lock:
        voice_status = "Listening..."
    print("  Voice ready. Say: cancel / map / inventory\n")

    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=2)
            text = recognizer.recognize_google(audio).lower().strip()
            with voice_lock:
                voice_status = f'Heard: "{text}"'
                last_voice_cmd = ""

            if "cancel" in text:
                send_esc()
                with voice_lock: last_voice_cmd = "CANCEL (ESC)"
                print("  VOICE: cancel -> ESC")
            elif "map" in text:
                tap_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK)
                with voice_lock: last_voice_cmd = "MAP"
                print("  VOICE: map -> Back")
            elif "inventory" in text:
                tap_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
                with voice_lock: last_voice_cmd = "INVENTORY"
                print("  VOICE: inventory -> Y")

        except sr.WaitTimeoutError:
            with voice_lock: voice_status = "Listening..."
        except sr.UnknownValueError:
            with voice_lock: voice_status = "Listening..."
        except Exception as e:
            with voice_lock: voice_status = f"Mic error"
            time.sleep(1)

# ── Drawing ───────────────────────────────────────────────────────────────────

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]

GESTURE_DISPLAY = {
    "open_palm":   "Forward  (stick up)",
    "move_backward": "Hang Loose -> Backward",
    "point_left":  "Strafe Left",
    "point_right": "Strafe Right",
    "jump":        "Jump (A)",
    "crouch":      "Crouch (B)",
    "interact":    "Interact (X)",
    "none":        "---",
}

def put(img, text, x, y, color, scale=0.48, thick=1):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_hands(frame, result):
    if not result or not result.hand_landmarks: return
    h, w = frame.shape[:2]
    for hand_lms in result.hand_landmarks:
        pts = [(int(lm.x*w), int(lm.y*h)) for lm in hand_lms]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0,200,100), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (255,255,255), -1)
            cv2.circle(frame, pt, 4, (0,180,80), 1)

def draw_nose(frame, face_result):
    if not face_result or not face_result.face_landmarks: return
    h, w = frame.shape[:2]
    nose = face_result.face_landmarks[0][1]
    nx, ny = int(nose.x*w), int(nose.y*h)
    cx, cy = int(nose_cx*w), int(nose_cy*h)
    cv2.circle(frame, (cx,cy), 7, (60,60,60), 1)
    cv2.circle(frame, (nx,ny), 5, (0,180,255), -1)
    cv2.line(frame, (cx,cy), (nx,ny), (0,140,220), 1)

def draw_main_overlay(frame, gesture, active, hold_progress, rx, ry, calibrating):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,110), (8,8,16), -1)
    cv2.addWeighted(ov, 0.62, frame, 0.38, 0, frame)

    if calibrating:
        put(frame, "CALIBRATING -- hold head still...", 14, 18, (0,220,230))
    else:
        put(frame, f"Camera  X={int(rx/327):+d}%  Y={int(ry/327):+d}%  [auto-recenter]", 14, 18, (0,170,255))

    g_col = (0,230,80) if gesture != "none" else (120,120,120)
    put(frame, f"Gesture:  {GESTURE_DISPLAY.get(gesture,'---')}", 14, 38, g_col)

    if active and active != "none":
        put(frame, f"ACTIVE:   {GESTURE_DISPLAY.get(active,'---')}", 14, 56, (0,230,240), thick=2)

    with voice_lock:
        vs = voice_status
        vc = last_voice_cmd
    vc_col = (220,190,60) if vc else (100,90,50)
    put(frame, f"Voice: {vc if vc else vs}", 14, 74, vc_col)

    bx, by, bw, bh = 14, 86, 240, 9
    cv2.rectangle(frame, (bx,by), (bx+bw, by+bh), (40,40,40), -1)
    fill = int(bw * hold_progress)
    if fill > 0:
        cv2.rectangle(frame, (bx,by), (bx+fill, by+bh), (0,200,80), -1)
    put(frame, "Hold", bx+bw+6, by+8, (90,90,90), 0.34)
    put(frame, "Q=quit  R=recalibrate", 14, h-10, (70,70,70), 0.36)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global nose_cx, nose_cy, last_move_time

    print("\n"+"="*58)
    print("  Tomb Raider Gesture Controller — Clean Edition")
    print("="*58+"\n")

    ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL)
    ensure_model(FACE_MODEL_PATH, FACE_MODEL_URL)

    hand_landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ))

    face_landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not found."); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    reset_all()

    threading.Thread(target=voice_thread, daemon=True).start()

    # Calibration
    calibrating  = True
    calib_frames = 0
    calib_x = calib_y = 0.0
    CALIB_TARGET = 60

    pending_gesture    = "none"
    gesture_start_time = 0.0
    active_gesture     = "none"
    last_active_time   = 0.0
    rx = ry            = STICK_ZERO
    last_move_time     = time.time()

    print("  Look straight at the camera and hold still...")
    print("  Calibrating head position (2 sec)...\n")


    while True:
        ret, frame = cap.read()
        if not ret: continue

        frame     = cv2.flip(frame, 1)
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int(time.time() * 1000)
        now       = time.time()

        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp)
        face_result = face_landmarker.detect_for_video(mp_image, timestamp)

        # Calibration
        if calibrating:
            if face_result and face_result.face_landmarks:
                nose = face_result.face_landmarks[0][1]
                calib_x += nose.x; calib_y += nose.y; calib_frames += 1
            if calib_frames >= CALIB_TARGET:
                nose_cx = calib_x / calib_frames
                nose_cy = calib_y / calib_frames
                calibrating = False
                last_move_time = now
                print(f"  Calibrated! ({nose_cx:.3f}, {nose_cy:.3f})")
                print("  Ready! Show a gesture or turn your head.\n")

        # Head
        if not calibrating:
            rx, ry = update_head(face_result, now)

        # Gesture state machine
        detected = classify_gesture(hand_result)

        if detected != pending_gesture:
            pending_gesture    = detected
            gesture_start_time = now

        hold_elapsed  = now - gesture_start_time
        hold_progress = min(hold_elapsed / HOLD_DURATION, 1.0)

        if hold_elapsed >= HOLD_DURATION and detected != "none":
            if detected != active_gesture:
                active_gesture = detected
                apply_gesture(detected)
                print(f"  GESTURE: {detected}")
            last_active_time = now
        elif detected == "none":
            if (now - last_active_time) > RELEASE_DELAY and active_gesture != "none":
                active_gesture = "none"
                release_gesture()
                print("  GESTURE: released")

        push()

        # Draw
        draw_hands(frame, hand_result)
        draw_nose(frame, face_result)
        draw_main_overlay(frame, detected, active_gesture, hold_progress, rx, ry, calibrating)

        cv2.imshow("Gesture Controller -- Tomb Raider", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n  Quitting..."); break
        elif key == ord('r'):
            calibrating = True; calib_frames = 0; calib_x = calib_y = 0.0
            print("  Recalibrating -- look straight at camera...")

    reset_all()
    cap.release()
    hand_landmarker.close()
    face_landmarker.close()
    cv2.destroyAllWindows()
    print("  Done.\n")

if __name__ == "__main__":
    main()
