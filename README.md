# 🎮 Tomb Raider Gesture Controller

A computer-vision-based experimental controller for **Rise of the Tomb Raider**, using a webcam to recognize hand gestures and head movement, with optional voice commands.

> **Project status:** Archived / Completed  
> This is an older project. The original game installation, screenshots, and demonstration recordings are no longer available. This repository preserves the source code and documentation of the implementation.

## ✨ Features

- ✋ Hand-gesture-based game controls
- 🎮 Virtual Xbox 360 controller input
- 🧑 Head tracking for right-stick camera movement
- 🎤 Voice commands
- 🎯 Gesture hold/release logic to reduce accidental inputs
- 🔄 Automatic head-position recentering
- 📷 Real-time webcam visualization
- 🛠️ Separate gesture calibration/testing program

## 🧠 How It Works

```text
                 Webcam
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
    Hand Detection        Face Detection
          │                   │
          ↓                   ↓
 Gesture Recognition     Head Tracking
          │                   │
          └─────────┬─────────┘
                    ↓
             Virtual Gamepad
                    │
                    ↓
        Rise of the Tomb Raider

Microphone → Speech Recognition → Game Commands
```

The main controller uses MediaPipe hand and face landmark detection and sends gamepad input through `vgamepad`.

## ✋ Gesture Controls

| Gesture | Action |
|---|---|
| Open Palm | Move Forward |
| Hang Loose | Move Backward |
| Point Left | Strafe Left |
| Point Right | Strafe Right |
| Both Hands Up | Jump |
| Thumb Up | Crouch / Cover |
| Peace Sign | Interact |

## 🧑 Head Controls

Head movement is mapped to the virtual controller's right analog stick.

- Turn/tilt head → camera movement
- Head position is calibrated when the program starts
- The camera automatically recenters after a period of stillness
- Press `R` to recalibrate

## 🎤 Voice Commands

| Voice command | Action |
|---|---|
| `cancel` | ESC |
| `map` | Back button |
| `inventory` | Y button |

## 📁 Files

### `gesture_controller.py`
Main implementation. Handles:

- MediaPipe hand tracking
- MediaPipe face tracking
- Gesture classification
- Virtual Xbox controller input
- Head tracking
- Voice commands
- Calibration
- Webcam overlay

### `calibrate.py`
A testing utility for checking whether gestures are recognized correctly.

**It does not send game keypresses.**

### `gesture_control_legacy.py`
Earlier keyboard-based implementation using `pynput`. It is retained as a legacy version for reference.

## ⚙️ Requirements

The project was developed for a Windows environment and uses:

- Python
- OpenCV
- MediaPipe
- NumPy
- vgamepad
- SpeechRecognition
- PyAudio

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

> `vgamepad` and the Windows virtual-gamepad setup may require additional system configuration. This repository documents the Python implementation; the original development environment is no longer available.

## 🚀 Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Test gestures first

```bash
python calibrate.py
```

This opens the webcam and displays the detected gesture without sending game inputs.

### 3. Run the main controller

```bash
python gesture_controller.py
```

The main program downloads the required MediaPipe hand and face landmark model files automatically if they are not already present.

### 4. Start the game

The controller was intended to be run before switching to the game window.

## ⌨️ Hotkeys

| Key | Function |
|---|---|
| `Q` | Quit |
| `R` | Recalibrate head position |

## 🛠️ Technologies

- **Python** — application logic
- **OpenCV** — webcam capture and visualization
- **MediaPipe** — hand and face landmark detection
- **NumPy** — numerical processing
- **vgamepad** — virtual Xbox controller input
- **SpeechRecognition** — voice command recognition

## 📌 Notes

This repository contains only the project's source code. It does **not** contain:

- Rise of the Tomb Raider game files
- Copyrighted game assets
- Screenshots or videos from the original project
- MediaPipe model files

The game itself must be obtained and installed separately by the user.

## 📜 Project Background

The project was created as an experiment in combining **computer vision, gesture recognition, head tracking, voice recognition, and virtual controller input** to create an alternative way of interacting with a video game.

