import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import serial
import time
import threading

# Serial communication settings
serial_port = '/dev/ttymxc3'
baud_rate = 115200

GESTURE_LABELS = {
    0: "Unknown",
    1: "Closed_Fist",
    2: "Open_Palm",
    3: "Pointing_Up",
    4: "Thumb_Down",
    5: "Thumb_Up",
    6: "Victory",
    7: "ILoveYou"
}

COMMANDS = {
    "Open_Palm": {"T": 132, "IO4": 255, "IO5": 255},  # Turn on LED
    "Closed_Fist": {"T": 132, "IO4": 0, "IO5": 0},    # Turn off LED
    "ILoveYou": {"T": 133, "X": -90, "Y": -30, "SPD": 0, "ACC": 0},  # Shy (turn left, look down)
    "Thumb_Up": {"T": 133, "X": -30, "Y": 180, "SPD": 0, "ACC": 0},  # all up
    "Thumb_Down": {"T": 133, "X": -30, "Y": -30, "SPD": 0, "ACC": 0}, # all down
    "Victory": {"T": 133, "X": 180, "Y": 0, "SPD": 0, "ACC": 0} # round 180
}

ZERO_COMMAND = {"T": 133, "X": 0, "Y": 0, "SPD": 0, "ACC": 0}

# Initialize the gesture recognition model
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

# Initialize serial communication
ser = serial.Serial(serial_port, baud_rate, timeout=1)

def send_serial_command(command):
    json_command = json.dumps(command)
    ser.write(f"{json_command}\n".encode())
    time.sleep(0.1)  # Small delay to ensure command is sent

def send_zero_command():
    send_serial_command(ZERO_COMMAND)
    print("Zero command sent")

try:
    while cap.isOpened():
        # Read a frame from the camera
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from camera.")
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Recognize gestures
        recognition_result = recognizer.recognize(image)

        # Process the result
        if recognition_result.gestures:
            top_gesture = recognition_result.gestures[0][0]
            print(f"Detected gesture: {top_gesture.category_name}")

            if top_gesture.category_name in COMMANDS:
                gesture_command = COMMANDS[top_gesture.category_name]
                try:
                    send_serial_command(gesture_command)
                    print(f"Command sent: {gesture_command}")
                    
                    # For specific gestures, schedule a zero command after 2 seconds
                    if top_gesture.category_name in ["ILoveYou", "Thumb_Up", "Thumb_Down", "Victory"]:
                        threading.Timer(2.0, send_zero_command).start()
                        
                except Exception as e:
                    print(f'Failed to send gesture command: {e}')

        # Exit on key press
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Release the camera and close serial connection
    cap.release()
    ser.close()
