import cv2
import mediapipe as mp
import serial
import json
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Serial communication setup
serial_port = '/dev/ttymxc3'
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate, timeout=1)

# Gesture labels
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

# Initialize the hand tracking model
hands = mp.solutions.hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    static_image_mode=False
)

# Gesture Recognizer setup
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

def send_serial_command(command):
    try:
        ser.write((json.dumps(command) + '\n').encode('utf-8'))
        print("Command sent:", command)
    except serial.SerialException as e:
        print(f'Failed to send command: {e}')

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame from camera.")
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using MediaPipe Hands
    results = hands.process(rgb_frame)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    recognition_result = recognizer.recognize(image)

    # Process the gesture recognition result
    if recognition_result.gestures:
        top_gesture = recognition_result.gestures[0][0].category_name
        print("Recognized Gesture:", top_gesture)

        # Send corresponding command via serial
        if top_gesture in COMMANDS:
            send_serial_command(COMMANDS[top_gesture])

            # Send the zero command after a delay for specific gestures
            if top_gesture in ["ILoveYou", "Thumb_Up", "Thumb_Down", "Victory"]:
                def send_zero_command():
                    send_serial_command(ZERO_COMMAND)
                
                # Start a timer to send the zero command after 2 seconds
                threading.Timer(2.0, send_zero_command).start()

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
        break

# Release resources
cap.release()
ser.close()
cv2.destroyAllWindows()
