import cv2
import time
import numpy as np
import pygame
import qrcode
from PIL import Image
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import os # Import the os module to handle paths
import math # Import for sine wave calculation

# --- INITIALIZATION ---

# 1. State Machine and Global Variables
STATE = "WAITING_FOR_PERSON"
PERSON_CONFIDENCE_THRESHOLD = 0.6
PERSON_PRESENCE_TIME_THRESHOLD = 1.0  # seconds
YOUR_CLUB_WEBSITE_URL = "https://www.yourclubwebsite.com" # <--- IMPORTANT: Change this URL
latest_gesture_result = None

# --- IMPORTANT: Download this model from MediaPipe's website ---
# https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/index#models
MODEL_PATH = "gesture_recognizer.task"

# 2. YOLO Model for Person Detection
print("Loading YOLO model...")
try:
    yolo_model = YOLO("yolov8n.pt")  # Uses a small, fast version of YOLOv8
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# 3. MediaPipe Drawing Utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 4. Pygame for Audio Playback
print("Initializing Pygame Mixer...")
try:
    pygame.mixer.init()
    audio_folder = "robot_audio"
    greeting_sound = pygame.mixer.Sound(os.path.join(audio_folder, "greeting.wav"))
    soda_sound = pygame.mixer.Sound(os.path.join(audio_folder, "about_soda.wav"))
    qr_prompt_sound = pygame.mixer.Sound(os.path.join(audio_folder, "qr_show.wav"))
    print("Pygame Mixer initialized.")
except Exception as e:
    print(f"Error initializing Pygame or loading sound files: {e}")
    exit()

# 5. QR Code Generation
print("Generating QR Code...")
qr_code_obj = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L)
qr_code_obj.add_data(YOUR_CLUB_WEBSITE_URL)
qr_code_obj.make(fit=True)
qr_img_pil = qr_code_obj.make_image(fill_color="black", back_color="white").convert('RGB')
qr_img_pil = qr_img_pil.resize((200, 200))
qr_img_cv = cv2.cvtColor(np.array(qr_img_pil), cv2.COLOR_RGB2BGR)
print("QR Code generated.")

# --- HELPER FUNCTIONS ---

def process_gesture_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    """
    Callback function to receive and process gesture recognition results.
    """
    global latest_gesture_result, STATE
    latest_gesture_result = result

    if result.gestures:
        # Don't process gestures if audio is playing to avoid accidental triggers
        if pygame.mixer.get_busy():
            return

        top_gesture = result.gestures[0][0]
        gesture_name = top_gesture.category_name
        
        # --- State Transition Logic based on Gestures ---
        if STATE == "GREETING" and gesture_name == "Thumb_Up":
            print("Thumbs up detected! Explaining soda.")
            STATE = "EXPLAINING"
            soda_sound.play()
        
        elif STATE == "EXPLAINING" and gesture_name == "Victory":
            print("Victory (peace) gesture detected! Showing QR code.")
            STATE = "SHOWING_QR"
            qr_prompt_sound.play() # Play sound when QR is shown

# --- MAIN APPLICATION ---

# Configure Gesture Recognizer
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=process_gesture_result
)

# Initialize camera and windows
print("Starting camera feed...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

VOICE_WINDOW_NAME = "Robot Voice Visualizer"
cv2.namedWindow(VOICE_WINDOW_NAME)
voice_window_size = (300, 300)

person_detected_time = None

# Create the recognizer and run the main loop
with vision.GestureRecognizer.create_from_options(options) as recognizer:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # --- Voice Visualizer Logic ---
        visualizer_frame = np.zeros((voice_window_size[1], voice_window_size[0], 3), dtype=np.uint8)
        if pygame.mixer.get_busy():
            pulse = (math.sin(time.time() * 10) + 1) / 2
            center = (voice_window_size[0] // 2, voice_window_size[1] // 2)
            max_radius = int(voice_window_size[0] * 0.4)
            for i in range(3):
                radius = int(max_radius * (pulse + i * 0.3) % max_radius)
                color = (150 + i * 30, 0, 150 - i * 30)
                cv2.circle(visualizer_frame, center, radius, color, 2)

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # --- Asynchronous Gesture Recognition ---
        timestamp_ms = int(time.time() * 1000)
        recognizer.recognize_async(mp_image, timestamp_ms)

        # Make frame writeable for drawing
        frame.flags.writeable = True

        # --- State: WAITING_FOR_PERSON / PERSON_DETECTED ---
        if STATE == "WAITING_FOR_PERSON" or STATE == "PERSON_DETECTED":
            yolo_results = yolo_model(rgb_frame, classes=[0], verbose=False)
            person_in_frame = False
            for detection in yolo_results[0].boxes:
                if detection.conf.item() > PERSON_CONFIDENCE_THRESHOLD:
                    box = detection.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                    if (x2 - x1) > frame_width * 0.2 and (y2 - y1) > frame_height * 0.4:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        person_in_frame = True
                        if STATE == "WAITING_FOR_PERSON":
                            STATE = "PERSON_DETECTED"
                            person_detected_time = time.time()
                        if time.time() - (person_detected_time or 0) > PERSON_PRESENCE_TIME_THRESHOLD:
                            STATE = "GREETING"
                            greeting_sound.play()
                        break
            if not person_in_frame:
                STATE = "WAITING_FOR_PERSON"
                person_detected_time = None

        # --- State: GREETING (Waiting for Thumbs Up) ---
        elif STATE == "GREETING":
            if not pygame.mixer.get_busy():
                cv2.putText(frame, "Show me a THUMBS UP!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # --- State: EXPLAINING (Waiting for Peace Sign) ---
        elif STATE == "EXPLAINING":
            if not pygame.mixer.get_busy():
                cv2.putText(frame, "Show me a PEACE sign!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # --- State: SHOWING_QR ---
        elif STATE == "SHOWING_QR":
            qr_h, qr_w, _ = qr_img_cv.shape
            frame[10:10+qr_h, frame_width-qr_w-10:frame_width-10] = qr_img_cv
            cv2.putText(frame, "Scan for our website!", (frame_width - 250, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if 'qr_start_time' not in locals(): qr_start_time = time.time()
            if time.time() - qr_start_time > 15:
                STATE = "WAITING_FOR_PERSON"
                del qr_start_time
        
        # --- Draw Hand Landmarks on Frame ---
        if latest_gesture_result:
            for hand_landmarks in latest_gesture_result.hand_landmarks:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                ])
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                )

        # Display the final frames
        cv2.imshow('Robot Interaction View', frame)
        cv2.imshow(VOICE_WINDOW_NAME, visualizer_frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

# --- CLEANUP ---
print("Cleaning up and closing.")
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
