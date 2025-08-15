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
import os
import math
import json
import random
import wave
import hashlib

# --- TTS Class for On-the-Fly Audio Generation ---
# NOTE: You must set the 'GEMINI_API_KEY' environment variable for this to work.
try:
    # Corrected import based on the new library structure
    from google import genai
    from google.genai import types
    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False

class RobotTTS:
    def __init__(self):
        if not IMPORT_SUCCESS:
            raise ImportError("Could not import google.genai. Please run 'pip install google-genai'")
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # The client is used for the TTS model
        self.client = genai.Client(api_key=api_key)
        
        self.audio_dir = "robot_audio"
        os.makedirs(self.audio_dir, exist_ok=True)
        
        self.voice_name = 'Puck' # Default voice

    def _get_audio_filename(self, text_or_name):
        if ' ' in text_or_name:
            text_hash = hashlib.md5(text_or_name.encode()).hexdigest()
            return os.path.join(self.audio_dir, f"{text_hash}.wav")
        return os.path.join(self.audio_dir, f"{text_or_name}.wav")

    def _save_wave_file(self, filename, pcm_data):
        try:
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(pcm_data)
            return True
        except Exception as e:
            print(f"ERROR: Failed to save audio to {filename}: {e}")
            return False

    def generate_speech(self, text, style="", filename_override=None):
        audio_filename = self._get_audio_filename(filename_override or text)
        if os.path.exists(audio_filename):
            return audio_filename
        
        try:
            print(f"Generating new audio for: {filename_override or text[:20]}...")
            prompt = f"Say {style}: {text}" if style else text
            
            # --- CORRECTED API CALL FOR TTS based on new documentation ---
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=self.voice_name,
                            )
                        )
                    ),
                ))
            
            audio_data = response.candidates[0].content.parts[0].inline_data.data
            if self._save_wave_file(audio_filename, audio_data):
                return audio_filename
        except Exception as e:
            print(f"ERROR generating speech for '{text[:20]}...': {e}")
        return None

# --- INITIALIZATION ---

# 1. State Machine and Global Variables
STATE = "WAITING_FOR_PERSON"
PERSON_CONFIDENCE_THRESHOLD = 0.6
PERSON_PRESENCE_TIME_THRESHOLD = 1.0
YOUR_CLUB_WEBSITE_URL = "https://www.yourclubwebsite.com"
latest_gesture_result = None
MODEL_PATH = "gesture_recognizer.task"
last_person_seen_time = time.time()
PERSON_RESET_TIMEOUT = 2.0 # seconds

# Quiz game variables
quiz_questions = []
current_question = None
answered_questions = set()
skip_available = True
hovered_option = -1
hover_start_time = None
SELECTION_LOCK_DURATION = 3.0 # 3 seconds to lock answer by pointing
user_is_winner = False

# Subtitle variables
SUBTITLES = {}
current_subtitle = ""

# --- Load Quiz Questions ---
try:
    with open('questions.json', 'r') as f:
        quiz_questions = json.load(f)['questions']
    print(f"Loaded {len(quiz_questions)} quiz questions.")
except Exception as e:
    print(f"Error loading questions.json: {e}"); exit()

# 2. Initialize TTS
try:
    tts = RobotTTS()
except Exception as e:
    print(f"Failed to initialize TTS: {e}"); exit()

# --- HELPER & GAME FUNCTIONS ---
def pregenerate_static_audio():
    """Generates all necessary static audio files at startup and populates the subtitle dictionary."""
    print("Pre-generating static audio files if they don't exist...")
    
    audio_map = {
        "greeting": ("Hey there! Nice to meet you! Give me a thumbs up to learn about SoDA!", "cheerfully"),
        "about_soda": ("SoDA is the Software Development Association! We build cool projects and learn together.", "enthusiastically"),
        "game_request": ("Would you like to answer a question for a potential prize? Show thumbs up for yes, or thumbs down for no.", "playfully"),
        "skip_quiz_prompt": ("No problem! Show me a peace sign to get our QR code instead.", "calmly"),
        "qr_show": ("Awesome! Here's how to join us!", "happily"),
        "correct_answer": ("Correct! You win! You can collect your prize later.", "excitedly"),
        "wrong_answer": ("Aww, that's not right. Better luck next time.", "gently"),
        "qr_prompt_after_quiz": ("Show me a peace sign if you'd like to know more about us.", "invitingly"),
        "goodbye": ("Thanks for playing! Goodbye!", "friendly")
    }

    for name, (text, style) in audio_map.items():
        tts.generate_speech(text, style, name)
        SUBTITLES[name] = text # Store text for subtitles
    
    print("Static audio ready.")

def play_audio_by_name(filename):
    global current_subtitle
    filepath = os.path.join(tts.audio_dir, f"{filename}.wav")
    if os.path.exists(filepath):
        pygame.mixer.Sound(filepath).play()
        current_subtitle = SUBTITLES.get(filename, "")
    else:
        print(f"ERROR: Audio file not found: {filepath}")

def play_dynamic_audio(text, style=""):
    # Dynamic audio (like questions) won't have subtitles for now to keep the screen clean.
    # This could be changed by setting current_subtitle here if desired.
    filepath = tts.generate_speech(text, style)
    if filepath:
        pygame.mixer.Sound(filepath).play()

def get_new_question():
    global current_question, answered_questions
    available_q = [q for q in quiz_questions if q['id'] not in answered_questions]
    if not available_q:
        answered_questions = set()
        available_q = quiz_questions
        play_dynamic_audio("You've answered all the questions! Let's start over.", "excitedly")
    current_question = random.choice(available_q)

def reset_game_state():
    global STATE, current_question, answered_questions, skip_available, user_is_winner, hovered_option, hover_start_time, person_detected_time, current_subtitle
    print("Resetting game state...")
    STATE = "WAITING_FOR_PERSON"
    current_question = None
    answered_questions = set()
    skip_available = True
    user_is_winner = False
    hovered_option = -1
    hover_start_time = None
    person_detected_time = None
    current_subtitle = ""
    pygame.mixer.stop()

def draw_subtitles(frame, text):
    """Draws text with a background at the bottom of the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Position at the bottom center
    x = (frame.shape[1] - text_width) // 2
    y = frame.shape[0] - 30
    
    # Draw background rectangle
    cv2.rectangle(frame, (x - 10, y - text_height - 10), (x + text_width + 10, y + baseline), bg_color, -1)
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)

def process_gesture_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_gesture_result, STATE
    latest_gesture_result = result
    if pygame.mixer.get_busy() or not result.gestures: return

    gesture_name = result.gestures[0][0].category_name
    
    if STATE == "GREETING" and gesture_name == "Thumb_Up":
        STATE = "EXPLAINING"
        play_audio_by_name("about_soda")
    elif STATE == "AWAITING_QUIZ_CHOICE":
        if gesture_name == "Thumb_Up":
            STATE = "QUIZ_MODE"
            get_new_question()
        elif gesture_name == "Thumb_Down":
            STATE = "PROMPT_FOR_QR"
            play_audio_by_name("skip_quiz_prompt")
    elif STATE == "PROMPT_FOR_QR" and gesture_name == "Victory":
        STATE = "SHOWING_QR"
        play_audio_by_name("qr_show")

# --- MAIN APPLICATION SETUP ---
pregenerate_static_audio()
yolo_model = YOLO("yolov8n.pt")
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
pygame.mixer.init()
qr_code_obj = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L)
qr_code_obj.add_data(YOUR_CLUB_WEBSITE_URL)
qr_code_obj.make(fit=True)
qr_img_pil = qr_code_obj.make_image(fill_color="black", back_color="white").convert('RGB').resize((200, 200))
qr_img_cv = cv2.cvtColor(np.array(qr_img_pil), cv2.COLOR_RGB2BGR)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.GestureRecognizerOptions(base_options=base_options, running_mode=vision.RunningMode.LIVE_STREAM, num_hands=2, result_callback=process_gesture_result)

print("Starting camera feed...")
cap = cv2.VideoCapture(0)
VOICE_WINDOW_NAME = "Robot Voice Visualizer"
cv2.namedWindow(VOICE_WINDOW_NAME)
voice_window_size = (300, 300)

with vision.GestureRecognizer.create_from_options(options) as recognizer:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        if time.time() - last_person_seen_time > PERSON_RESET_TIMEOUT and STATE != "WAITING_FOR_PERSON":
            reset_game_state()

        visualizer_frame = np.zeros((voice_window_size[1], voice_window_size[0], 3), dtype=np.uint8)
        if pygame.mixer.get_busy():
            pulse = (math.sin(time.time() * 10) + 1) / 2
            center = (voice_window_size[0] // 2, voice_window_size[1] // 2)
            max_radius = int(voice_window_size[0] * 0.4)
            for i in range(3):
                radius = int((max_radius * (pulse + i * 0.3)) % max_radius)
                color = (150 + i * 30, 0, 150 - i * 30)
                cv2.circle(visualizer_frame, center, radius, color, 2)

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        recognizer.recognize_async(mp_image, int(time.time() * 1000))
        frame.flags.writeable = True

        person_in_frame = False
        yolo_results = yolo_model(rgb_frame, classes=[0], verbose=False, max_det=1)
        if any(d.conf.item() > PERSON_CONFIDENCE_THRESHOLD for d in yolo_results[0].boxes):
            person_in_frame = True
            last_person_seen_time = time.time()
        
        if STATE in ["WAITING_FOR_PERSON", "PERSON_DETECTED"]:
            if person_in_frame:
                if STATE == "WAITING_FOR_PERSON":
                    STATE = "PERSON_DETECTED"
                    person_detected_time = time.time()
                if time.time() - (person_detected_time or 0) > PERSON_PRESENCE_TIME_THRESHOLD:
                    STATE = "GREETING"
                    play_audio_by_name("greeting")
            elif STATE == "PERSON_DETECTED":
                STATE = "WAITING_FOR_PERSON"
        
        if STATE == "GREETING" and not pygame.mixer.get_busy():
            cv2.putText(frame, "Show me a THUMBS UP!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif STATE == "EXPLAINING" and not pygame.mixer.get_busy():
            STATE = "AWAITING_QUIZ_CHOICE"
            play_audio_by_name("game_request")
        elif STATE == "AWAITING_QUIZ_CHOICE" and not pygame.mixer.get_busy():
            cv2.putText(frame, "Quiz? Thumbs UP (Yes) or DOWN (No)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        elif STATE == "AWAITING_FEEDBACK_END" and not pygame.mixer.get_busy():
            STATE = "PROMPT_FOR_QR"
            play_audio_by_name("qr_prompt_after_quiz")
        elif STATE == "PROMPT_FOR_QR" and not pygame.mixer.get_busy():
            cv2.putText(frame, "Show me a PEACE sign for the QR Code!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        elif STATE == "SHOWING_QR":
            frame[10:210, frame_width-210:frame_width-10] = qr_img_cv
            if 'qr_start_time' not in locals(): qr_start_time = time.time()
            if time.time() - qr_start_time > 8 and not pygame.mixer.get_busy():
                play_audio_by_name("goodbye")
                time.sleep(2) # Let goodbye message play
                reset_game_state()
                del qr_start_time
        elif STATE == "QUIZ_MODE":
            if current_question and not pygame.mixer.get_busy():
                cv2.putText(frame, current_question['question'], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                option_boxes = []
                for i, option in enumerate(current_question['options']):
                    y_pos = 100 + i * 60; box = (50, y_pos, 550, y_pos + 50); option_boxes.append(box)
                    color = (0, 255, 0) if hovered_option == i else (255, 100, 0)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(frame, f"{i+1}. {option}", (60, y_pos + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
                currently_pointing_at = -1
                if latest_gesture_result and latest_gesture_result.hand_landmarks:
                    index_tip = latest_gesture_result.hand_landmarks[0][mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    px, py = int(index_tip.x * frame_width), int(index_tip.y * frame_height)
                    
                    for i, box in enumerate(option_boxes):
                        if box[0] < px < box[2] and box[1] < py < box[3]:
                            currently_pointing_at = i
                            break
                    
                    if currently_pointing_at != hovered_option:
                        hovered_option = currently_pointing_at
                        hover_start_time = time.time() if currently_pointing_at != -1 else None
                    
                    if hover_start_time and hovered_option != -1:
                        elapsed_time = time.time() - hover_start_time
                        progress = elapsed_time / SELECTION_LOCK_DURATION
                        
                        cv2.ellipse(frame, (px, py), (20, 20), 270, 0, progress * 360, (0, 255, 255), 3)

                        if elapsed_time > SELECTION_LOCK_DURATION:
                            if hovered_option == current_question['answer']:
                                user_is_winner = True
                                play_audio_by_name("correct_answer")
                            else:
                                play_audio_by_name("wrong_answer")
                            
                            answered_questions.add(current_question['id'])
                            STATE = "AWAITING_FEEDBACK_END" # New state to wait for audio
                            hover_start_time = None
                            hovered_option = -1
        
        # Draw subtitles if audio is playing
        if pygame.mixer.get_busy() and current_subtitle:
            draw_subtitles(frame, current_subtitle)
        else:
            current_subtitle = ""

        if latest_gesture_result:
            for hand_landmarks in latest_gesture_result.hand_landmarks:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in hand_landmarks])
                mp_drawing.draw_landmarks(frame, hand_landmarks_proto, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Robot Interaction View', frame)
        cv2.imshow(VOICE_WINDOW_NAME, visualizer_frame)
        if cv2.waitKey(5) & 0xFF == 27: break

# --- CLEANUP ---
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
