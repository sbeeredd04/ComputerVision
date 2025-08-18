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
import torch

# --- Dependency for SVG rendering ---
try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    SVG_SUPPORT = True
except ImportError:
    SVG_SUPPORT = False
    print("WARNING: svglib or reportlab is not installed. The SVG logo will not be displayed. Please run 'pip install svglib reportlab Pillow'")


# --- TTS Class for On-the-Fly Audio Generation ---
# NOTE: You must set the 'GEMINI_API_KEY' environment variable for this to work.
try:
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
        
        self.client = genai.Client(api_key=api_key)
        self.audio_dir = "robot_audio"
        os.makedirs(self.audio_dir, exist_ok=True)
        self.voice_name = 'Puck'

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
STATE = "SCREENSAVER"
PERSON_CONFIDENCE_THRESHOLD = 0.6
PERSON_PRESENCE_TIME_THRESHOLD = 3.0  # Configurable timer for person detection
YOUR_CLUB_WEBSITE_URL = "https://www.yourclubwebsite.com"
latest_gesture_result = None
MODEL_PATH = "gesture_recognizer.task"
last_person_seen_time = 0
PERSON_RESET_TIMEOUT = 5.0
screen_saver_message = "Step Up to Play!"
LOGO_SVG_PATH = "soda.svg"
logo_img = None
person_detected_time = None # Initialize here
quiz_questions = []
current_question = None
answered_questions = set()
skip_available = True
hovered_option = -1
hover_start_time = None
SELECTION_LOCK_DURATION = 3.0
user_is_winner = False
SUBTITLES = {}
current_subtitle = ""

# --- UI CUSTOMIZATION VARIABLES ---
OPTION_BG_COLOR_NORMAL = (50, 50, 70)  # Normal option background color (BGR)
OPTION_BG_COLOR_HOVER = (70, 130, 180)  # Hovered option background color (BGR)
OPTION_BORDER_COLOR_NORMAL = (120, 120, 140)  # Normal option border color (BGR)
OPTION_BORDER_COLOR_HOVER = (100, 200, 255)  # Hovered option border color (BGR)

# --- CUDA DETECTION ---
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    DEVICE = 'cuda'
else:
    print("CUDA not available. Using CPU.")
    DEVICE = 'cpu'

try:
    with open('questions.json', 'r') as f:
        quiz_questions = json.load(f)['questions']
    print(f"Loaded {len(quiz_questions)} quiz questions.")
except Exception as e:
    print(f"Error loading questions.json: {e}"); exit()

try:
    tts = RobotTTS()
except Exception as e:
    print(f"Failed to initialize TTS: {e}"); exit()

# --- HELPER & GAME FUNCTIONS ---

def load_logo(target_width):
    """Loads SVG, renders it to a PIL Image, and converts to an OpenCV-compatible format."""
    if not SVG_SUPPORT or not os.path.exists(LOGO_SVG_PATH):
        print(f"DEBUG: Logo not loaded. SVG support: {SVG_SUPPORT}, Path exists: {os.path.exists(LOGO_SVG_PATH)}")
        return None
    try:
        print("DEBUG: Attempting to load and render SVG...")
        drawing = svg2rlg(LOGO_SVG_PATH)
        if drawing.width == 0: return None # Handle empty SVG
        
        scale_factor = target_width / drawing.width
        drawing.width *= scale_factor
        drawing.height *= scale_factor
        drawing.scale(scale_factor, scale_factor)
        
        # This is the most compatible way to render, using the Pillow backend
        pil_image = renderPM.drawToPIL(drawing, bg=(255, 255, 255, 0)) # Transparent BG
        
        print("DEBUG: SVG successfully rendered to PIL Image.")
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)
    except Exception as e:
        print(f"CRITICAL ERROR loading or converting SVG logo: {e}")
        return None

def overlay_transparent_image(background, overlay, x, y):
    """Overlays a BGRA image with transparency onto a BGR background."""
    h, w, _ = overlay.shape
    alpha = overlay[:, :, 3] / 255.0
    overlay_rgb = overlay[:, :, :3]
    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = (alpha * overlay_rgb[:, :, c] +
                                       (1 - alpha) * background[y:y+h, x:x+w, c])

def create_dynamic_gradient_background(frame_height, frame_width):
    """Creates a dynamic gradient background with white, black, blue, and red that changes constantly."""
    gradient = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    # Time-based animation for living gradient effect
    t = time.time()
    
    # Multiple wave patterns for complex color mixing
    wave1 = math.sin(t * 0.5) * 0.5 + 0.5  # Slow wave
    wave2 = math.sin(t * 1.2) * 0.5 + 0.5  # Medium wave
    wave3 = math.sin(t * 2.0) * 0.5 + 0.5  # Fast wave
    wave4 = math.cos(t * 0.8) * 0.5 + 0.5  # Cosine wave for variation
    
    for i in range(frame_height):
        for j in range(frame_width):
            # Normalize coordinates
            y_norm = i / frame_height
            x_norm = j / frame_width
            
            # Create multiple gradient factors with time animation
            diagonal_factor = (y_norm + x_norm) / 2
            radial_factor = math.sqrt((x_norm - 0.5)**2 + (y_norm - 0.5)**2)
            
            # Animated color mixing with white, black, blue, and red
            # Blue component (animated)
            blue_base = int(30 + diagonal_factor * 100 * wave1 + radial_factor * 80 * wave2)
            blue = max(0, min(255, blue_base))
            
            # Red component (animated)
            red_base = int(20 + (1 - diagonal_factor) * 120 * wave3 + y_norm * 60 * wave4)
            red = max(0, min(255, red_base))
            
            # Green component (creates white when combined, animated)
            green_base = int(15 + x_norm * 80 * wave2 + (1 - radial_factor) * 100 * wave1)
            green = max(0, min(255, green_base))
            
            # Add some white highlights that move around
            white_factor = math.sin(t + x_norm * 10) * math.cos(t * 1.5 + y_norm * 8)
            if white_factor > 0.7:
                white_intensity = int((white_factor - 0.7) * 200)
                blue = min(255, blue + white_intensity)
                green = min(255, green + white_intensity)
                red = min(255, red + white_intensity)
            
            # Add some black areas that shift
            black_factor = math.sin(t * 0.3 + x_norm * 5) * math.cos(t * 0.7 + y_norm * 6)
            if black_factor < -0.6:
                black_intensity = abs(black_factor + 0.6) * 0.8
                blue = int(blue * (1 - black_intensity))
                green = int(green * (1 - black_intensity))
                red = int(red * (1 - black_intensity))
            
            gradient[i, j] = [blue, green, red]
    
    return gradient

def draw_modern_text(frame, text, position, font_scale=1.0, thickness=2, color=(255, 255, 255)):
    """Draws text with modern styling including shadow effect."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Draw shadow first (offset by 2 pixels)
    shadow_pos = (position[0] + 2, position[1] + 2)
    cv2.putText(frame, text, shadow_pos, font, font_scale, (0, 0, 0), thickness + 1)
    
    # Draw main text
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

def pregenerate_static_audio():
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
        SUBTITLES[name] = text
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
    print("DEBUG: Resetting game state to screensaver...")
    STATE = "SCREENSAVER"
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
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = (frame.shape[1] - text_width) // 2
    y = frame.shape[0] - 30
    
    # Draw solid black background for better readability
    padding = 15
    cv2.rectangle(frame, 
                 (x - padding, y - text_height - padding), 
                 (x + text_width + padding, y + baseline + padding), 
                 (0, 0, 0), -1)  # Solid black background
    
    # Add a subtle border
    cv2.rectangle(frame, 
                 (x - padding, y - text_height - padding), 
                 (x + text_width + padding, y + baseline + padding), 
                 (50, 50, 50), 2)  # Dark gray border
    
    # Draw the text
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)

def draw_speaking_orb(frame):
    h, w, _ = frame.shape
    center = (int(w * 0.1), int(h * 0.9))
    t = time.time()
    base_pulse = (math.sin(t * 5) + 1) / 2 * 0.5 + 0.5
    max_radius = int(min(h, w) * 0.08)
    orb_overlay = frame.copy()
    radius1 = int(max_radius * base_pulse)
    cv2.circle(orb_overlay, center, radius1, (255, 100, 100), -1)
    radius2 = int(radius1 * (0.7 + (math.sin(t * 7) + 1) / 2 * 0.2))
    cv2.circle(orb_overlay, center, radius2, (255, 180, 180), -1)
    radius3 = int(radius2 * 0.5)
    cv2.circle(orb_overlay, center, radius3, (255, 255, 255), -1)
    cv2.addWeighted(orb_overlay, 0.7, frame, 0.3, 0, frame)

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
# Initialize YOLO model with CUDA support if available
yolo_model = YOLO("yolov8n.pt")
if CUDA_AVAILABLE:
    yolo_model.to(DEVICE)
    print(f"YOLO model loaded on {DEVICE}")
else:
    print(f"YOLO model loaded on {DEVICE}")
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
success, temp_frame = cap.read()
if not success:
    print("Could not read from camera. Exiting.")
    exit()
frame_height, frame_width, _ = temp_frame.shape
logo_img = load_logo(target_width=int(frame_width * 0.4))

with vision.GestureRecognizer.create_from_options(options) as recognizer:
    while cap.isOpened():
        # --- SCREENSAVER LOGIC ---
        if STATE == "SCREENSAVER" or STATE == "PERSON_DETECTED":
            # Create dynamic gradient background
            frame = create_dynamic_gradient_background(frame_height, frame_width)
            
            # Add subtle animated pattern
            t = time.time()
            pattern_alpha = int((math.sin(t * 0.5) + 1) * 30 + 20)  # Oscillates between 20-80
            
            # Create subtle geometric pattern
            pattern_overlay = frame.copy()
            for i in range(0, frame_width, 120):
                for j in range(0, frame_height, 120):
                    offset_x = int(math.sin(t * 0.3 + i * 0.01) * 10)
                    offset_y = int(math.cos(t * 0.3 + j * 0.01) * 10)
                    cv2.circle(pattern_overlay, (i + offset_x, j + offset_y), 3, (100, 150, 200), 1)
            
            cv2.addWeighted(pattern_overlay, 0.3, frame, 0.7, 0, frame)
            
            # Display logo with modern positioning
            if logo_img is not None:
                logo_h, logo_w, _ = logo_img.shape
                x_pos = (frame_width - logo_w) // 2
                y_pos = int(frame_height * 0.25)  # Higher up for better composition
                overlay_transparent_image(frame, logo_img, x_pos, y_pos)
            
            # Modern main text styling
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.2
            thickness = 3
            (text_w, text_h), _ = cv2.getTextSize(screen_saver_message, font, font_scale, thickness)
            text_x = (frame_width - text_w) // 2
            text_y = int(frame_height * 0.7)
            
            # Draw main text with modern styling
            draw_modern_text(frame, screen_saver_message, (text_x, text_y), font_scale, thickness, (255, 255, 255))
            
            # Add subtle instruction text
            instruction_text = "Stand in front to begin"
            instruction_font_scale = 0.8
            instruction_thickness = 2
            (inst_w, inst_h), _ = cv2.getTextSize(instruction_text, font, instruction_font_scale, instruction_thickness)
            inst_x = (frame_width - inst_w) // 2
            inst_y = text_y + 60
            
            draw_modern_text(frame, instruction_text, (inst_x, inst_y), instruction_font_scale, instruction_thickness, (180, 180, 180))

            success, real_frame = cap.read()
            if not success: continue
            
            person_found = False
            yolo_results = yolo_model(real_frame, classes=[0], verbose=False, max_det=1)
            if any(d.conf.item() > PERSON_CONFIDENCE_THRESHOLD for d in yolo_results[0].boxes):
                person_found = True
                if STATE == "SCREENSAVER":
                    STATE = "PERSON_DETECTED"
                    person_detected_time = time.time()
                    print(f"DEBUG: Person detected! Starting {PERSON_PRESENCE_TIME_THRESHOLD}s timer.")
                
                elapsed_time = time.time() - (person_detected_time or 0)
                # Modern timer display
                timer_text = f"Starting in {PERSON_PRESENCE_TIME_THRESHOLD - elapsed_time:.1f}s"
                timer_bg = frame.copy()
                cv2.rectangle(timer_bg, (40, 50), (350, 100), (50, 50, 50), -1)
                cv2.addWeighted(timer_bg, 0.8, frame, 0.2, 0, frame)
                draw_modern_text(frame, timer_text, (50, 85), 0.9, 2, (0, 255, 150))

                if elapsed_time > PERSON_PRESENCE_TIME_THRESHOLD:
                    print("DEBUG: Timer finished. Switching to GREETING state.")
                    STATE = "GREETING"
                    # --- BUG FIX: Initialize last_person_seen_time here! ---
                    # This prevents the immediate timeout in the interactive loop.
                    last_person_seen_time = time.time()
                    play_audio_by_name("greeting")
            
            elif STATE == "PERSON_DETECTED":
                 print("DEBUG: Person lost. Returning to screensaver.")
                 STATE = "SCREENSAVER"
                 person_detected_time = None

            cv2.imshow('Robot Interaction View', frame)
            if cv2.waitKey(5) & 0xFF == 27: break
            continue

        # --- INTERACTIVE MODE LOGIC ---
        if time.time() - last_person_seen_time > PERSON_RESET_TIMEOUT:
            reset_game_state()
            continue

        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        recognizer.recognize_async(mp_image, int(time.time() * 1000))
        frame.flags.writeable = True

        person_found_interactive = False
        yolo_results = yolo_model(rgb_frame, classes=[0], verbose=False, max_det=1)
        if any(d.conf.item() > PERSON_CONFIDENCE_THRESHOLD for d in yolo_results[0].boxes):
            last_person_seen_time = time.time()
            person_found_interactive = True
        
        # --- State Machine for Interaction ---
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
                time.sleep(2) 
                reset_game_state()
                del qr_start_time
        elif STATE == "QUIZ_MODE":
            if current_question and not pygame.mixer.get_busy():
                # CENTERED QUIZ UI - Modern and clean layout
                
                # Question text - centered at top
                question_text = current_question['question']
                font = cv2.FONT_HERSHEY_SIMPLEX
                question_font_scale = 1.0
                question_thickness = 2
                
                # Split question into multiple lines if too long
                max_chars_per_line = 60
                question_lines = []
                words = question_text.split(' ')
                current_line = ""
                
                for word in words:
                    if len(current_line + word) < max_chars_per_line:
                        current_line += word + " "
                    else:
                        if current_line:
                            question_lines.append(current_line.strip())
                        current_line = word + " "
                
                if current_line:
                    question_lines.append(current_line.strip())
                
                # Draw question lines centered
                start_y = 80
                line_height = 40
                for i, line in enumerate(question_lines):
                    (text_w, text_h), _ = cv2.getTextSize(line, font, question_font_scale, question_thickness)
                    text_x = (frame_width - text_w) // 2
                    text_y = start_y + (i * line_height)
                    draw_modern_text(frame, line, (text_x, text_y), question_font_scale, question_thickness, (255, 255, 255))
                
                # Options - centered vertically and horizontally
                num_options = len(current_question['options'])
                option_height = 70
                option_width = 600
                total_options_height = num_options * option_height
                
                # Center the options block vertically
                options_start_y = (frame_height - total_options_height) // 2 + 50
                options_start_x = (frame_width - option_width) // 2
                
                option_boxes = []
                for i, option in enumerate(current_question['options']):
                    y_pos = options_start_y + i * option_height
                    box = (options_start_x, y_pos, options_start_x + option_width, y_pos + option_height - 10)
                    option_boxes.append(box)
                    
                    # Modern option styling
                    is_hovered = (hovered_option == i)
                    
                    # Background rectangle with configurable colors
                    bg_color = OPTION_BG_COLOR_HOVER if is_hovered else OPTION_BG_COLOR_NORMAL
                    border_color = OPTION_BORDER_COLOR_HOVER if is_hovered else OPTION_BORDER_COLOR_NORMAL
                    
                    # Draw rounded rectangle effect
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), bg_color, -1)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), border_color, 3)
                    
                    # Option text - centered in each box
                    option_text = f"{i+1}. {option}"
                    option_font_scale = 0.8
                    option_thickness = 2
                    
                    (opt_text_w, opt_text_h), _ = cv2.getTextSize(option_text, font, option_font_scale, option_thickness)
                    text_x = box[0] + (option_width - opt_text_w) // 2
                    text_y = box[1] + (option_height + opt_text_h) // 2
                    
                    text_color = (255, 255, 255) if is_hovered else (220, 220, 220)
                    draw_modern_text(frame, option_text, (text_x, text_y), option_font_scale, option_thickness, text_color)
                
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
                        
                        # Modern selection indicator
                        cv2.ellipse(frame, (px, py), (25, 25), 270, 0, progress * 360, (0, 255, 255), 4)
                        cv2.circle(frame, (px, py), 8, (255, 255, 255), -1)

                        if elapsed_time > SELECTION_LOCK_DURATION:
                            if hovered_option == current_question['answer']:
                                user_is_winner = True
                                play_audio_by_name("correct_answer")
                            else:
                                play_audio_by_name("wrong_answer")
                            
                            answered_questions.add(current_question['id'])
                            STATE = "AWAITING_FEEDBACK_END"
                            hover_start_time = None
                            hovered_option = -1
        
        # --- Drawing Overlays ---
        if pygame.mixer.get_busy():
            draw_speaking_orb(frame)
        
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
        if cv2.waitKey(5) & 0xFF == 27: break

# --- CLEANUP ---
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()