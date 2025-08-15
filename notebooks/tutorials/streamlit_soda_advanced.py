import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import qrcode
import time
import threading
import random
import base64
from PIL import Image
import io
import av
import streamlit_webrtc as webrtc
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os

# Import TTS functionality
try:
    from tts_utils import RobotTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    st.warning("TTS module not available. Install google-genai to enable voice features.")

# ---------- Configuration ----------
CLUB_NAME = "SoDA"
SIGNUP_URL = "https://thesoda.io/"  # Replace with actual link
CONFETTI_COUNT = 50

# ---------- Helpers ----------
def generate_qr_image(data, size=200):
    """Generate a QR code image with specified size"""
    qr = qrcode.QRCode(box_size=2, border=1)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img

# Initialize TTS system
if TTS_AVAILABLE:
    try:
        tts_system = RobotTTS()
    except Exception as e:
        st.error(f"Failed to initialize TTS: {e}")
        TTS_AVAILABLE = False
else:
    tts_system = None

# ---------- Video Processor Class ----------
class GestureProcessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=4,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize YOLO
        try:
            self.model = YOLO("yolov8n.pt")
        except Exception:
            self.model = None

        # State tracking
        self.person_detected_time = None
        self.last_gesture_time = 0
        self.gesture_cooldown = 2.0

        # Conversation state
        self.conversation_state = 'waiting'
        self.greeted = False
        self.talked_about_soda = False
        self.asked_for_heart = False
        self.after_talk_time = None

        # Gesture detection
        self.last_gesture = None
        self.gesture_confidence = 0

    def classify_gesture(self, hand_landmarks, h, w):
        coords = lambda i: np.array([hand_landmarks.landmark[i].x * w, hand_landmarks.landmark[i].y * h])

        thumb_tip = coords(4)
        thumb_ip = coords(3)
        index_tip = coords(8)
        middle_tip = coords(12)
        ring_tip = coords(16)
        pinky_tip = coords(20)

        def is_extended(tip, pip):
            return tip[1] < pip[1]

        thumb_up = thumb_tip[1] < thumb_ip[1]
        other_fingers_folded = (
            not is_extended(index_tip, coords(6)) and
            not is_extended(middle_tip, coords(10)) and
            not is_extended(ring_tip, coords(14)) and
            not is_extended(pinky_tip, coords(18))
        )
        if thumb_up and other_fingers_folded:
            return "thumbs_up", 0.9

        if (
            is_extended(index_tip, coords(6)) and
            is_extended(middle_tip, coords(10)) and
            not is_extended(ring_tip, coords(14)) and
            not is_extended(pinky_tip, coords(18))
        ):
            index_mcp = coords(5)
            middle_mcp = coords(9)
            if (index_tip[1] > index_mcp[1] and middle_tip[1] > middle_mcp[1]):
                return "heart", 0.8

        if (
            is_extended(index_tip, coords(6)) and
            is_extended(middle_tip, coords(10)) and
            not is_extended(ring_tip, coords(14)) and
            not is_extended(pinky_tip, coords(18))
        ):
            return "peace", 0.7

        return None, 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # Person detection with YOLO
        person_detected = False
        if self.model is not None:
            try:
                results = self.model(img, verbose=False)[0]
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.model.names.get(cls_id, str(cls_id))
                    if conf > 0.3 and label == "person":
                        person_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, f"Person: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        break
            except Exception:
                pass

        now = time.time()
        if person_detected:
            if self.person_detected_time is None:
                self.person_detected_time = now
            if (now - self.person_detected_time) >= 1.0 and not self.greeted:
                self.conversation_state = 'greeting'
                self.greeted = True
                # Trigger light "talk" and audio
                if TTS_AVAILABLE:
                    audio_file = tts_system.get_robot_greeting_audio()
                    st.session_state.last_audio_file = audio_file
                st.session_state.light_action = {"type": "light_action", "action": "speak", "durationMs": 5000}
        else:
            self.person_detected_time = None
            self.conversation_state = 'waiting'

        # Hand gesture detection
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                gesture, confidence = self.classify_gesture(hand_landmarks, h, w)
                if gesture and confidence > 0.7 and (now - self.last_gesture_time > self.gesture_cooldown):
                    self.last_gesture_time = now
                    self.last_gesture = gesture
                    self.gesture_confidence = confidence

                    if gesture == "thumbs_up" and self.conversation_state in ('greeting', 'waiting'):
                        self.conversation_state = 'talking_about_soda'
                        self.talked_about_soda = True
                        self.after_talk_time = now
                        if TTS_AVAILABLE:
                            audio_file = tts_system.get_soda_info_audio()
                            st.session_state.last_audio_file = audio_file
                        st.session_state.light_action = {"type": "light_action", "action": "speak", "durationMs": 6000}

                    elif gesture == "heart" and self.conversation_state == 'talking_about_soda':
                        self.conversation_state = 'asking_for_heart'
                        self.asked_for_heart = True
                        if TTS_AVAILABLE:
                            audio_file = tts_system.get_heart_request_audio()
                            st.session_state.last_audio_file = audio_file
                        st.session_state.light_action = {"type": "light_action", "action": "speak", "durationMs": 4000}

                if gesture:
                    wrist = hand_landmarks.landmark[0]
                    label_x = int(wrist.x * w) + 10
                    label_y = int(wrist.y * h) - 10
                    cv2.putText(img, f"{gesture} ({confidence:.2f})", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw conversation state
        cv2.putText(img, f"State: {self.conversation_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if self.last_gesture:
            cv2.putText(img, f"Last Gesture: {self.last_gesture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------- Talking Light HTML ----------
def get_light_html():
    return """
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Talking Light</title>
    <style>
    body {
        margin: 0;
        padding: 0;
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 60vh;
        font-family: Arial, sans-serif;
        overflow: hidden;
    }
    .light-container {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .light {
        width: 220px;
        height: 220px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,0.95) 0%, rgba(0, 255, 255, 0.45) 60%, rgba(0, 180, 255, 0.2) 100%);
        box-shadow: 0 0 50px rgba(0,255,255,0.85), 0 0 140px rgba(0,255,255,0.35);
        animation: breathe 3s ease-in-out infinite;
        transition: filter 0.3s ease;
    }
    @keyframes breathe {
        0%, 100% { transform: scale(1); filter: brightness(1); }
        50% { transform: scale(1.05); filter: brightness(1.15); }
    }
    .light.talking {
        animation: talk 0.6s ease-in-out infinite;
        filter: brightness(1.35);
        box-shadow: 0 0 70px rgba(0,255,255,0.95), 0 0 180px rgba(0,255,255,0.45);
    }
    @keyframes talk {
        0%, 100% { transform: scale(1.00); }
        50% { transform: scale(1.12); }
    }
    </style>
    </head>
    <body>
        <div class=\"light-container\">
            <div class=\"light\" id=\"sodaLight\"></div>
        </div>
        <script>
        (function(){
            const light = document.getElementById('sodaLight');
            let talkTimer = null;
            function startTalking(durationMs){
                if (!light) return;
                light.classList.add('talking');
                if (talkTimer) clearTimeout(talkTimer);
                talkTimer = setTimeout(() => {
                    light.classList.remove('talking');
                }, durationMs || 3000);
            }
            window.addEventListener('message', (event) => {
                try {
                    const data = event.data || {};
                    if (data.type === 'light_action' && data.action === 'speak') {
                        startTalking(data.durationMs || 3000);
                    }
                } catch (e) {}
            });
        })();
        </script>
    </body>
    </html>
    """

# ---------- Main Streamlit App ----------
def main():
    st.set_page_config(
        page_title="SoDA Interactive",
        page_icon=None,
        layout="wide"
    )

    st.title("SoDA Interactive")
    st.markdown("---")

    # Initialize session state
    if 'light_action' not in st.session_state:
        st.session_state.light_action = None
    if 'last_audio_file' not in st.session_state:
        st.session_state.last_audio_file = None

    # Create two columns for split screen
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Camera Feed")
        st.caption("Real-time person and gesture detection")

        webrtc_ctx = webrtc_streamer(
            key="gesture-processor",
            video_processor_factory=GestureProcessor,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if webrtc_ctx.state.playing:
            if hasattr(webrtc_ctx, 'video_processor') and webrtc_ctx.video_processor is not None:
                proc = webrtc_ctx.video_processor
                st.info(f"Current State: {proc.conversation_state}")
                if proc.last_gesture:
                    st.success(f"Last Gesture: {proc.last_gesture} (confidence: {proc.gesture_confidence:.2f})")
        else:
            st.warning("Camera not active. Click Start to begin.")

    with col2:
        st.subheader("Talking Light")
        st.caption("Breathing light that emphasizes when speaking")

        light_html = get_light_html()
        st.components.v1.html(light_html, height=400)

        # Voice selection only
        if TTS_AVAILABLE:
            st.markdown("---")
            st.subheader("Voice Selection")
            voices = tts_system.list_available_voices()
            selected_voice = st.selectbox(
                "Choose Robot Voice:",
                voices,
                index=voices.index("Puck -- Upbeat") if "Puck -- Upbeat" in voices else 0
            )
            if st.button("Apply Voice"):
                tts_system.change_voice(selected_voice)
                st.success(f"Voice changed to: {selected_voice}")
        else:
            st.warning("Text-to-speech is not available.")

        # Show the three conversation texts
        st.markdown("---")
        st.subheader("Conversation Texts")
        st.markdown("Greeting:")
        st.code("Hey there! Nice to meet you! Give me a thumbs up if you'd like to know more about SoDA!", language="text")
        st.markdown("About SoDA:")
        st.code("SoDA is the Software Development Association! We build amazing projects, learn new technologies, and have fun together. Want to join us?", language="text")
        st.markdown("Ask for Heart:")
        st.code("If you like what you see, give me a heart gesture!", language="text")

        # Play latest generated audio if available
        if st.session_state.last_audio_file:
            st.markdown("---")
            st.subheader("Latest Speech")
            st.audio(st.session_state.last_audio_file)

    # Send light action message to the embedded component
    if st.session_state.light_action:
        st.markdown(
            """
            <script>
            const frames = document.querySelectorAll('iframe');
            if (frames && frames.length > 0) {
                const f = frames[frames.length - 1];
                try {
                    f.contentWindow.postMessage({
                        type: 'light_action',
                        action: 'speak',
                        durationMs: 4000
                    }, '*');
                } catch (e) {}
            }
            </script>
            """,
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown("---")
    st.caption("SoDA Interactive - Built with Streamlit, OpenCV, MediaPipe, and YOLO")

if __name__ == "__main__":
    main()
