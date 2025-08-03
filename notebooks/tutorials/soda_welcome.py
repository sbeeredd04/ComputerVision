import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import qrcode
import time
import threading
import random

# ---------- Configuration ----------
CLUB_NAME = "SoDA"
SIGNUP_URL = "https://your-club-signup.example.com"  # Replace with actual link
CONFETTI_COUNT = 50

# ---------- Helpers ----------
def generate_qr_image(data, size=60):
    """Generate a QR code image with specified size"""
    qr = qrcode.QRCode(box_size=2, border=1)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img

def draw_confetti(frame, confetti_particles):
    """Draw confetti particles on the frame"""
    h, w = frame.shape[:2]
    new_particles = []
    for (x, y, vy, color, life) in confetti_particles:
        if 0 <= int(x) < w and 0 <= int(y) < h:
            cv2.circle(frame, (int(x), int(y)), 2, color, -1)
        y += vy
        life -= 1
        if y < h and life > 0:
            new_particles.append((x, y, vy + 0.1, color, life))
    return new_particles

def classify_gesture(hand_landmarks, h, w):
    """Classify hand gesture based on MediaPipe landmarks"""
    # Landmarks index as per MediaPipe:
    # 4: thumb_tip, 8: index_tip, 12: middle_tip, 16: ring_tip, 20: pinky_tip
    coords = lambda i: np.array([hand_landmarks.landmark[i].x * w, hand_landmarks.landmark[i].y * h])
    
    thumb_tip = coords(4)
    thumb_ip = coords(3)
    index_tip = coords(8)
    middle_tip = coords(12)
    ring_tip = coords(16)
    pinky_tip = coords(20)

    def is_extended(tip, pip):
        return tip[1] < pip[1]  # y decreases upward

    # Thumbs up: thumb extended, other fingers folded
    thumb_up = thumb_tip[1] < thumb_ip[1]
    other_fingers_folded = (not is_extended(index_tip, coords(6)) and 
                           not is_extended(middle_tip, coords(10)) and 
                           not is_extended(ring_tip, coords(14)) and 
                           not is_extended(pinky_tip, coords(18)))
    
    if thumb_up and other_fingers_folded:
        return "thumbs_up"

    # Peace sign: index & middle extended, ring and pinky folded
    if (is_extended(index_tip, coords(6)) and 
        is_extended(middle_tip, coords(10)) and 
        not is_extended(ring_tip, coords(14)) and 
        not is_extended(pinky_tip, coords(18))):
        return "peace"

    return None

def draw_minimal_hand_landmarks(frame, hand_landmarks, color=(255, 255, 255)):
    """Draw minimal hand landmarks with thin lines"""
    h, w = frame.shape[:2]
    
    # Draw only key points with small circles
    key_points = [4, 8, 12, 16, 20]  # fingertips
    for point_id in key_points:
        landmark = hand_landmarks.landmark[point_id]
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 3, color, -1)
    
    # Draw minimal connections
    connections = [(4, 3), (8, 6), (12, 10), (16, 14), (20, 18)]  # finger tips to PIP
    for connection in connections:
        start_point = hand_landmarks.landmark[connection[0]]
        end_point = hand_landmarks.landmark[connection[1]]
        start_x, start_y = int(start_point.x * w), int(start_point.y * h)
        end_x, end_y = int(end_point.x * w), int(end_point.y * h)
        cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 1)

def draw_minimal_bounding_box(frame, x1, y1, x2, y2, label, conf, color=(255, 255, 255)):
    """Draw minimal bounding box with thin lines"""
    # Draw thin rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    
    # Draw minimal label
    label_text = f"{label}"
    font_scale = 0.4
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
    
    # Draw background rectangle for text
    cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, label_text, (x1, y1 - 5), font, font_scale, color, thickness)

# ---------- Main Loop ----------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    model = YOLO("yolov8n.pt")  # lightweight pretrained model

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=4,  # Support multiple hands
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    last_trigger_time = 0
    confetti_particles = []
    
    # Generate QR code once
    qr_img = generate_qr_image(SIGNUP_URL)

    print("Starting Club Fair Attraction...")
    print("Press 'ESC' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
            
        frame = cv2.flip(frame, 1)  # mirror
        h, w = frame.shape[:2]

        # YOLO detection (person or any object)
        results = model(frame, verbose=False)[0]
        
        # Draw minimal bounding boxes
        person_detected = False
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names.get(cls_id, str(cls_id))
            if conf < 0.3:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Only show person detections
            if label == "person":
                person_detected = True
                draw_minimal_bounding_box(frame, x1, y1, x2, y2, label, conf, (255, 255, 255))

        # MediaPipe hand detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_res = hands.process(rgb)
        gestures_detected = []
        
        if hand_res.multi_hand_landmarks:
            for i, handLms in enumerate(hand_res.multi_hand_landmarks):
                # Draw minimal hand landmarks
                draw_minimal_hand_landmarks(frame, handLms, (255, 255, 255))
                
                # Classify gesture
                gesture = classify_gesture(handLms, h, w)
                if gesture:
                    gestures_detected.append(gesture)
                    
                    # Draw minimal gesture label
                    label_text = f"{gesture}"
                    font_scale = 0.5
                    thickness = 1
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    
                    # Position label near the hand
                    wrist = handLms.landmark[0]
                    label_x = int(wrist.x * w) + 10
                    label_y = int(wrist.y * h) - 10
                    
                    # Ensure label is within frame bounds
                    label_x = max(10, min(label_x, w - 100))
                    label_y = max(30, min(label_y, h - 10))
                    
                    # Draw background for text
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                    cv2.rectangle(frame, (label_x - 2, label_y - text_height - 2), 
                                (label_x + text_width + 2, label_y + 2), (0, 0, 0), -1)
                    
                    # Draw text
                    cv2.putText(frame, label_text, (label_x, label_y), 
                               font, font_scale, (255, 255, 255), thickness)

        # Trigger on gesture if person is present
        now = time.time()
        if person_detected and gestures_detected and now - last_trigger_time > 2:
            last_trigger_time = now
            print(f"Gestures detected: {gestures_detected}")
            
            # Create confetti burst
            confetti_particles = []
            for _ in range(CONFETTI_COUNT):
                x = random.uniform(0, w)
                y = random.uniform(-50, 0)
                vy = random.uniform(2, 5)
                color = (random.randint(100,255), random.randint(100,255), random.randint(100,255))
                life = random.randint(20, 40)
                confetti_particles.append((x, y, vy, color, life))

        # Draw confetti if active
        if confetti_particles:
            confetti_particles = draw_confetti(frame, confetti_particles)

        # Draw minimal welcome text if person present
        if person_detected:
            welcome_text = f"Join {CLUB_NAME}!"
            font_scale = 0.6
            thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Position at bottom center
            (text_width, text_height), baseline = cv2.getTextSize(welcome_text, font, font_scale, thickness)
            text_x = (w - text_width) // 2
            text_y = h - 20
            
            # Draw background for text
            cv2.rectangle(frame, (text_x - 5, text_y - text_height - 5), 
                         (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, welcome_text, (text_x, text_y), 
                       font, font_scale, (255, 255, 255), thickness)

        # Draw QR code in top-right corner
        if qr_img is not None:
            qh, qw = qr_img.shape[:2]
            # Position in top-right corner with small margin
            qr_x = w - qw - 10
            qr_y = 10
            
            # Ensure QR code fits
            if qr_x >= 0 and qr_y >= 0 and qr_x + qw <= w and qr_y + qh <= h:
                frame[qr_y:qr_y + qh, qr_x:qr_x + qw] = qr_img
                
                # Add minimal "Scan" text
                scan_text = "Scan"
                font_scale = 0.3
                thickness = 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # Position text below QR code
                (text_width, text_height), baseline = cv2.getTextSize(scan_text, font, font_scale, thickness)
                scan_x = qr_x + (qw - text_width) // 2
                scan_y = qr_y + qh + text_height + 5
                
                if scan_y + text_height <= h:
                    cv2.putText(frame, scan_text, (scan_x, scan_y), 
                               font, font_scale, (255, 255, 255), thickness)

        cv2.imshow("Club Fair Attraction", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main()