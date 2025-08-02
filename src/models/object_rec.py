import cv2
import numpy as np
from ultralytics import YOLO 
import mediapipe as mp

model = YOLO('yolov8s.pt')

# using webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
        
    # detecting objects
    results = model(frame)[0]  # get first result
    
    # drawing bounding boxes
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        # Convert to integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        class_name = results.names[int(class_id)]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_name} {score:.2f}', (x1, y1), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    cv2.imshow('Object Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()