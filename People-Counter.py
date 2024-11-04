import cv2
import numpy as np
import math
from ultralytics import YOLO
from sort import *

# Initialize YOLO model and tracker
model = YOLO("yolov8l.pt")
model.conf = 0.12      # Confidence threshold
model.iou = 0.35       # IOU threshold
model.classes = [0]    # Person class only
model.verbose = False

# Initialize tracker
tracker = Sort(max_age=25, min_hits=1, iou_threshold=0.25)

# Add STS colors
STS_GREEN = (0, 153, 76)  # BGR format
STS_BLUE = (153, 76, 0)   # BGR format
BACKGROUND_COLOR = (51, 51, 51)  # Dark gray

def list_available_cameras():
    """List all available camera indices"""
    available_cameras = []
    for i in range(10):  # Check first 10 indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def select_camera():
    """Let user select from available cameras"""
    cameras = list_available_cameras()
    if not cameras:
        print("No cameras found!")
        return None
    
    print("\nAvailable cameras:")
    for i, cam in enumerate(cameras):
        print(f"{i}: Camera {cam}")
    
    selected = 0  # Default to first camera
    print(f"\nSelected Camera {cameras[selected]}")
    return cameras[selected]

# Initialize camera
selected_camera = select_camera()
if selected_camera is None:
    exit()

cap = cv2.VideoCapture(selected_camera)
if not cap.isOpened():
    print(f"Failed to open camera {selected_camera}")
    exit()

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Initialize window
window_name = "STS Innovation Lab People Counter"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Keep track of the last frame with overlay
last_overlay = None
frame_buffer = 2
frame_counter = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break
        
    # Get frame dimensions
    h, w, _ = img.shape
    original_img = img.copy()  # Keep a clean copy
    
    frame_counter += 1
    if frame_counter % frame_buffer != 0:
        # Show the previous frame with existing overlay if available
        if last_overlay is not None:
            cv2.imshow(window_name, last_overlay)
        else:
            cv2.imshow(window_name, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # Object Detection and tracking
    results = model.track(img, classes=model.classes)
    
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.15:
                detections.append([x1, y1, x2, y2, conf])

    # Update tracker
    if detections:
        detections_np = np.array(detections)
    else:
        detections_np = np.empty((0, 5))

    tracker_outputs = tracker.update(detections_np)

    # First draw the semi-transparent top banner
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    # Draw detections
    for output in tracker_outputs:
        x1, y1, x2, y2, track_id = output
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw solid border first
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw ID with background
        text = f"ID: {int(track_id)}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Draw text background
        cv2.rectangle(img, (x1, y1 - 20), (x1 + text_size[0], y1), (0, 255, 0), -1)
        # Draw text
        cv2.putText(img, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Display count with better visibility
    count_text = f"{len(tracker_outputs)} People currently enjoying the Innovation Lab"
    text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    text_x = (w - text_size[0]) // 2
    
    # Draw text with shadow effect
    cv2.putText(img, count_text, (text_x + 2, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    cv2.putText(img, count_text, (text_x, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Store this frame as the last overlay
    last_overlay = img.copy()
    
    # Display the frame
    cv2.imshow(window_name, img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
