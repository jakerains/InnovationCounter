import cv2
import numpy as np
from ultralytics import YOLO
import math
from sort import Sort  # Ensure you have SORT implemented or installed
import warnings
from collections import defaultdict

# Suppress specific FutureWarnings from PyTorch related to `torch.load`
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Download and use YOLOv8-Medium instead of Large
model = YOLO("yolov8m.pt")  # This will download automatically if not present
model.conf = 0.12
model.iou = 0.35
model.classes = [0]
model.verbose = False

# Define class names (only 'person' is needed, but including all for potential future use)
classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Optimize SORT tracker for faster movement
tracker = Sort(max_age=25,       # Increased to maintain tracks longer
              min_hits=1,        # Reduced to pick up tracks faster
              iou_threshold=0.25) # Lower IOU threshold for faster movement

# Initialize window name and create window
window_name = "Webcam People Counter"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Add fullscreen state variable and get screen resolution
is_fullscreen = False
screen = cv2.getWindowImageRect(window_name)
screen_width = 1920  # Default full HD width
screen_height = 1080  # Default full HD height

# Function to maintain aspect ratio
def resize_with_aspect_ratio(image, target_width, target_height):
    h, w = image.shape[:2]
    aspect = w / h
    
    # Calculate new dimensions maintaining aspect ratio
    if target_width / target_height > aspect:
        new_width = int(target_height * aspect)
        new_height = target_height
    else:
        new_width = target_width
        new_height = int(target_width / aspect)
    
    return cv2.resize(image, (new_width, new_height))

# Keep track of the last frame with overlay
last_overlay = None
frame_buffer = 2
frame_counter = 0

# Function to list available cameras
def list_available_cameras(max_cameras=10):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

# Function to select a camera
def select_camera():
    cameras = list_available_cameras()
    if not cameras:
        print("No cameras found.")
        return None

    print("Available cameras:")
    for idx, camera in enumerate(cameras):
        print(f"{idx}: Camera {camera}")

    # For simplicity, select the first available camera
    selected_camera = cameras[0]
    print(f"Selected Camera {selected_camera}")
    return selected_camera

# Select camera
selected_camera = select_camera()
if selected_camera is None:
    exit()  # Exit if no camera is found

cap = cv2.VideoCapture(selected_camera)
if not cap.isOpened():
    print(f"Failed to open camera {selected_camera}")
    exit()

# Get the default camera resolution first
default_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
default_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Optimize camera settings for maximum speed
cap.set(cv2.CAP_PROP_FPS, 60)               # Try to get 60 FPS
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)         # Minimize buffer delay
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # Use MJPG codec

# Processing resolution (smaller for speed)
process_width = 640
process_height = 480

cv2.resizeWindow(window_name, default_width, default_height)

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
    
    # Before displaying the frame
    if is_fullscreen:
        # Resize image maintaining aspect ratio
        display_img = resize_with_aspect_ratio(img, screen_width, screen_height)
        
        # Create black background
        black_bg = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        
        # Calculate position to center the image
        y_offset = (screen_height - display_img.shape[0]) // 2
        x_offset = (screen_width - display_img.shape[1]) // 2
        
        # Place the resized image on the black background
        black_bg[y_offset:y_offset + display_img.shape[0], 
                x_offset:x_offset + display_img.shape[1]] = display_img
        
        # Display the properly formatted fullscreen image
        cv2.imshow(window_name, black_bg)
    else:
        # Normal windowed mode display
        cv2.imshow(window_name, img)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):  # Toggle fullscreen with 'f' key
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, default_width, default_height)

cap.release()
cv2.destroyAllWindows()
