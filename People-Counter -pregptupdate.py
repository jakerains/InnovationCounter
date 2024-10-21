import numpy as np
from ultralytics import YOLO
import cv2
import pyresearch
import math
from sort import *

def list_available_cameras():
    index = 0
    cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            cameras.append(index)
        cap.release()
        index += 1
    return cameras

def select_camera():
    cameras = list_available_cameras()
    if not cameras:
        print("No cameras found.")
        return None
    
    print("Available cameras:")
    for i, camera in enumerate(cameras):
        print(f"{i}: Camera {camera}")
    
    while True:
        try:
            selection = int(input("Select a camera by number: "))
            if 0 <= selection < len(cameras):
                return cameras[selection]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Replace the webcam initialization code
selected_camera = select_camera()
if selected_camera is None:
    print("No camera selected. Exiting.")
    exit()

cap = cv2.VideoCapture(selected_camera)  # Use 0 for default webcam, or try other indices if you have multiple cameras

# Get webcam resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Adjust counting lines for webcam resolution
limitsUp = [0, height // 2 - 30, width, height // 2 - 30]
limitsDown = [0, height // 2 + 30, width, height // 2 + 30]

model = YOLO("yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

totalCountUp = []
totalCountDown = []

def draw_line(event, x, y, flags, param):
    global line1_start, line1_end, line2_start, line2_end, current_line
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_line == "line1_start":
            line1_start = (x, y)
            current_line = "line1_end"
        elif current_line == "line1_end":
            line1_end = (x, y)
            current_line = "line2_start"
        elif current_line == "line2_start":
            line2_start = (x, y)
            current_line = "line2_end"
        elif current_line == "line2_end":
            line2_end = (x, y)
            current_line = None

line1_start = None
line1_end = None
line2_start = None
line2_end = None
current_line = "line1_start"

cv2.namedWindow("Webcam People Counter")
cv2.setMouseCallback("Webcam People Counter", draw_line)

print("Click to set the start point of the first counting line")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    if current_line:
        if current_line == "line1_start":
            cv2.putText(img, "Click to set start point of first line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif current_line == "line1_end":
            cv2.putText(img, "Click to set end point of first line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif current_line == "line2_start":
            cv2.putText(img, "Click to set start point of second line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif current_line == "line2_end":
            cv2.putText(img, "Click to set end point of second line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if line1_start and line1_end:
        cv2.line(img, line1_start, line1_end, (0, 0, 255), 2)
    if line2_start and line2_end:
        cv2.line(img, line2_start, line2_end, (255, 0, 0), 2)

    results = model(img, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        pyresearch.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        pyresearch.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                               scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if line1_start and line1_end and line2_start and line2_end:
            if intersect((cx, cy), (cx, cy-1), line1_start, line1_end):
                if id not in totalCountUp:
                    totalCountUp.append(id)
                    cv2.line(img, line1_start, line1_end, (0, 255, 0), 5)
            elif intersect((cx, cy), (cx, cy-1), line2_start, line2_end):
                if id not in totalCountDown:
                    totalCountDown.append(id)
                    cv2.line(img, line2_start, line2_end, (0, 255, 0), 5)

    cv2.putText(img, f"Up: {len(totalCountUp)}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.putText(img, f"Down: {len(totalCountDown)}", (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    cv2.imshow("Webcam People Counter", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        line1_start = line1_end = line2_start = line2_end = None
        current_line = "line1_start"
        totalCountUp.clear()
        totalCountDown.clear()
        print("Lines reset. Click to set the start point of the first counting line")

cap.release()
cv2.destroyAllWindows()

def intersect(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return False
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    if ua < 0 or ua > 1:
        return False
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    if ub < 0 or ub > 1:
        return False
    return True
