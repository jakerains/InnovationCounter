import numpy as np
from ultralytics import YOLO
import cv2
import pyresearch
import math
from sort import *

# Function to check if two lines intersect
def intersect(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return False  # Lines are parallel and do not intersect
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    if ua < 0 or ua > 1:
        return False  # Intersection point is not within the segment of line 1
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    if ub < 0 or ub > 1:
        return False  # Intersection point is not within the segment of line 2
    return True  # Lines intersect

# List all available cameras by attempting to open them
# and return a list of their indices
def list_available_cameras():
    available_cameras = []
    for i in range(10):  # Check first 10 indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

# Allows user to select a camera from the available cameras
# and returns the selected camera index
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

# Camera selection and initialization
selected_camera = select_camera()
if selected_camera is None:
    print("No camera selected. Exiting.")
    exit()

cap = cv2.VideoCapture(selected_camera)

if not cap.isOpened():
    print(f"Error: Could not open camera {selected_camera}.")
    exit()

# Get webcam resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera initialized with resolution: {width}x{height}")

# Load the YOLOv8 model for detecting people
model = YOLO("yolov8l.pt")

# Define class names to be detected
classNames = ["person"]

# Initialize the SORT tracker with specific parameters to tune performance
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Lists to store counts of people entering/exiting through two separate entrances
entrance1_count = []
entrance2_count = []
people_in_lab = 0

# Function to handle mouse click events for drawing counting lines
def draw_line(event, x, y, flags, param):
    global line1_start, line1_end, line2_start, line2_end, current_line
    if event == cv2.EVENT_LBUTTONDOWN:
        # Set the start and end points of lines based on current state
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

# Initialize line variables and set the first line to be drawn
line1_start = None
line1_end = None
line2_start = None
line2_end = None
current_line = "line1_start"

# Create a window to display the webcam feed and set mouse callback for drawing lines
cv2.namedWindow("Webcam People Counter")
# Function to toggle between windowed and fullscreen modes
def toggle_fullscreen():
    global fullscreen
    if fullscreen:
        cv2.setWindowProperty("Webcam People Counter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        fullscreen = False
    else:
        cv2.setWindowProperty("Webcam People Counter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        fullscreen = True

fullscreen = False
cv2.setMouseCallback("Webcam People Counter", draw_line)

print("Click to set the start point of the first counting line")

while True:
    try:
        success, img = cap.read()
        if not success or img is None:
            print("Failed to grab frame")
            continue

        # Display instructions for setting lines
        if current_line:
            if current_line == "line1_start":
                cv2.putText(img, "Click to set start point of first line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif current_line == "line1_end":
                cv2.putText(img, "Click to set end point of first line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif current_line == "line2_start":
                cv2.putText(img, "Click to set start point of second line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif current_line == "line2_end":
                cv2.putText(img, "Click to set end point of second line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw the counting lines if they have been defined
        if line1_start and line1_end:
            cv2.line(img, line1_start, line1_end, (0, 0, 255), 2)
        if line2_start and line2_end:
            cv2.line(img, line2_start, line2_end, (255, 0, 0), 2)

        # Run YOLO model on the current frame to detect people
        results = model(img, stream=True)
        detections = np.empty((0, 5))

        # Parse the results to extract bounding boxes and other details
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Get coordinates of the bounding box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = box.conf[0]  # Confidence of the detection
                cls = int(box.cls[0])
                if cls < len(classNames):
                    currentClass = classNames[cls]
                    if currentClass == "person" and conf > 0.3:
                        currentArray = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((detections, currentArray))

        # Update tracker with the current detections
        resultsTracker = tracker.update(detections)

        # Draw tracked bounding boxes and IDs on the frame
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            pyresearch.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            pyresearch.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                                   scale=2, thickness=3, offset=10)

            # Calculate the center point of the bounding box
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Check if the person crosses either of the counting lines
            if line1_start and line1_end and line2_start and line2_end:
                if intersect((cx, cy), (cx, cy-1), line1_start, line1_end):
                    if id not in entrance1_count:
                        entrance1_count.append(id)
                        cv2.line(img, line1_start, line1_end, (0, 255, 0), 5)  # Highlight line on crossing
                        people_in_lab += 1
                elif intersect((cx, cy), (cx, cy-1), line2_start, line2_end):
                    if id not in entrance2_count:
                        entrance2_count.append(id)
                        cv2.line(img, line2_start, line2_end, (0, 255, 0), 5)  # Highlight line on crossing
                        people_in_lab = max(0, people_in_lab - 1)

        # Display the count of visitors and current occupancy
        cv2.putText(img, f"Total Visitors: {len(entrance2_count)}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.putText(img, f"Currently in Lab: {people_in_lab}", (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow("Webcam People Counter", img)

    except Exception as e:
        print(f"An error occurred: {e}")
        continue

    # Handle key events for quitting and resetting
    key = cv2.waitKey(1) & 0xFF
    if key == ord('f'):
        toggle_fullscreen()
    elif key == ord('q'):
        break
    elif key == ord('r'):
        # Reset all counts and lines if 'r' key is pressed
        line1_start = line1_end = line2_start = line2_end = None
        current_line = "line1_start"
        entrance1_count.clear()
        entrance2_count.clear()
        people_in_lab = 0
        print("Lines reset. Click to set the start point of the first counting line")

# Release resources when done
cap.release()
cv2.destroyAllWindows()
