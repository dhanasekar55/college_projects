import cv2
import torch
from django.shortcuts import render
from django.http import StreamingHttpResponse

# Load YOLOv5 model from PyTorch Hub for object detection
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to detect people and count them
def detect_people(frame):
    # Run object detection on the frame using YOLOv5
    results = yolo_model(frame)
    detections = results.pandas().xyxy[0].to_dict(orient="records")

    # Filter detections to only include 'person' objects
    people_detections = [obj for obj in detections if obj['name'] == 'person']
    people_count = len(people_detections)  # Count only people

    # Draw bounding boxes for detected people
    for obj in people_detections:
        x1, y1, x2, y2 = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
        confidence = obj['confidence']
        
        # Draw bounding box for person (Green color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Overlay people count on the frame
    cv2.putText(frame, f"People Count: {people_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, people_count  # Return the frame with bounding boxes and people count


# Function to detect all objects with bounding boxes
def detect_objects(frame):
    # Run object detection on the frame using YOLOv5
    results = yolo_model(frame)
    detections = results.pandas().xyxy[0].to_dict(orient="records")

    # Draw bounding boxes for all detected objects
    for obj in detections:
        x1, y1, x2, y2 = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
        confidence = obj['confidence']
        label = obj['name']

        # Set color for bounding boxes
        if label == 'person':
            color = (0, 255, 0)  # Green for people
        else:
            color = (255, 0, 0)  # Red for other objects

        # Draw bounding box for detected object
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame  # Return the frame with bounding boxes for all detected objects


# View for streaming people count video
def video_feed_people_count(request):
    def gen_people_count():
      
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Unable to connect to the camera.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            # Detect people and count them
            frame, people_count = detect_people(frame)

            # Encode the frame in JPEG format
            _, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()

            # Yield the frame for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        cap.release()

    return StreamingHttpResponse(gen_people_count(), content_type='multipart/x-mixed-replace; boundary=frame')


# View for streaming object detection video
def video_feed_object_detection(request):
    def gen_object_detection():
        # Construct RTSP camera URL
        # cctv_url = f'rtsp://{camera_config["username"]}:{camera_config["password"]}@{camera_config["ip_address"]}:{camera_config["port"]}/{camera_config["path"]}'
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Unable to connect to the camera.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            # Detect all objects
            frame = detect_objects(frame)

            # Encode the frame in JPEG format
            _, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()

            # Yield the frame for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        cap.release()

    return StreamingHttpResponse(gen_object_detection(), content_type='multipart/x-mixed-replace; boundary=frame')


# Home view for displaying the HTML page
def home(request):
    return render(request, 'home.html')

def main(request):
    return render(request, 'main.html')

