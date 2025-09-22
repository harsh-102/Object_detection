import cv2
import torch
import numpy as np
from ultralytics import YOLO
from geopy.distance import geodesic
import torchvision.transforms as T
from PIL import Image
import torch.backends.cudnn as cudnn
from concurrent.futures import ThreadPoolExecutor
import threading

# Tell the CPU to optimize itself
cudnn.benchmark = False
cudnn.deterministic = False

# Force to use CPU (even if a GPU is available)
device = torch.device("cpu")
print(f"Using device: {device}")

CAMERA_GPS = (37.7749, -122.4194)

# Set known building locations to help calculate object positions
LANDMARKS = {
    "Building A": (37.7755, -122.4185),
    "Building B": (37.7740, -122.4200)
}

try:
    # Load models
    model = YOLO("yolov5su.pt")  # Use YOLOv5s (smaller model for faster CPU performance)
    model.to(device)
    
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(device).eval()  # Ensure MiDaS is in evaluation mode
    
except Exception as e:
    print(f"Error loading models: {str(e)}")
    print("Make sure all necessary packages are installed.")
    

# Image processing steps
transform = T.Compose([
    T.Resize((640, 480), antialias=True),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Thread-local tensor storage
thread_local = threading.local()

def get_thread_local_tensor():
    if not hasattr(thread_local, 'tensor'):
        thread_local.tensor = torch.zeros((1, 3, 640, 480), dtype=torch.float32, device=device)
    return thread_local.tensor

@torch.inference_mode()
def estimate_depth(frame, box):
    try:
        x1, y1, x2, y2 = box
        object_image = frame[y1:y2, x1:x2]
        if object_image.shape[0] < 10 or object_image.shape[1] < 10:
            return 1.0
        object_image = Image.fromarray(cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB))
        input_tensor = get_thread_local_tensor()
        with torch.no_grad():
            input_data = transform(object_image).to(device, dtype=torch.float32, non_blocking=True)
            input_tensor[0] = input_data
            depth_map = midas(input_tensor)
            average_depth = torch.mean(depth_map).item()
        return max(average_depth, 1.0)
    except Exception as e:
        print(f"Problem measuring depth: {str(e)}")
        return 1.0

def triangulate_position(landmark1, landmark2, d1, d2):
    """Calculate the GPS position of an object using nearby landmarks"""
    try:
        # Get the coordinates of our reference buildings
        lat1, lon1 = landmark1
        lat2, lon2 = landmark2
        
        # Convert latitude to radians for math calculations
        lat1_rad = np.radians(lat1)
        
        # Calculate positions in meters instead of coordinates
        x1 = np.cos(lat1_rad) * 111320 * (lon1)
        y1 = lat1 * 111320
        x2 = np.cos(lat1_rad) * 111320 * (lon2)
        y2 = lat2 * 111320
        
        # Use weighted averages based on distance to estimate position
        total_distance = d1 + d2
        weight1 = d2 / total_distance
        weight2 = d1 / total_distance
        
        x_obj = (weight1 * x1 + weight2 * x2)
        y_obj = (weight1 * y1 + weight2 * y2)
        
        # Convert back to GPS coordinates
        return (y_obj / 111320, x_obj / (111320 * np.cos(lat1_rad)))
    except Exception as e:
        print(f"Problem calculating position: {str(e)}")
        return CAMERA_GPS


# Start the camera
cap = cv2.VideoCapture(0)  # Use 0 for built-in webcam
if not cap.isOpened():
    print("Couldn't access the camera.")
    exit(1)

executor = ThreadPoolExecutor(max_workers=2)  # Adjust threads for CPU usage

while True:
    ret, frame = cap.read()
    if not ret:
        print("Problem getting image from camera.")
        break

    results = model(frame, conf=0.4, iou=0.45, max_det=10)
    
    futures = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            object_type = model.names[int(box.cls[0])]
            if (x2 - x1) * (y2 - y1) < 100:
                continue
            future = executor.submit(estimate_depth, frame, (x1, y1, x2, y2))
            futures.append((future, (x1, y1, x2, y2), object_type, confidence))
    
    for future, box, object_type, confidence in futures:
        depth = future.result()
        x1, y1, x2, y2 = box
        obj_lat, obj_lon = triangulate_position(
            LANDMARKS["Building A"], LANDMARKS["Building B"], depth, depth + 20
        )
        color = (255, 0, 255) if object_type.lower() in ["drone", "airplane"] else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        texts = [
            (f"{object_type} ({confidence:.2f})", (x1, y1 - 20)),
            (f"GPS: {obj_lat:.6f}, {obj_lon:.6f}", (x1, y1 - 5))
        ]
        for text, pos in texts:
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Object Detection with GPS", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
