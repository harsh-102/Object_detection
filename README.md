## This repository includes 2 instances of object detection 
# AI-Powered Traffic Management System
This project is a real-time traffic light simulation system powered by YOLO (You Only Look Once) for vehicle detection.
It uses live webcam input, processes frames with a deep learning model, and dynamically adjusts traffic light states (red, yellow, green) based on vehicle density.
Features
 .Real-time vehicle detection using YOLOv8.
 .Adaptive traffic light simulation (red, yellow, green) based on traffic congestion.
 .Multi-threaded frame capture and processing for smooth performance.
 .Logging system to track events and errors (traffic_system.log).
 .Configurable thresholds and model path via config.json.
# Object Detection & GPS Position Estimation
This project combines YOLO-based object detection, depth estimation (MiDaS), and geospatial triangulation to approximate the GPS position of detected objects from a live webcam feed.
The system detects objects in real time, estimates their distance using depth maps, and calculates approximate GPS coordinates relative to known landmarks.
Features
 .Real-time object detection using YOLOv5 (small model for CPU efficiency).
 .Depth estimation powered by MiDaS (monocular depth prediction).
 .GPS triangulation with known landmarks to estimate object positions.
 .Multi-threaded depth inference for faster performance.
 .Visual overlay with bounding boxes, confidence scores, and GPS coordinates.
