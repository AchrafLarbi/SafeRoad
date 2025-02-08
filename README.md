# SafeRoad API Documentation

SafeRoad is an AI-powered traffic management system integrating **smart infrastructure**, a **mobile app**, and an **administrative dashboard**. This documentation focuses on the **API** components utilizing **MQTT, OpenCV, and YOLO models**.

## 📡 MQTT-Based Communication
SafeRoad uses **MQTT** to enable real-time communication between traffic sensors, smart signals, and the central system.

### 🔧 MQTT Broker Setup
- Broker: **Eclipse Mosquitto**
- Host: `mqtt.safroad.com`
- Port: `1883` (default) / `8883` (SSL)
- Topics:
  - `safroad/traffic/data` → Traffic sensor updates
  - `safroad/alerts` → Real-time congestion alerts
  - `safroad/speed_limit` → Dynamic speed adjustments

### 📝 Sample MQTT Message (Traffic Data)
```json
{
  "sensor_id": "T123",
  "location": "Main St & 5th Ave",
  "vehicles": 25,
  "congestion_level": "high",
  "timestamp": "2025-02-08T12:34:56Z"
}
```

## 🎥 OpenCV for Traffic Analysis
OpenCV is used for **vehicle detection and traffic flow analysis** from live camera feeds.

### 🚗 Vehicle Detection with OpenCV
```python
import cv2

cap = cv2.VideoCapture('traffic_feed.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Traffic Analysis', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

## 🔍 YOLO for Object Detection
SafeRoad integrates **YOLOv8** to detect and classify vehicles in real time.

### 🏎 YOLO Model Inference
```python
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
frame = cv2.imread('traffic.jpg')
results = model(frame)
results.show()
```

### 🏙 YOLO Detected Objects
- **Car**
- **Bus**
- **Motorcycle**
- **Traffic Signs**

## 🚀 API Endpoints
### 📌 `POST /api/traffic-data`
**Description:** Receives processed traffic data from MQTT and OpenCV.

**Request Body:**
```json
{
  "sensor_id": "T123",
  "vehicles": 25,
  "congestion_level": "high",
  "yolo_detections": ["car", "bus"]
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Traffic data received"
}
```

---
### 📌 `GET /api/live-traffic`
**Description:** Retrieves real-time traffic updates.

**Response:**
```json
{
  "location": "Main St & 5th Ave",
  "congestion_level": "moderate",
  "vehicles": 18
}
```

## 📡 Deployment
- **MQTT Broker**: Mosquitto
- **Backend**: FastAPI / Flask
- **AI Models**: YOLOv8, OpenCV

## 🔗 Contributing
Feel free to submit issues and pull requests to improve SafeRoad!

## 📜 License
MIT License
