import cv2
import numpy as np
import base64
import json
import paho.mqtt.client as mqtt
from datetime import datetime

# YOLO model setup (Assuming you have YOLOv4 model files)
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Constants for rows (Row boundaries as pixel heights)
ROW_BOUNDARIES = [
    (0, 200),   # Row 1 (top boundary, bottom boundary)
    (200, 400), # Row 2
    (400, 600), # Row 3
    (600, 800)  # Row 4 (if applicable)
]

def detect_vehicles(image, traffic_id):
    """
    Detect vehicles in an image and count them per row, and also count the total number of cars and trucks.
    """
    height, width, _ = image.shape

    # Pre-process the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Analyze detections
    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6 and classes[class_id] in ["car", "bus", "truck"]:
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.3)

    # Count vehicles in each row
    row_counts = [{"row_id": i + 1, "cars": 0, "trucks": 0} for i in range(len(ROW_BOUNDARIES))]

    # Initialize total vehicle counters
    total_cars = 0
    total_trucks = 0

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]

        # Determine the row of the detection based on the y-coordinate
        for row_id, (top, bottom) in enumerate(ROW_BOUNDARIES):
            if top <= y + h // 2 < bottom:  # Vehicle is in this row
                if label == "car":
                    row_counts[row_id]["cars"] += 1
                    total_cars += 1
                elif label in ["bus", "truck"]:
                    row_counts[row_id]["trucks"] += 1
                    total_trucks += 1

    # Add the total count to the result
    result = {
        "traffic_id": traffic_id,
        "rows": row_counts,
        "total": {
            "cars": total_cars,
            "trucks": total_trucks
        }
    }

    return result

# Callback function to handle MQTT messages for each traffic light
def on_message(client, userdata, msg):
    topic = msg.topic
    print(f"Message received on topic: {topic}")  # Print the topic

    # Printing the payload of the message
    try:
        payload = json.loads(msg.payload.decode())
        print(f"Payload: {json.dumps(payload, indent=4)}")  # Print the JSON payload
    except json.JSONDecodeError:
        print("Payload could not be decoded as JSON.")

    traffic_id = None

    if topic == 'traffic/light/image1':
        traffic_id = 1
    elif topic == 'traffic/light/image2':
        traffic_id = 2
    elif topic == 'traffic/light/image3':
        traffic_id = 3
    elif topic == 'traffic/light/image4':
        traffic_id = 4

    if traffic_id:
        image_data = payload['image']
        
        # Decode the base64 image
        img_data = base64.b64decode(image_data)
        np_img = np.frombuffer(img_data, dtype=np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Call the vehicle detection function
        traffic_entry = detect_vehicles(image, traffic_id)

        # Send the result back over MQTT to the corresponding topic
        client.publish(f'traffic/vehicle_count{traffic_id}', json.dumps(traffic_entry))

        # Save the message to a JSON file
        save_message_to_file(traffic_entry)



import os

def save_message_to_file(message):
    """
    Save the incoming message to a JSON file as an array of objects.
    Create the file if it doesn't exist and ensure proper JSON formatting.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message["timestamp"] = timestamp

    try:
        # File path
        file_path = 'traffic_data.json'

        # Check if the file exists, if not create an empty JSON array
        if not os.path.exists(file_path):
            with open(file_path, 'w') as file:
                json.dump([], file)

        # Read existing data
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)  # Load existing JSON data
            except json.JSONDecodeError:
                data = []  # Initialize as empty list if file is invalid

        # Append the new message to the list
        data.append(message)

        # Write the updated list back to the file
        with open(file_path, 'w') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        print("Message saved successfully.")
    except Exception as e:
        print(f"Error saving message: {e}")


# MQTT setup
client = mqtt.Client()
client.on_message = on_message
client.connect("broker.emqx.io", 1883, 60)

# Subscribe to all the traffic light image topics
client.subscribe("traffic/light/image1")
client.subscribe("traffic/light/image2")
client.subscribe("traffic/light/image3")
client.subscribe("traffic/light/image4")

# Start the MQTT loop
client.loop_forever()
