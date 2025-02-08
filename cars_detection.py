import cv2
import numpy as np
import os
import json

# Load YOLO model
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Specify input folder and output JSON file
input_folder = "images/"
output_json = "traffic_results.json"

# Initialize results list
results = []
traffic_id = 1  # Start ID from 1

# Loop through all images in the folder
for image_name in os.listdir(input_folder):
    if image_name.endswith((".jpg", ".png", ".jpeg")):  # Check for valid image extensions
        # Load image
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)
        height, width, channels = image.shape

        # Pre-process image
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        # Forward pass
        outs = net.forward(output_layers)

        # Analyze detections
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.6:  # Higher confidence threshold for more reliable detections
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

        # Count cars and trucks
        car_count = 0
        truck_count = 0
        for i in indices.flatten():
            label = classes[class_ids[i]]
            if label == "car":
                car_count += 1
            elif label in ["bus", "truck"]:
                truck_count += 1

        # Append result for the current image with ID
        results.append({
            "traffic_id": traffic_id,  # Use unique ID
            "cars": car_count,
            "trucks": truck_count
        })

        # Increment traffic ID
        traffic_id += 1

# Save results to JSON file
with open(output_json, "w") as json_file:
    json.dump(results, json_file, indent=4)

print(f"Results saved to {output_json}")
