import cv2
import numpy as np
import base64
import json

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
    Detect vehicles in an image and count them per row.
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

                boxes.append([x, y, w, h, center_y])  # Include center_y for annotation
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes([box[:4] for box in boxes], confidences, 0.6, 0.3)

    row_counts = [{"row_id": i + 1, "cars": 0, "trucks": 0} for i in range(len(ROW_BOUNDARIES))]
    annotated_boxes = []

    for i in indices.flatten():
        x, y, w, h, center_y = boxes[i]
        label = classes[class_ids[i]]

        # Determine the row of the detection
        for row_id, (top, bottom) in enumerate(ROW_BOUNDARIES):
            if top <= y + h // 2 < bottom:  # Vehicle is in this row
                if label == "car":
                    row_counts[row_id]["cars"] += 1
                elif label in ["bus", "truck"]:
                    row_counts[row_id]["trucks"] += 1

        # Store bounding box and center_y for visualization
        annotated_boxes.append((x, y, w, h, label, center_y))

    return row_counts, annotated_boxes

# Visualize detections on the image
def visualize_detections(image, annotated_boxes):
    for x, y, w, h, label, center_y in annotated_boxes:
        # Draw the bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Annotate with the label and center_y
        text = f"{label} y={center_y}"
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return image

# Main logic to test without MQTT
if __name__ == "__main__":
    image_path = "images/image2.jpg"  # Replace with your image path
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image.")
    else:
        # Process the image
        row_counts, annotated_boxes = detect_vehicles(image, traffic_id=1)

        # Visualize the detections
        result_image = visualize_detections(image, annotated_boxes)

        # Display the results
        print("Row Counts:", json.dumps(row_counts, indent=4))
        cv2.imshow("Vehicle Detections", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
