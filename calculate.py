import cv2
import json
import numpy as np
import paho.mqtt.client as mqtt

# Constants for timing
MIN_TIME = 2         # Minimum green light time in seconds
MAX_TIME = 5         # Maximum green light time in seconds
CAR_THRESHOLD = 4    # Cars count threshold for heavy traffic
TRUCK_WEIGHT = 2.5   # Weighting factor for trucks
MAX_TRAFFIC_TIME = 30  # Maximum allowed total time for each traffic light (not per row)
STOP_LINE_DISTANCE = 2  # Example threshold for detecting vehicle crossing the stop line

# Load YOLO
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# MQTT Configuration
MQTT_BROKER = "broker.emqx.io"  # Replace with your broker address
MQTT_PORT = 1883
MQTT_TOPIC = "traffic/light/status"

def detect_vehicles(image_path):
    """
    Detect vehicles in an image using YOLO model.
    :param image_path: Path to the image file.
    :return: Integer count of vehicles detected.
    """
    # Load the image
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    
    # Prepare the image for YOLO model (scale and normalize)
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists to store detected vehicle details
    class_ids = []
    confidences = []
    boxes = []

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter for vehicles (class_id 2 corresponds to "car" in COCO dataset)
            if confidence > 0.5 and class_id == 2:  # 2 corresponds to 'car' class in COCO dataset
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression to remove duplicate boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Return the number of detected vehicles
    return len(indexes)

def allocate_time_per_row(cars_count, trucks_count, weather):
    """
    Allocate time based on the number of cars and trucks in a row.
    Trucks are weighted more heavily than cars.
    Weather conditions also affect the time allocation.
    """
    effective_vehicle_count = cars_count + int(trucks_count * TRUCK_WEIGHT)
    print(f"Effective Vehicle Count (Cars + Trucks): {effective_vehicle_count}")  # Debugging line
    
    # Adjust time based on weather conditions
    weather_factor = 1.0
    if weather == "rainy":
        weather_factor = 1.5
    elif weather == "snowy":
        weather_factor = 2
    elif weather == "foggy":
        weather_factor = 1.5

    if effective_vehicle_count <= 0:
        return MIN_TIME  # No vehicles, allocate minimum time
    elif effective_vehicle_count >= CAR_THRESHOLD:
        # Proportional time with 1.5 seconds per vehicle, capped at MAX_TIME
        time = MIN_TIME + (effective_vehicle_count - 1) * 1.5 * weather_factor
        return min(time, MAX_TIME)
    else:
        return MIN_TIME * weather_factor  # Less than threshold, allocate minimum time

def allocate_time(traffic_data):
    """
    Allocate time for each traffic light based on the rows' vehicle counts and weather conditions.
    The max time is applied per traffic light, not per row.
    """
    traffic_timings = []
    for traffic in traffic_data:
        total_time = 0
        for row in traffic["rows"]:
            # Calculate row time considering both cars and trucks and weather
            row_time = allocate_time_per_row(row["cars"], row["trucks"], "foggy")
            print(f"Row {row['row_id']} Time: {row_time}")  # Debugging line
            total_time += row_time
        
        # Apply the max time for each traffic light, not per row
        total_time = min(total_time, MAX_TRAFFIC_TIME)

        traffic_timings.append({
            "traffic_id": traffic["traffic_id"],
            "time": total_time
        })
    return traffic_timings

def detect_violation(vehicle_count, light_state):
    """
    Detect if the vehicle has violated traffic rules.
    :param vehicle_count: Integer count of vehicles detected in the image.
    :param light_state: Boolean - True if the light is green, False if red.
    :return: String indicating violation status.
    """
    if not light_state and vehicle_count > 0:
        return f"Violation: {vehicle_count} vehicles present while light is red!"
    return f"No Violation: {vehicle_count} vehicles detected."

def load_traffic_data(json_file):
    """
    Load traffic data from a JSON file.
    """
    with open(json_file, "r") as file:
        return json.load(file)

def send_timings_over_mqtt(client, timing_results):
    """
    Send timing results over MQTT.
    :param client: MQTT client instance.
    :param timing_results: List of timing data.
    """
    for timing in timing_results:
        message = json.dumps(timing)  # Convert timing data to JSON string
        client.publish(MQTT_TOPIC, message)
        print(f"Published to {MQTT_TOPIC}: {message}")

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    # You can subscribe to a topic if needed
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    print(f"Received message: {msg.payload.decode()}")

def main():
    # Load traffic data from JSON
    json_file_path = "traffic_data.json"
    traffic_data = load_traffic_data(json_file_path)
    
    # Calculate timings
    timing_results = allocate_time(traffic_data)
    
    # Setup MQTT client
    client = mqtt.Client()
    
    # Set up MQTT callbacks
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"Failed to connect to MQTT broker: {e}")
        return
    
    # Start MQTT loop in the background
    client.loop_start()

    # Publish timings over MQTT
    send_timings_over_mqtt(client, timing_results)
    
    # Detect violation example (optional)
    image_path = "images/image2.jpg"
    vehicle_count = detect_vehicles(image_path)
    light_state = False
    violation_result = detect_violation(vehicle_count, light_state)
    
    # Prepare data to write to JSON
    output_data = {
        "traffic_timings": timing_results,
        "violation_check": violation_result
    }
    
    # Write results to JSON file
    with open("output_results.json", "w") as output_file:
        json.dump(output_data, output_file, indent=4)

    print("Results have been written to output_results.json")

if __name__ == "__main__":
    main()
