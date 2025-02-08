// MQTT Configuration
const mqttBroker = "wss://broker.emqx.io:8084/mqtt"; // Use an MQTT WebSocket broker
const topic = "traffic/light/status";

// Connect to MQTT broker
const client = mqtt.connect(mqttBroker);

client.on("connect", () => {
  console.log("Connected to MQTT broker");
  client.subscribe(topic, (err) => {
    if (!err) {
      console.log(`Subscribed to topic: ${topic}`);
    }
  });
});

// Handle incoming MQTT messages
client.on("message", (receivedTopic, message) => {
  if (receivedTopic === topic) {
    const status = message.toString();
    updateTrafficLight(status);
  }
});

// Update Traffic Light
function updateTrafficLight(status) {
  const redLight = document.getElementById("red-light");
  const yellowLight = document.getElementById("yellow-light");
  const greenLight = document.getElementById("green-light");

  // Reset all lights
  redLight.classList.remove("active");
  yellowLight.classList.remove("active");
  greenLight.classList.remove("active");

  // Activate the correct light
  if (status === "RED") {
    redLight.classList.add("active");
  } else if (status === "YELLOW") {
    yellowLight.classList.add("active");
  } else if (status === "GREEN") {
    greenLight.classList.add("active");
  }
}
