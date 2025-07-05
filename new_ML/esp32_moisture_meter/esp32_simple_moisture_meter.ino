/*
 * ESP32 Simple Moisture Meter with API Integration
 * 
 * Features:
 * - Crop selection via Serial commands
 * - ADC reading from moisture sensor
 * - DHT11 temperature and humidity
 * - WiFi connectivity
 * - API communication with Render
 * - Serial output for readings
 * 
 * Hardware Connections:
 * - Moisture Sensor: GPIO 36 (ADC1_CH0)
 * - DHT11: GPIO 4
 * - LED indicator: GPIO 2
 * 
 * Serial Commands:
 * - 'g' or 'groundnut': Select groundnut crop
 * - 'm' or 'mustard': Select mustard crop
 * - 'r' or 'read': Take immediate reading
 * - 's' or 'status': Show current status
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>

// WiFi Configuration
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// API Configuration
const char* apiUrl = "https://your-render-app.onrender.com/predict";

// Pin Definitions
#define MOISTURE_SENSOR_PIN 36  // ADC1_CH0
#define DHT_PIN 4
#define LED_PIN 2

// Sensor Objects
DHT dht(DHT_PIN, DHT11);

// Global Variables
String selectedCrop = "groundnut";  // Default crop
unsigned long lastReading = 0;
const unsigned long READING_INTERVAL = 10000;  // 10 seconds
bool autoReading = true;

// Moisture reading structure
struct MoistureReading {
  int adc;
  float temperature;
  float humidity;
  float moisture;
  String cropType;
  String modelUsed;
  bool success;
  String error;
};

void setup() {
  Serial.begin(115200);
  
  // Initialize pins
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  
  // Initialize sensors
  dht.begin();
  
  // Show welcome message
  Serial.println("\n🌾 ESP32 Moisture Meter with API Integration");
  Serial.println("=============================================");
  Serial.println("Commands:");
  Serial.println("  'g' or 'groundnut' - Select groundnut crop");
  Serial.println("  'm' or 'mustard'   - Select mustard crop");
  Serial.println("  'r' or 'read'      - Take immediate reading");
  Serial.println("  's' or 'status'    - Show current status");
  Serial.println("  'a' or 'auto'      - Toggle auto reading");
  Serial.println("  'h' or 'help'      - Show this help");
  Serial.println();
  
  // Connect to WiFi
  connectToWiFi();
  
  // Show initial status
  showStatus();
  
  Serial.println("ESP32 Moisture Meter initialized!");
}

void loop() {
  // Handle serial commands
  handleSerialCommands();
  
  // Take automatic readings
  if (autoReading && millis() - lastReading >= READING_INTERVAL) {
    MoistureReading reading = takeMoistureReading();
    displayReading(reading);
    lastReading = millis();
  }
  
  delay(100);
}

void connectToWiFi() {
  Serial.print("Connecting to WiFi");
  digitalWrite(LED_PIN, HIGH);
  
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ WiFi connected!");
    Serial.print("📡 IP address: ");
    Serial.println(WiFi.localIP());
    digitalWrite(LED_PIN, LOW);
  } else {
    Serial.println("\n❌ WiFi connection failed!");
    digitalWrite(LED_PIN, HIGH);
  }
}

void handleSerialCommands() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    command.toLowerCase();
    
    if (command == "g" || command == "groundnut") {
      selectedCrop = "groundnut";
      Serial.println("✅ Crop changed to: Groundnut");
    }
    else if (command == "m" || command == "mustard") {
      selectedCrop = "mustard";
      Serial.println("✅ Crop changed to: Mustard");
    }
    else if (command == "r" || command == "read") {
      Serial.println("📊 Taking reading...");
      MoistureReading reading = takeMoistureReading();
      displayReading(reading);
      lastReading = millis();
    }
    else if (command == "s" || command == "status") {
      showStatus();
    }
    else if (command == "a" || command == "auto") {
      autoReading = !autoReading;
      Serial.print("🔄 Auto reading: ");
      Serial.println(autoReading ? "ON" : "OFF");
    }
    else if (command == "h" || command == "help") {
      Serial.println("\n🌾 Available Commands:");
      Serial.println("  'g' or 'groundnut' - Select groundnut crop");
      Serial.println("  'm' or 'mustard'   - Select mustard crop");
      Serial.println("  'r' or 'read'      - Take immediate reading");
      Serial.println("  's' or 'status'    - Show current status");
      Serial.println("  'a' or 'auto'      - Toggle auto reading");
      Serial.println("  'h' or 'help'      - Show this help");
    }
    else if (command.length() > 0) {
      Serial.println("❌ Unknown command. Type 'h' for help.");
    }
  }
}

MoistureReading takeMoistureReading() {
  MoistureReading reading;
  reading.success = false;
  reading.error = "";
  
  // Read sensors
  reading.adc = analogRead(MOISTURE_SENSOR_PIN);
  reading.temperature = dht.readTemperature();
  reading.humidity = dht.readHumidity();
  reading.cropType = selectedCrop;
  
  // Validate sensor readings
  if (isnan(reading.temperature) || isnan(reading.humidity)) {
    reading.error = "Failed to read DHT sensor";
    return reading;
  }
  
  // Send data to API
  if (WiFi.status() == WL_CONNECTED) {
    reading = sendToAPI(reading);
  } else {
    reading.error = "WiFi not connected";
    // Try to reconnect
    connectToWiFi();
  }
  
  return reading;
}

MoistureReading sendToAPI(MoistureReading reading) {
  HTTPClient http;
  http.begin(apiUrl);
  http.addHeader("Content-Type", "application/json");
  
  // Prepare JSON payload
  StaticJsonDocument<200> doc;
  doc["adc"] = reading.adc;
  doc["temperature"] = reading.temperature;
  doc["humidity"] = reading.humidity;
  doc["crop_type"] = reading.cropType;
  
  String jsonString;
  serializeJson(doc, jsonString);
  
  Serial.println("📤 Sending to API: " + jsonString);
  
  // Send POST request
  int httpResponseCode = http.POST(jsonString);
  
  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.println("📥 API Response: " + response);
    
    // Parse response
    StaticJsonDocument<512> responseDoc;
    DeserializationError error = deserializeJson(responseDoc, response);
    
    if (!error) {
      if (responseDoc["status"] == "success") {
        reading.moisture = responseDoc["prediction"]["moisture_percentage"];
        reading.modelUsed = responseDoc["prediction"]["model_used"].as<String>();
        reading.success = true;
      } else {
        reading.error = responseDoc["error"].as<String>();
      }
    } else {
      reading.error = "JSON parsing failed";
    }
  } else {
    reading.error = "HTTP Error: " + String(httpResponseCode);
  }
  
  http.end();
  return reading;
}

void displayReading(MoistureReading reading) {
  Serial.println("\n" + String("=", 50));
  Serial.println("🌾 MOISTURE METER READING");
  Serial.println(String("=", 50));
  Serial.printf("📅 Time: %s\n", getTimeString().c_str());
  Serial.printf("🌡️  Temperature: %.1f°C\n", reading.temperature);
  Serial.printf("💧 Humidity: %.1f%%\n", reading.humidity);
  Serial.printf("📊 ADC Reading: %d\n", reading.adc);
  Serial.printf("🌱 Crop Type: %s\n", reading.cropType.c_str());
  
  if (reading.success) {
    Serial.printf("🧠 Model: %s\n", reading.modelUsed.c_str());
    Serial.printf("💧 Moisture Content: %.2f%%\n", reading.moisture);
    Serial.println("✅ Prediction successful!");
  } else {
    Serial.printf("❌ Error: %s\n", reading.error.c_str());
  }
  Serial.println(String("=", 50));
}

void showStatus() {
  Serial.println("\n📊 SYSTEM STATUS");
  Serial.println("================");
  Serial.printf("🌱 Selected Crop: %s\n", selectedCrop.c_str());
  Serial.printf("📡 WiFi Status: %s\n", WiFi.status() == WL_CONNECTED ? "Connected" : "Disconnected");
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("🌐 IP Address: %s\n", WiFi.localIP().toString().c_str());
  }
  Serial.printf("🔄 Auto Reading: %s\n", autoReading ? "ON" : "OFF");
  Serial.printf("⏱️  Reading Interval: %d seconds\n", READING_INTERVAL / 1000);
  Serial.println("================\n");
}

String getTimeString() {
  unsigned long seconds = millis() / 1000;
  unsigned long minutes = seconds / 60;
  unsigned long hours = minutes / 60;
  seconds %= 60;
  minutes %= 60;
  
  char timeStr[20];
  sprintf(timeStr, "%02lu:%02lu:%02lu", hours, minutes, seconds);
  return String(timeStr);
}

// WiFi reconnection function
void checkWiFiConnection() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("📡 WiFi disconnected. Reconnecting...");
    connectToWiFi();
  }
} 