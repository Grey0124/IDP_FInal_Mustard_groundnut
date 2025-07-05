/*
 * ESP32 Moisture Meter with API Integration
 * 
 * Features:
 * - Crop selection (Groundnut/Mustard)
 * - ADC reading from moisture sensor
 * - DHT11 temperature and humidity
 * - WiFi connectivity
 * - API communication with Render
 * - OLED display for readings
 * 
 * Hardware Connections:
 * - Moisture Sensor: GPIO 36 (ADC1_CH0)
 * - DHT11: GPIO 4
 * - OLED Display: I2C (SDA: GPIO 21, SCL: GPIO 22)
 * - Button for crop selection: GPIO 0
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// WiFi Configuration
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// API Configuration
const char* apiUrl = "https://your-render-app.onrender.com/predict";

// Pin Definitions
#define MOISTURE_SENSOR_PIN 36  // ADC1_CH0
#define DHT_PIN 4
#define BUTTON_PIN 0
#define OLED_SDA 21
#define OLED_SCL 22

// Display Configuration
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C

// Sensor Objects
DHT dht(DHT_PIN, DHT11);
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Global Variables
String selectedCrop = "groundnut";  // Default crop
bool buttonPressed = false;
unsigned long lastButtonPress = 0;
unsigned long lastReading = 0;
const unsigned long READING_INTERVAL = 5000;  // 5 seconds

// Moisture reading structure
struct MoistureReading {
  int adc;
  float temperature;
  float humidity;
  float moisture;
  String cropType;
  String modelUsed;
  bool success;
};

void setup() {
  Serial.begin(115200);
  
  // Initialize pins
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  // Initialize sensors
  dht.begin();
  
  // Initialize display
  if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println(F("SSD1306 allocation failed"));
  }
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  
  // Connect to WiFi
  connectToWiFi();
  
  // Show initial screen
  showInitialScreen();
  
  Serial.println("ESP32 Moisture Meter initialized!");
}

void loop() {
  // Handle button press for crop selection
  handleButtonPress();
  
  // Take readings at intervals
  if (millis() - lastReading >= READING_INTERVAL) {
    MoistureReading reading = takeMoistureReading();
    displayReading(reading);
    lastReading = millis();
  }
  
  delay(100);  // Small delay to prevent watchdog reset
}

void connectToWiFi() {
  Serial.print("Connecting to WiFi");
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("Connecting to WiFi...");
  display.display();
  
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    display.setCursor(attempts * 6, 20);
    display.print(".");
    display.display();
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
    
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("WiFi Connected!");
    display.setCursor(0, 20);
    display.println(WiFi.localIP().toString());
    display.display();
    delay(2000);
  } else {
    Serial.println("\nWiFi connection failed!");
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("WiFi Failed!");
    display.display();
  }
}

void handleButtonPress() {
  if (digitalRead(BUTTON_PIN) == LOW && !buttonPressed && 
      millis() - lastButtonPress > 1000) {
    buttonPressed = true;
    lastButtonPress = millis();
    
    // Toggle crop selection
    if (selectedCrop == "groundnut") {
      selectedCrop = "mustard";
    } else {
      selectedCrop = "groundnut";
    }
    
    Serial.print("Crop changed to: ");
    Serial.println(selectedCrop);
    
    // Show crop selection on display
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("Crop Selected:");
    display.setCursor(0, 20);
    display.println(selectedCrop);
    display.display();
    delay(1000);
  }
  
  if (digitalRead(BUTTON_PIN) == HIGH) {
    buttonPressed = false;
  }
}

MoistureReading takeMoistureReading() {
  MoistureReading reading;
  reading.success = false;
  
  // Read sensors
  reading.adc = analogRead(MOISTURE_SENSOR_PIN);
  reading.temperature = dht.readTemperature();
  reading.humidity = dht.readHumidity();
  reading.cropType = selectedCrop;
  
  // Validate sensor readings
  if (isnan(reading.temperature) || isnan(reading.humidity)) {
    Serial.println("Failed to read DHT sensor!");
    return reading;
  }
  
  // Send data to API
  if (WiFi.status() == WL_CONNECTED) {
    reading = sendToAPI(reading);
  } else {
    Serial.println("WiFi not connected!");
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
  
  Serial.println("Sending to API: " + jsonString);
  
  // Send POST request
  int httpResponseCode = http.POST(jsonString);
  
  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.println("API Response: " + response);
    
    // Parse response
    StaticJsonDocument<512> responseDoc;
    DeserializationError error = deserializeJson(responseDoc, response);
    
    if (!error) {
      if (responseDoc["status"] == "success") {
        reading.moisture = responseDoc["prediction"]["moisture_percentage"];
        reading.modelUsed = responseDoc["prediction"]["model_used"].as<String>();
        reading.success = true;
        
        Serial.print("Moisture: ");
        Serial.print(reading.moisture);
        Serial.println("%");
      } else {
        Serial.println("API Error: " + responseDoc["error"].as<String>());
      }
    } else {
      Serial.println("JSON parsing failed");
    }
  } else {
    Serial.print("HTTP Error: ");
    Serial.println(httpResponseCode);
  }
  
  http.end();
  return reading;
}

void displayReading(MoistureReading reading) {
  display.clearDisplay();
  
  // Header
  display.setCursor(0, 0);
  display.println("Moisture Meter");
  display.println("---------------");
  
  if (reading.success) {
    // Moisture reading
    display.setCursor(0, 16);
    display.setTextSize(2);
    display.print(reading.moisture, 1);
    display.setTextSize(1);
    display.println("%");
    
    // Crop and model info
    display.setCursor(0, 32);
    display.println(reading.cropType);
    display.println(reading.modelUsed);
    
    // Sensor values
    display.setCursor(0, 48);
    display.print("T:");
    display.print(reading.temperature, 1);
    display.print("C H:");
    display.print(reading.humidity, 0);
    display.println("%");
  } else {
    // Error display
    display.setCursor(0, 16);
    display.println("Reading Failed!");
    display.println("Check WiFi/API");
    
    // Show sensor values anyway
    display.setCursor(0, 48);
    display.print("ADC:");
    display.println(reading.adc);
  }
  
  display.display();
}

void showInitialScreen() {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("Moisture Meter");
  display.println("ESP32 + API");
  display.println("");
  display.println("Press button to");
  display.println("change crop type");
  display.println("");
  display.print("Current: ");
  display.println(selectedCrop);
  display.display();
  delay(3000);
}

// WiFi reconnection function
void checkWiFiConnection() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected. Reconnecting...");
    connectToWiFi();
  }
} 