#include <WiFi.h>
#include <HTTPClient.h>
#include "DHT.h"
#include <TFT_eSPI.h>
#include <ArduinoJson.h>

// === CONFIGURATION ===
#define DHTPIN 4  // GPIO4 for ESP32
#define DHTTYPE DHT11
#define SOIL_ADC 36  // GPIO36 (VP) for ESP32

const char* ssid     = "RMRaikar";
const char* password = "RMRaikar@777";
const String server_url = "https://idp-final-mustard-groundnut-1.onrender.com";

// === Objects ===
DHT dht(DHTPIN, DHTTYPE);
TFT_eSPI tft = TFT_eSPI();

String cropType = "groundnut";  // default
bool readyToSend = false;
unsigned long lastUpdateTime = 0;
const unsigned long sensorInterval = 2000; // update every 2 seconds

float temp = 0;
float hum = 0;
int adcValue = 0;

// === Helper Functions ===
void centerText(String text, int y, uint16_t color = TFT_WHITE, int textSize = 2) {
  int charWidth = 6;
  int textWidth = text.length() * charWidth * textSize;
  int x = (tft.width() - textWidth) / 2;

  tft.setTextSize(textSize);
  tft.setTextColor(color, TFT_BLACK);
  tft.setCursor(x, y);
  tft.print(text);
}

void drawMainFrame() {
  tft.fillScreen(TFT_BLACK);
  tft.drawRoundRect(5, 5, tft.width() - 10, tft.height() - 10, 8, TFT_WHITE);

  centerText("Grain Moisture Analyzer", 10, TFT_GREEN, 2);
  centerText("Grain: " + cropType, 40, TFT_CYAN, 2);
  centerText("Temp: " + String(temp, 1) + " C", 70, TFT_WHITE, 2);
  centerText("Hum: " + String(hum, 1) + " %", 100, TFT_WHITE, 2);
  centerText("ADC: " + String(adcValue), 130, TFT_WHITE, 2);

  centerText("Type 'read' to send", 160, TFT_YELLOW, 2);
  centerText("Type crop name:", 190, TFT_YELLOW, 2);
  centerText("mustard / groundnut", 215, TFT_YELLOW, 2);
}

void testAPIConnection() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected for API test.");
    return;
  }

  HTTPClient http;
  String url = server_url + "/models";
  
  http.begin(url);
  int httpCode = http.GET();

  if (httpCode > 0) {
    String response = http.getString();
    Serial.println("API Models Response: " + response);
    
    DynamicJsonDocument doc(512);
    DeserializationError error = deserializeJson(doc, response);
    
    if (!error) {
      if (doc.containsKey("available_models")) {
        bool groundnutAvailable = doc["available_models"]["groundnut"]["available"];
        bool mustardAvailable = doc["available_models"]["mustard"]["available"];
        String modelType = doc["model_type"];
        
        Serial.println("Groundnut model: " + String(groundnutAvailable ? "Available" : "Not available"));
        Serial.println("Mustard model: " + String(mustardAvailable ? "Available" : "Not available"));
        Serial.println("Model type: " + modelType);
        
        tft.fillScreen(TFT_BLACK);
        centerText("API Connected", 20, TFT_GREEN, 2);
        centerText("Groundnut: " + String(groundnutAvailable ? "OK" : "FAIL"), 50, groundnutAvailable ? TFT_GREEN : TFT_RED);
        centerText("Mustard: " + String(mustardAvailable ? "OK" : "FAIL"), 80, mustardAvailable ? TFT_GREEN : TFT_RED);
        centerText("Type: " + modelType, 110, TFT_CYAN);
        delay(3000);
      }
    }
  } else {
    Serial.println("API test failed: " + String(httpCode));
    tft.fillScreen(TFT_BLACK);
    centerText("API Test Failed", 60, TFT_RED, 2);
    delay(2000);
  }
  
  http.end();
}

// === Setup ===
void setup() {
  Serial.begin(115200);
  delay(1000);

  tft.init();
  tft.setRotation(1);

  tft.fillScreen(TFT_BLACK);
  centerText("Connecting to WiFi", 60, TFT_CYAN);
  
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected.");
  centerText("WiFi Connected", 90, TFT_GREEN);
  delay(1000);

  dht.begin();
  drawMainFrame();

  Serial.println("Type 'mustard' or 'groundnut' to select crop.");
  Serial.println("Type 'read' to take reading and send to model.");

  // Test API connection and show model status
  testAPIConnection();
  
  drawMainFrame();
}

// === Loop ===
void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input == "mustard" || input == "groundnut") {
      cropType = input;
      Serial.println("Crop set to: " + cropType);
      drawMainFrame(); // Update UI with new crop
    } else if (input == "read") {
      readyToSend = true;
    } else if (input == "test") {
      Serial.println("Testing API connection...");
      testAPIConnection();
      drawMainFrame();
    } else {
      Serial.println("Unknown command. Use: mustard/groundnut/read/test");
    }
  }

  // Update sensor values every few seconds
  if (millis() - lastUpdateTime > sensorInterval) {
    lastUpdateTime = millis();

    float t = dht.readTemperature();
    float h = dht.readHumidity();
    int adc = analogRead(SOIL_ADC);

    if (!isnan(t) && !isnan(h)) {
      temp = t;
      hum = h;
      adcValue = adc;
      drawMainFrame();
    } else {
      Serial.println("Sensor error.");
    }
  }

  // Send values to model when requested
  if (readyToSend) {
    Serial.println("Sending to model...");
    sendToModel(adcValue, temp, hum);
    readyToSend = false;
  }
}

// === HTTP Request ===
void sendToModel(int adc, float temp, float hum) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected.");
    tft.fillScreen(TFT_BLACK);
    centerText("WiFi Error", 60, TFT_RED, 2);
    delay(2000);  // Show error briefly
    drawMainFrame();
    return;
  }

  HTTPClient http;

  // Use the main predict endpoint with crop_type parameter
  String url = server_url + "/predict";
  
  // Create JSON payload with crop_type
  String jsonPayload = "{\"adc\":" + String(adc) + 
                      ",\"temperature\":" + String(temp, 1) + 
                      ",\"humidity\":" + String(hum, 1) + 
                      ",\"crop_type\":\"" + cropType + "\"}";

  Serial.println("Sending to: " + url);
  Serial.println("Payload: " + jsonPayload);

  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  
  int httpCode = http.POST(jsonPayload);

  tft.fillScreen(TFT_BLACK);
  centerText("Model Response", 20, TFT_GREEN, 2);

  if (httpCode > 0) {
    String response = http.getString();
    Serial.println("HTTP Code: " + String(httpCode));
    Serial.println("Raw Response:\n" + response);

    // Parse JSON response
    DynamicJsonDocument doc(1024);
    DeserializationError error = deserializeJson(doc, response);

    if (!error) {
      if (doc.containsKey("prediction")) {
        float moisture = doc["prediction"]["moisture_percentage"];
        String modelUsed = doc["prediction"]["model_used"];
        String modelType = doc["prediction"]["model_type"];
        
        Serial.println("Moisture: " + String(moisture, 2) + "%");
        Serial.println("Model: " + modelUsed);
        Serial.println("Model Type: " + modelType);
        
        centerText("Moisture Content", 60, TFT_YELLOW);
        centerText(String(moisture, 2) + "%", 90, TFT_WHITE, 3);
        centerText("Model: " + modelUsed, 130, TFT_CYAN, 1);
        centerText("Type: " + modelType, 150, TFT_CYAN, 1);
        
      } else if (doc.containsKey("error")) {
        String errorMsg = doc["message"];
        Serial.println("Error: " + errorMsg);
        centerText("Error: " + errorMsg, 80, TFT_RED, 1);
      }
    } else {
      Serial.println("JSON parsing failed: " + String(error.c_str()));
      centerText("Invalid response", 80, TFT_RED);
    }

  } else {
    Serial.println("HTTP error: " + String(httpCode));
    centerText("HTTP Error: " + String(httpCode), 80, TFT_RED);
  }

  http.end();
  delay(3000);  // Show result for 3 seconds
  drawMainFrame();
} 