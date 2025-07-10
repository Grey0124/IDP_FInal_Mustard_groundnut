#include <WiFi.h>
#include "DHT.h"
#include <TFT_eSPI.h>
#include <ArduinoJson.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// === CONFIGURATION ===
#define DHTPIN 4  // GPIO4 for ESP32
#define DHTTYPE DHT11
#define SOIL_ADC 36  // GPIO36 (VP) for ESP32

const char* ssid     = "RMRaikar";
const char* password = "RMRaikar@777";

// === TFLite Model Data ===
// These will be the converted TFLite models as byte arrays
extern const unsigned char mustard_model_tflite[] asm("_binary_mustard_model_tflite_start");
extern const unsigned char groundnut_model_tflite[] asm("_binary_groundnut_model_tflite_start");
extern const unsigned int mustard_model_tflite_len asm("mustard_model_tflite_size");
extern const unsigned int groundnut_model_tflite_len asm("groundnut_model_tflite_size");

// === TFLite Objects ===
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// === Model Configuration ===
constexpr int kTensorArenaSize = 100 * 1024;  // 100KB for model
uint8_t tensor_arena[kTensorArenaSize];

// === Objects ===
DHT dht(DHTPIN, DHTTYPE);
TFT_eSPI tft = TFT_eSPI();

String cropType = "groundnut";  // default
bool modelLoaded = false;
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

  centerText("Type 'read' to predict", 160, TFT_YELLOW, 2);
  centerText("Type crop name:", 190, TFT_YELLOW, 2);
  centerText("mustard / groundnut", 215, TFT_YELLOW, 2);
  
  if (modelLoaded) {
    centerText("Model: Loaded", 240, TFT_GREEN, 1);
  } else {
    centerText("Model: Not Loaded", 240, TFT_RED, 1);
  }
}

// === TFLite Functions ===
bool loadModel(const String& crop) {
  // Free previous model if loaded
  if (interpreter != nullptr) {
    delete interpreter;
    interpreter = nullptr;
  }
  
  // Select model based on crop type
  const unsigned char* model_data;
  unsigned int model_size;
  
  if (crop == "mustard") {
    model_data = mustard_model_tflite;
    model_size = mustard_model_tflite_len;
    Serial.println("Loading mustard model...");
  } else if (crop == "groundnut") {
    model_data = groundnut_model_tflite;
    model_size = groundnut_model_tflite_len;
    Serial.println("Loading groundnut model...");
  } else {
    Serial.println("Unknown crop type");
    return false;
  }
  
  // Map the model into a usable data structure
  const tflite::Model* model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    return false;
  }
  
  // Pull in all the ops implementations
  static tflite::AllOpsResolver resolver;
  
  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  
  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return false;
  }
  
  // Get pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.println("Model loaded successfully!");
  return true;
}

float predictMoisture(int adc, float temperature, float humidity) {
  if (interpreter == nullptr || input == nullptr || output == nullptr) {
    Serial.println("Model not loaded!");
    return -1.0;
  }
  
  // Normalize inputs based on your model's expected ranges
  // These should match the normalization used during training
  float normalized_adc = (float)adc / 4095.0;  // ESP32 ADC is 12-bit (0-4095)
  float normalized_temp = (temperature - 20.0) / 30.0;  // Assuming range 20-50°C
  float normalized_hum = humidity / 100.0;  // Humidity is already 0-1
  
  // Set input tensor values
  input->data.f[0] = normalized_adc;
  input->data.f[1] = normalized_temp;
  input->data.f[2] = normalized_hum;
  
  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return -1.0;
  }
  
  // Get prediction from output tensor
  float prediction = output->data.f[0];
  
  // Denormalize prediction (assuming moisture range 5-25%)
  float moisture_percentage = prediction * 20.0 + 5.0;  // Scale to 5-25% range
  
  return moisture_percentage;
}

// === Setup ===
void setup() {
  Serial.begin(115200);
  delay(1000);

  tft.init();
  tft.setRotation(1);

  tft.fillScreen(TFT_BLACK);
  centerText("Initializing TFLite", 60, TFT_CYAN);
  
  // Initialize TFLite
  error_reporter->Report("Initializing TFLite...");
  
  // Load default model (groundnut)
  modelLoaded = loadModel(cropType);
  
  if (!modelLoaded) {
    centerText("Model Load Failed", 90, TFT_RED);
    delay(2000);
  } else {
    centerText("Model Loaded Successfully", 90, TFT_GREEN);
    delay(1000);
  }

  dht.begin();
  drawMainFrame();

  Serial.println("Type 'mustard' or 'groundnut' to select crop.");
  Serial.println("Type 'read' to take reading and predict moisture.");
}

// === Loop ===
void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input == "mustard" || input == "groundnut") {
      cropType = input;
      Serial.println("Loading model for: " + cropType);
      
      tft.fillScreen(TFT_BLACK);
      centerText("Loading " + cropType + " model", 60, TFT_CYAN);
      
      modelLoaded = loadModel(cropType);
      
      if (modelLoaded) {
        centerText("Model loaded successfully", 90, TFT_GREEN);
      } else {
        centerText("Model load failed", 90, TFT_RED);
      }
      
      delay(1000);
      drawMainFrame();
      
    } else if (input == "read") {
      if (!modelLoaded) {
        Serial.println("Model not loaded. Please select a crop first.");
        tft.fillScreen(TFT_BLACK);
        centerText("Model not loaded", 60, TFT_RED, 2);
        centerText("Select crop first", 90, TFT_RED, 2);
        delay(2000);
        drawMainFrame();
        return;
      }
      
      Serial.println("Predicting moisture content...");
      predictAndDisplay();
      
    } else {
      Serial.println("Unknown command. Use: mustard/groundnut/read");
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
}

void predictAndDisplay() {
  tft.fillScreen(TFT_BLACK);
  centerText("Predicting...", 60, TFT_CYAN, 2);
  
  float moisture = predictMoisture(adcValue, temp, hum);
  
  if (moisture >= 0) {
    Serial.println("Moisture: " + String(moisture, 2) + "%");
    Serial.println("Crop: " + cropType);
    Serial.println("ADC: " + String(adcValue));
    Serial.println("Temp: " + String(temp, 1) + "°C");
    Serial.println("Humidity: " + String(hum, 1) + "%");
    
    tft.fillScreen(TFT_BLACK);
    centerText("Moisture Content", 20, TFT_YELLOW, 2);
    centerText(String(moisture, 2) + "%", 60, TFT_WHITE, 4);
    centerText("Crop: " + cropType, 120, TFT_CYAN, 2);
    centerText("ADC: " + String(adcValue), 150, TFT_WHITE, 1);
    centerText("Temp: " + String(temp, 1) + "°C", 170, TFT_WHITE, 1);
    centerText("Hum: " + String(hum, 1) + "%", 190, TFT_WHITE, 1);
    centerText("TFLite Inference", 220, TFT_GREEN, 1);
    
  } else {
    Serial.println("Prediction failed!");
    tft.fillScreen(TFT_BLACK);
    centerText("Prediction Failed", 60, TFT_RED, 2);
    centerText("Check model loading", 90, TFT_RED, 2);
  }
  
  delay(5000);  // Show result for 5 seconds
  drawMainFrame();
} 