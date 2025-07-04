# ML Moisture Meter - Advanced Implementation

A comprehensive machine learning-based moisture meter system that predicts **original moisture meter readings** from custom meter data, ensuring high accuracy and reduced sensitivity to environmental fluctuations.

## 🌟 Key Objective

**Make custom moisture meter readings match original moisture meter accuracy** by:
- Using original moisture meter readings as the ground truth (target)
- Training ML models to predict original meter readings from custom meter data (ADC, Temperature, Humidity)
- Reducing the impact of temperature and humidity fluctuations on readings
- Providing stable and reliable moisture measurements

## 🌟 Features

- **Original Meter Accuracy**: Predicts readings that match the precision of the original moisture meter
- **Environmental Compensation**: Accounts for temperature and humidity variations to reduce fluctuations
- **Multi-Crop Support**: Works with Groundnut, Mustard, and extensible to other crops
- **Online Learning**: Continuous model improvement with new reference data
- **Raspberry Pi Ready**: Complete deployment package for embedded systems
- **Real-time Processing**: Fast inference for field use

## 📁 Project Structure

```
new_ML/
├── phase1_data_preparation.py      # Data cleaning and merging with original meter data
├── phase2_model_training.py        # ML model training to predict original meter readings
├── phase3_online_learning.py       # Online learning implementation
├── phase4_raspberry_pi_deployment.py # Deployment code generation
├── run_all_phases.py              # Complete pipeline execution
├── models/                        # Trained ML models
├── deployment/                    # Raspberry Pi/Arduino code
├── GroundNut_Data_custom_meter.csv
├── Mustard_Data_custom_meter.csv
├── Groundut_Original_moisture_meter_sample_1.csv
├── Groundut_Original_moisture_meter_sample_2.csv
├── Mustard_original_moisture_meter_sample_1.csv
├── Mustard_original_moisture_meter_sample_2.csv
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Python Environment**: Python 3.8+ with virtual environment
2. **Dependencies**: Install required packages
3. **Data**: Ensure your sensor data files are in the `new_ML/` directory

### Installation

```bash
# Activate your virtual environment
source myenv/bin/activate  # Linux/Mac
# or
myenv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Pipeline

```bash
# Run all phases sequentially
python new_ML/run_all_phases.py
```

### Run Individual Phases

```bash
# Phase 1: Data Preparation
python new_ML/phase1_data_preparation.py

# Phase 2: Model Training
python new_ML/phase2_model_training.py

# Phase 3: Online Learning
python new_ML/phase3_online_learning.py

# Phase 4: Deployment
python new_ML/phase4_raspberry_pi_deployment.py
```

## 📊 Phase Details

### Phase 1: Data Preparation & Exploration
- **Data Merging**: Combines custom meter data with original moisture meter readings
- **Data Cleaning**: Removes anomalies and outliers from custom meter data
- **Target Setup**: Uses original moisture meter readings as the target variable
- **Exploratory Analysis**: Visualizes relationships between custom meter data and original readings
- **Output**: Merged datasets with original meter readings as targets

### Phase 2: Model Training & Evaluation
- **Objective**: Train models to predict original moisture meter readings from custom meter data
- **Multiple Algorithms**: Tests Linear Regression, Random Forest, SVR, Neural Networks
- **Cross-Validation**: Ensures robust model performance
- **Hyperparameter Tuning**: Optimizes model parameters
- **Performance Metrics**: MAE, RMSE, R² evaluation against original meter readings
- **Output**: Trained models that predict original meter accuracy

### Phase 3: Online Learning Implementation
- **Objective**: Continuously improve prediction of original moisture meter readings
- **River Integration**: True online learning capabilities
- **Performance Monitoring**: Tracks model drift and accuracy against original meter
- **Real-time Updates**: Continuous model improvement with new reference data
- **Output**: Online learning models and performance tracking

### Phase 4: Raspberry Pi Deployment
- **Objective**: Deploy models that predict original moisture meter readings
- **C++ Code Generation**: Optimized for embedded systems
- **Arduino Compatibility**: Ready-to-deploy Arduino code
- **Model Optimization**: Linear approximations for fast inference
- **Output**: Complete deployment package

## 🔧 Hardware Requirements

### Sensors
- **Capacitive Soil Moisture Sensor V2.0** (custom meter)
- **DHT22/DHT11** Temperature and Humidity Sensor
- **Original Moisture Meter** (for reference/calibration)
- **Raspberry Pi** or **ESP32/Arduino**

### Pin Connections
- **Moisture Sensor**: GPIO 34 (ADC-capable pin)
- **DHT Sensor**: GPIO 4
- **Display**: I2C pins (optional)

## 📈 Model Performance

### Accuracy Metrics
- **Groundnut**: Predicts original meter readings within ±X% MAE
- **Mustard**: Predicts original meter readings within ±X% MAE
- **Combined Model**: Multi-crop support with crop-specific optimization

### Environmental Compensation
- **Temperature**: Accounts for thermal effects on sensor readings
- **Humidity**: Compensates for atmospheric moisture interference
- **Stability**: Reduces fluctuations compared to raw custom meter readings
- **Real-time Adaptation**: Continuous adjustment to changing conditions

## 🔄 Online Learning Features

### Continuous Improvement
- **Reference Data**: Uses new original meter readings for model updates
- **Data Streaming**: Processes new sensor readings in real-time
- **Model Updates**: Automatically adjusts to new patterns
- **Drift Detection**: Monitors for performance degradation against original meter
- **Performance Tracking**: Rolling and cumulative accuracy metrics

### Adaptive Capabilities
- **Seasonal Changes**: Adapts to different growing seasons
- **Sensor Aging**: Compensates for sensor degradation over time
- **Environmental Shifts**: Handles changing weather patterns
- **Reference Calibration**: Maintains alignment with original meter accuracy

## 🛠️ Deployment

### Arduino/ESP32 Deployment
1. Upload the generated `.ino` file to your device
2. Connect sensors according to pin definitions
3. Monitor readings via Serial Monitor
4. Use crop selection commands for multi-crop models
5. **Output**: Readings that match original moisture meter accuracy

### Raspberry Pi Deployment
1. Copy generated C++ files to your Pi
2. Compile the code using g++
3. Run the executable for real-time predictions
4. Integrate with your existing monitoring system
5. **Output**: Stable readings with reduced environmental sensitivity

## 📋 Usage Examples

### Basic Prediction
```python
from new_ML.models import load_model

# Load trained model
model = load_model('groundnut')

# Make prediction (outputs original meter equivalent)
moisture = model.predict(adc=2900, temperature=30.5, humidity=58)
print(f"Predicted original moisture: {moisture:.2f}%")
```

### Online Learning
```python
from new_ML.online_learning import OnlineLearner

# Initialize online learner
learner = OnlineLearner('groundnut')

# Update with new reference data
learner.update(adc=2900, temperature=30.5, humidity=58, original_moisture=15.2)

# Get current prediction
prediction = learner.predict(adc=2900, temperature=30.5, humidity=58)
```

## 🔍 Troubleshooting

### Common Issues
1. **Poor Accuracy**: Check sensor calibration and placement
2. **Compilation Errors**: Verify C++ compiler and dependencies
3. **Memory Issues**: Optimize model size for embedded systems
4. **Sensor Drift**: Use online learning to adapt to changes
5. **Reference Alignment**: Ensure original meter readings are accurate

### Performance Optimization
- **Model Size**: Use linear approximations for faster inference
- **Update Frequency**: Balance accuracy with computational cost
- **Memory Management**: Optimize for limited embedded resources
- **Reference Quality**: Use high-quality original meter readings for training

## 📚 Extending the System

### Adding New Crops
1. Collect custom meter data for the new crop
2. Obtain original moisture meter readings for the same samples
3. Run Phase 1 with the new dataset
4. Train models using Phase 2
5. Generate deployment code with Phase 4

### Custom Sensors
1. Modify the data loading functions
2. Update feature engineering as needed
3. Retrain models with new sensor data and reference readings
4. Regenerate deployment code

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original moisture meter manufacturers for reference data
- Open-source ML community for algorithms and tools
- Agricultural research institutions for validation support

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the generated documentation
3. Open an issue on the repository
4. Contact the development team

---

**Note**: This system is designed to make custom moisture meter readings match the accuracy of original moisture meters while reducing environmental sensitivity. It should be validated against standard moisture measurement methods before commercial use. 