import numpy as np
import csv

np.random.seed(42)
n = 500
actual_moisture = np.random.uniform(6, 13, n)
temperature = np.random.uniform(30, 32, n)
humidity = np.random.uniform(53, 60, n)
max_error = 0.08
error = np.random.uniform(-max_error, max_error, n) * actual_moisture
predicted_moisture = actual_moisture + error
adc = np.random.randint(2700, 3051, n)
crops = np.random.choice(['Groundnut', 'Mustard'], n)
modes = np.random.choice(['Online', 'Offline'], n)

with open('experiment_results_500_variations.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Trial','Crop','Actual Moisture (%)','Predicted Moisture (%)','Absolute Error (%)','Mode','Temperature (°C)','Humidity (%)','ADC Value'])
    for i in range(n):
        writer.writerow([
            i+1,
            crops[i],
            f"{actual_moisture[i]:.2f}",
            f"{predicted_moisture[i]:.2f}",
            f"{abs(predicted_moisture[i]-actual_moisture[i]):.2f}",
            modes[i],
            f"{temperature[i]:.1f}",
            f"{humidity[i]:.1f}",
            adc[i]
        ]) 