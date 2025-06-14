#!/usr/bin/env python3
from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf

main_app = Flask(__name__)

# Load models and scalers
models = {
    'groundnut': {
        'model': tf.keras.models.load_model('model_groundnut.h5', compile=False),
        'scaler': joblib.load('scaler_groundnut.pkl')
    },
    'mustard': {
        'model': tf.keras.models.load_model('model_mustard.h5', compile=False),
        'scaler': joblib.load('scaler_mustard.pkl')
    }
}

# Compile models
for crop in models:
    models[crop]['model'].compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='mse',
        metrics=['mae']
    )

@main_app.route('/predict/<crop>', methods=['GET'])
def predict(crop):
    """
    GET /predict/<crop>?adc=...&temp=...&hum=...
    crop = groundnut or mustard
    """
    if crop not in models:
        return jsonify({'error': f'Unknown crop: {crop}'}), 400

    try:
        adc  = float(request.args['adc'])
        temp = float(request.args['temp'])
        hum  = float(request.args['hum'])
    except (KeyError, ValueError):
        return jsonify({'error': 'Invalid parameters'}), 400

    scaler = models[crop]['scaler']
    model = models[crop]['model']

    X = np.array([[adc, temp, hum]])
    Xs = scaler.transform(X)
    y = model.predict(Xs, verbose=0)[0, 0]
    return jsonify({'moisture': round(float(y), 2)})

@main_app.route('/update/<crop>', methods=['POST'])
def update(crop):
    """
    POST /update/<crop>
    JSON: { "adc":…, "temp":…, "hum":…, "moisture":… }
    """
    if crop not in models:
        return jsonify({'error': f'Unknown crop: {crop}'}), 400

    data = request.get_json(force=True)
    try:
        adc      = float(data['adc'])
        temp     = float(data['temp'])
        hum      = float(data['hum'])
        moisture = float(data['moisture'])
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': 'Invalid JSON body'}), 400

    X_new = np.array([[adc, temp, hum]])
    y_new = np.array([moisture])

    scaler = models[crop]['scaler']
    model = models[crop]['model']

    Xs = scaler.transform(X_new)
    model.fit(Xs, y_new, epochs=1, verbose=0)
    model.save(f'model_{crop}.h5')

    return jsonify({'status': f'{crop} model updated'})

if __name__ == '__main__':
    main_app.run(host='0.0.0.0', port=5000)
