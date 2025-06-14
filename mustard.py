#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def main():
    # --- A) Load & preprocess full CSV (sensor’s own Moisture) ---
    df = pd.read_csv('Mustard_Data.csv').dropna()
    X_full = df[['ADC', 'Temperature', 'Humidity']].values
    y_full = df['Moisture'].values

    # --- B) Load your 7 true‑moisture reference points ---
    truth = np.array([
        [2881, 30.9, 59.0, 11.4],
        [2866, 30.8, 58.0, 11.5],
        [2941, 30.8, 59.0, 7.2],
        [2815, 30.8, 59.0, 11.6],
        [2928, 30.7, 58.0, 7.4],
        [2925, 30.7, 57.0, 7.5],
        [2775, 30.7, 57.0, 13.2],
        [2815, 30.7, 57.0, 12.0],
    ])
    X_truth = truth[:, :3]
    y_truth = truth[:, 3]

    # --- C) Fit scaler on full data & transform both sets ---
    scaler = StandardScaler().fit(X_full)
    X_full_s  = scaler.transform(X_full)
    X_truth_s = scaler.transform(X_truth)
    joblib.dump(scaler, 'scaler_mustard.pkl')
    print("[Scaler] Saved to scaler_mustard.pkl")

    # --- D) Build model architecture ---
    def build_model():
        m = models.Sequential([
            layers.Input(shape=(3,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        m.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return m

    model = build_model()

    # --- E) Pre‑train on the large sensor‑only dataset ---
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    model.fit(
        X_full_s, y_full,
        validation_split=0.1,
        epochs=50,
        batch_size=64,
        callbacks=[early_stop],
        verbose=2
    )

    # --- F) Re-compile with lower LR and fine‑tune on true‑moisture points ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae']
    )
    model.fit(
        X_truth_s, y_truth,
        epochs=200,
        batch_size=7,
        verbose=2
    )

    # --- G) Save final model ---
    model.save('model_mustard.h5')
    print("[Model] Saved to model_mustard.h5")

if __name__ == '__main__':
    main()
