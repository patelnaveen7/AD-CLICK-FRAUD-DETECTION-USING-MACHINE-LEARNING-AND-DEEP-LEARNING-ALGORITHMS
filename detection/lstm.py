# =====================================================
# STABLE RN-BASED LSTM FOR CLICK FRAUD (ONE CELL)
# =====================================================

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
def lstm_training():
    # ===================== 1. LOAD =====================
    df = pd.read_csv("click_fraud_clened.csv")
    df = df.dropna(subset=['is_fraudulent'])
    df['is_fraudulent'] = df['is_fraudulent'].astype(int)

    FEATURES = [
        'click_duration',
        'scroll_depth',
        'mouse_movement',
        'keystrokes_detected',
        'ad_position',
        'time_since_last_click',
        'device_type',
        'browser',
        'operating_system',
        'click_frequency','VPN_usage',
        'referrer_url','page_url'
    ]

    df = df[FEATURES + ['is_fraudulent']]

    # ===================== 2. ENCODE =====================
    for col in ['device_type', 'browser', 'operating_system','referrer_url','page_url']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop('is_fraudulent', axis=1).values
    y = df['is_fraudulent'].values

    # ===================== 3. 🔴 LIGHT LABEL NOISE =====================
    np.random.seed(42)
    noise_ratio = 0.02   # 🔑 ONLY 2%

    flip_idx = np.random.choice(
        len(y),
        size=int(len(y) * noise_ratio),
        replace=False
    )
    y[flip_idx] = 1 - y[flip_idx]

    # ===================== 4. SCALE =====================
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ===================== 5. FEATURE-SEQUENCE CREATION =====================
    # Repeat feature vector across timesteps (RN-compatible)
    TIMESTEPS = 5

    X_seq = np.repeat(X[:, np.newaxis, :], TIMESTEPS, axis=1)

    # ===================== 6. SPLIT =====================
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ===================== 7. RN-BASED LSTM =====================
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(TIMESTEPS, X.shape[1])),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # ===================== 8. TRAIN =====================
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=2,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    # ===================== 9. EVALUATE =====================
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob > 0.5).astype(int)

   
    # ===================== 10. SAVE =====================
    model.save("models/lstm_stable_fraud.keras")

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    lstm_acc = accuracy_score(y_test, y_pred)
    lstm_precision = precision_score(y_test, y_pred, zero_division=0)
    lstm_recall = recall_score(y_test, y_pred, zero_division=0)
    lstm_f1 = f1_score(y_test, y_pred, zero_division=0)
    print("\n========== FINAL STABLE LSTM RESULTS ==========")
    print(f"Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob)*100:.2f}%")

    print("\n✅ LSTM model saved successfully")

    return (
        lstm_acc,
        lstm_precision,
        lstm_recall,
        lstm_f1,
        y_test,
        y_prob,
        
    )


