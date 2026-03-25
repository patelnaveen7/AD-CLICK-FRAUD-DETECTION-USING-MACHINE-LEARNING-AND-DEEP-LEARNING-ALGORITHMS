from django.shortcuts import render, redirect
from django.http import JsonResponse

from clickfraud import settings
from detection.lstm import lstm_training
from detection.rfc import rfc_training


def home(request):
    return render(request, 'detection/home.html')

import os
import joblib
import numpy as np
import pandas as pd

from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def training(request):

        # ================= GET RFC RESULTS =================
    (
        rfc_acc,
        rfc_precision,
        rfc_recall,
        rfc_f1,
        rfc_y_test,
        rfc_prob
    ) = rfc_training()

    # ================= GET LSTM RESULTS =================
    (
        lstm_acc,
        lstm_precision,
        lstm_recall,
        lstm_f1,
        lstm_y_test,
        lstm_prob,
    ) = lstm_training()


    # ================= HYBRID FUSION =================
    min_len = min(len(rfc_prob), len(lstm_prob))

    hybrid_prob = (0.52 * rfc_prob[:min_len]) + (0.48 * lstm_prob[:min_len])
    print(hybrid_prob)
    hybrid_pred = (hybrid_prob >= 0.62).astype(int)

    y_true = rfc_y_test.iloc[:min_len]

    hybrid_acc = accuracy_score(y_true, hybrid_pred)
    hybrid_precision = precision_score(y_true, hybrid_pred, zero_division=0)
    hybrid_recall = recall_score(y_true, hybrid_pred, zero_division=0)
    hybrid_f1 = f1_score(y_true, hybrid_pred, zero_division=0)
    hybrid_auc = roc_auc_score(y_true, hybrid_prob)


    # ================= METRICS DICTIONARY =================
    metrics = {
        # RFC
        'rfc_accuracy': round(rfc_acc * 100, 2),
        'rfc_precision': round(rfc_precision * 100, 2),
        'rfc_recall': round(rfc_recall * 100, 2),
        'rfc_f1': round(rfc_f1 * 100, 2),

        # LSTM
        'lstm_accuracy': round(lstm_acc * 100, 2),
        'lstm_precision': round(lstm_precision * 100, 2),
        'lstm_recall': round(lstm_recall * 100, 2),
        'lstm_f1': round(lstm_f1 * 100, 2),

        # HYBRID
        'hybrid_accuracy': round(hybrid_acc * 100, 2),
        'hybrid_precision': round(hybrid_precision * 100, 2),
        'hybrid_recall': round(hybrid_recall * 100, 2),
        'hybrid_f1': round(hybrid_f1 * 100, 2),
        'hybrid_auc': round(hybrid_auc * 100, 2),
    }

    return render(request, 'detection/training.html', {'metrics': metrics})

def detect_click_type(features):
    if (
        features['click_duration'] < 0.3 or
        features['mouse_movement'] < 5 or
        features['scroll_depth'] == 0 or
        features['click_frequency'] > 10 or
        features['VPN_usage'] is True
    ):
        return "bot"
    return "human"

def get_alert_and_suggestion(probability):
    """
    probability: value between 0 and 1
    """

    prob_percent = probability * 100

    if prob_percent <= 20:
        return {
            "risk": "Very Low",
            "alert": "✅ Very Safe Click",
            "suggestion": "Allow user"
        }

    elif prob_percent <= 40:
        return {
            "risk": "NO Risk Found",
            "alert": "🟢 Low Risk Detected",
            "suggestion": "Its geniue link"
        }

    elif prob_percent <= 60:
        return {
            "risk": "Medium",
            "alert": "🟡 Suspicious Activity",
            "suggestion": "Apply soft verification"
        }

    elif prob_percent <= 90:
        return {
            "risk": "High",
            "alert": "🟠 High Risk Click",
            "suggestion": "Rate-limit or CAPTCHA"
        }

    else:
        return {
            "risk": "Critical",
            "alert": "🔴 Fraud Confirmed",
            "suggestion": "Block user / IP"
        }


import numpy as np
import joblib
from django.shortcuts import render
from tensorflow.keras.models import load_model

from .models import Ad, ClickEvent, TrainingMetrics
from .incremental_dataset import append_to_dataset


import os
import json
import numpy as np
import joblib

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from tensorflow.keras.models import load_model

from .models import ClickEvent
from .incremental_dataset import append_to_dataset


# =====================================================
# SAFE LABEL ENCODER (HANDLE UNSEEN VALUES)
# =====================================================
def safe_label_encode(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return 0


# =====================================================
# PREDICTION VIEW
# =====================================================
def prediction(request):

    # =================================================
    # POST → REAL-TIME FRAUD PREDICTION
    # =================================================
    if request.method == "POST":
        data = json.loads(request.body)

        # ---------------- LOAD MODELS ----------------
        rf = joblib.load("models/rfc_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        encoders = joblib.load("models/encoders.pkl")
        lstm = load_model("models/lstm_stable_fraud.keras")

        # ---------------- ENCODE CATEGORICAL ----------------
        device = safe_label_encode(encoders['device_type'], data['device_type'])
        browser = safe_label_encode(encoders['browser'], data['browser'])
        os_ = safe_label_encode(encoders['operating_system'], data['operating_system'])
        ref = safe_label_encode(encoders['referrer_url'], data['referrer_url'])
        page = safe_label_encode(encoders['page_url'], data['page_url'])

        # ---------------- FEATURE VECTOR ----------------
        X = np.array([[
            float(data['click_duration']),
            float(data['scroll_depth']),
            float(data['mouse_movement']),
            float(data['keystrokes_detected']),
            int(data['ad_position']),
            float(data['time_since_last_click']),
            device,
            browser,
            os_,
            int(data['click_frequency']),
            int(data['VPN_usage']),
            ref,
            page
        ]])

        # ---------------- SCALE ----------------
        X_scaled = scaler.transform(X)

        # ---------------- RFC ----------------
        rf_prob = rf.predict_proba(X_scaled)[0][1]

        # ---------------- LSTM ----------------
        seq = np.repeat(X_scaled, 5, axis=0).reshape(1, 5, X_scaled.shape[1])
        lstm_prob = float(lstm.predict(seq, verbose=0)[0][0])

        # ---------------- HYBRID FUSION ----------------
        hybrid_prob = (0.4 * rf_prob) + (0.6 * lstm_prob)

        # =================================================
        # CLICKS-PER-SECOND (CPS) BUSINESS RULE
        # =================================================
        time_window = max(float(data.get("time_since_last_click", 1.0)), 0.5)
        click_frequency = int(data.get("click_frequency", 0))
        print(click_frequency)
        clicks_per_second = click_frequency / time_window

        forced_label = None

        if data.get("is_manual_click", False) and click_frequency > 5:
            hybrid_prob = max(hybrid_prob, 0.90)
            forced_label = "fraud"

        # 🟢 Human safety override
        elif (
            data.get("is_manual_click", False)
            and click_frequency <= 5
            and data["mouse_movement"] > 10
            and data["scroll_depth"] > 0.2
        ):
            hybrid_prob = min(hybrid_prob, 0.30)

        # ---------------- FINAL LABEL ----------------
        label = forced_label if forced_label else (
            "fraud" if hybrid_prob >= 0.85 else "legit"
        )

        # ---------------- ALERT ----------------
        alert = get_alert_and_suggestion(hybrid_prob)

       
        append_to_dataset({
            "click_duration": float(data['click_duration']),
            "scroll_depth": float(data['scroll_depth']),
            "mouse_movement": float(data['mouse_movement']),
            "keystrokes_detected": int(data['keystrokes_detected']),
            "ad_position": int(data['ad_position']),
            "time_since_last_click": float(data['time_since_last_click']),
            "device_type": data['device_type'],
            "browser": data['browser'],
            "operating_system": data['operating_system'],
            "click_frequency": int(data['click_frequency']),
            "VPN_usage": int(data['VPN_usage']),
            "referrer_url": data['referrer_url'],
            "page_url": data['page_url'],
            "is_fraudulent": 1 if label == "fraud" else 0
        })

        # ---------------- RESPONSE ----------------
        return JsonResponse({
            "fraud_probability": round(hybrid_prob * 100, 2),
            "risk": alert['risk'],
            "alert": alert['alert'],
            "suggestion": alert['suggestion'],
            "label": label.upper(),
            "clicks_per_second": round(clicks_per_second, 2)
        })

    # =================================================
    # GET → LOAD ADS PAGE
    # =================================================
    ads_dir = os.path.join(settings.MEDIA_ROOT, "ads")
    ads = [
        f"/media/ads/{img}"
        for img in os.listdir(ads_dir)
        if img.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]

    return render(request, "detection/prediction.html", {"ads": ads})
