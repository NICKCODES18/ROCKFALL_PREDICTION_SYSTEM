import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, brier_score_loss
import joblib

np.random.seed(42)

# Generate synthetic data
n = 30000
slope = np.random.uniform(0, 90, size=n)
rain = np.random.exponential(scale=15.0, size=n)
rain = np.clip(rain, 0, 150)

# True function with noise
slope_factor = np.interp(slope, [0, 45, 90], [0.05, 0.6, 1.0])
rain_factor = np.interp(rain, [0, 20, 100], [0.1, 0.6, 1.0])
true_risk = np.clip(slope_factor * (0.4 + 0.6 * rain_factor), 0, 1)

# Probabilistic labels to avoid hard thresholds
prob = np.clip(true_risk + np.random.normal(0, 0.08, size=n), 0, 1)
y = (np.random.rand(n) < prob).astype(int)

# Features (keep simple and smooth)
X = np.column_stack([
    slope,
    rain,
    slope_factor,
    rain_factor,
    slope * rain / 100.0,
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline with standardization and calibrated logistic regression
base = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=500, C=0.5, class_weight="balanced")
)
model = CalibratedClassifierCV(base, cv=3, method='sigmoid')
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
aps = average_precision_score(y_test, probs)
brier = brier_score_loss(y_test, probs)
print(f"AP: {aps:.3f} | Brier: {brier:.3f}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, os.path.join("models", "baseline.joblib"))
print("Saved models/baseline.joblib")
