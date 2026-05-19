# =============================
# IMPORTS
# =============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.set_printoptions(precision=3, suppress=True)


# =============================
# DATEN LADEN
# =============================

train_data = pd.read_csv(
    r"C:\Project Informatics\prediction of length of stay regression\train.csv",
    index_col=0
)

outcome = "Length of stay"

y = train_data[outcome]
X = train_data.drop(columns=[outcome])

# ID ist nur eine Identifikationsnummer und soll nicht als Feature verwendet werden
if "ID" in X.columns:
    X = X.drop(columns=["ID"])


# =============================
# TRAIN / VALIDATION SPLIT
# =============================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =============================
# DUMMY BASELINE MODEL
# =============================
# Der DummyRegressor ignoriert die Eingabevariablen vollständig.
# Mit strategy="median" sagt er für jeden Patienten
# immer den Median der Aufenthaltsdauer aus den Trainingsdaten voraus.

model = DummyRegressor(
    strategy="median"
)


# =============================
# MODELL TRAINIEREN
# =============================

model.fit(X_train, y_train)


# =============================
# VALIDATION PREDICTION
# =============================

y_pred = model.predict(X_val)

# Negative Aufenthaltsdauer wäre theoretisch nicht möglich,
# kommt beim Median aber ohnehin nicht vor.
y_pred = np.clip(y_pred, 1, None)


# =============================
# MODELLAUSWERTUNG
# =============================

mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print("\n" + "=" * 60)
print("LOCAL VALIDATION RESULTS")
print("=" * 60)
print("Model: Dummy Regressor")
print("Strategy: Median baseline")
print("Target transformation: None")
print("Main metric: Mean Absolute Error")
print("MAE:  %.4f days" % mae)
print("RMSE: %.4f days" % rmse)
print("R²:   %.4f" % r2)


# =============================
# AUSGEGEBENER KONSTANTER WERT
# =============================

constant_prediction = y_pred[0]

print("\n" + "=" * 60)
print("DUMMY BASELINE PREDICTION")
print("=" * 60)
print("Predicted value for every patient: %.4f days" % constant_prediction)


# =============================
# CROSS-VALIDATION
# =============================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="neg_mean_absolute_error"
)

cv_mae_scores = -cv_scores

print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)
print("MAE Scores:", cv_mae_scores)
print("Mean CV MAE: %.4f (+/- %.4f)" % (np.mean(cv_mae_scores), np.std(cv_mae_scores)))


# =============================
# GRAPHISCHE VISUALISIERUNG
# =============================

residuals = y_val - y_pred
absolute_errors = np.abs(residuals)


# 1. Echte Werte vs. Vorhersagen
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred, alpha=0.7)

min_value = min(y_val.min(), y_pred.min())
max_value = max(y_val.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Length of Stay")
plt.ylabel("Predicted Length of Stay")
plt.title("Actual vs. Predicted Length of Stay\nDummy Regressor: Median Baseline")
plt.grid(True)
plt.tight_layout()
plt.show()


# 2. Residual Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Length of Stay")
plt.ylabel("Residuals: Actual - Predicted")
plt.title("Residual Plot\nDummy Regressor: Median Baseline")
plt.grid(True)
plt.tight_layout()
plt.show()


# 3. Verteilung der absoluten Fehler
plt.figure(figsize=(8, 6))
plt.hist(absolute_errors, bins=20)

plt.xlabel("Absolute Prediction Error in Days")
plt.ylabel("Number of Patients")
plt.title("Distribution of Absolute Prediction Errors\nDummy Regressor: Median Baseline")
plt.grid(True)
plt.tight_layout()
plt.show()


# =============================
# SUMMARY FOR REPORT
# =============================

print("\n" + "=" * 60)
print("SUMMARY FOR REPORT")
print("=" * 60)
print("Problem type: Regression")
print("Target variable: Length of stay")
print("Evaluation metric: Mean Absolute Error")
print("Model: Dummy Regressor")
print("Strategy: Median baseline")
print("Target transformation: None")
print("Validation strategy: Train-validation split and 5-fold cross-validation")
print("Mean CV MAE: %.4f days" % np.mean(cv_mae_scores))