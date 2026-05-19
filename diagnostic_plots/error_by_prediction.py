# Prediction error by predicted value - Real Model & Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# Load model and data
model_path = Path(__file__).parent.parent / 'final_model_pipeline.joblib'
data_path = Path(__file__).parent.parent / 'train.csv'

try:
    import joblib
    model = joblib.load(model_path)
except:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

df = pd.read_csv(data_path, index_col=0)

# Separate features and target
y_true = df['Length of stay'].values
X = df.drop('Length of stay', axis=1)

# Generate predictions
y_pred = model.predict(X)

# Calculate absolute errors
abs_errors = np.abs(y_pred - y_true)

# Plot absolute error vs predicted values
plt.figure(figsize=(10, 7))
plt.scatter(y_pred, abs_errors, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)

# Add trend line
z = np.polyfit(y_pred, abs_errors, 2)
p = np.poly1d(z)
y_trend = p(np.sort(y_pred))
plt.plot(np.sort(y_pred), y_trend, "r--", linewidth=2.5, label='Trend (degree 2)')

plt.xlabel('Predicted LOS (days)', fontsize=12)
plt.ylabel('Absolute Error (days)', fontsize=12)
plt.title('Absolute Error vs Predicted Values', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('error_by_prediction.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Mean Absolute Error: {np.mean(abs_errors):.4f}")
print(f"Median Absolute Error: {np.median(abs_errors):.4f}")
