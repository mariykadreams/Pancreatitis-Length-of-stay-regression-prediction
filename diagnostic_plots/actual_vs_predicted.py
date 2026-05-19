# Actual vs Predicted plot - Real Model & Data
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

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)

# Perfect prediction line
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect Prediction')

plt.xlabel('Actual LOS (days)', fontsize=12)
plt.ylabel('Predicted LOS (days)', fontsize=12)
plt.title('Actual vs Predicted Values - Model Performance', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate metrics manually
mae = np.mean(np.abs(y_true - y_pred))
mse = np.mean((y_true - y_pred) ** 2)
rmse = np.sqrt(mse)
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
