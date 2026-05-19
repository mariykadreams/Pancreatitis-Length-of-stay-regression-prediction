# Residuals plot - Real Model & Data
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

# Calculate residuals
residuals = y_pred - y_true

# Plot residuals vs fitted values
plt.figure(figsize=(10, 7))
plt.scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
plt.axhline(0, linestyle='--', linewidth=2.5, color='red', label='Zero Error')

plt.xlabel('Fitted Values (Predicted LOS)', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residual Plot (Homoscedasticity Check)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('residuals_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Mean Residual: {np.mean(residuals):.4f}")
print(f"Std Dev of Residuals: {np.std(residuals):.4f}")
