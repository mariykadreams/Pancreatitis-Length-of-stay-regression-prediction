# Q-Q plot (Normality check) - Real Model & Data
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

# Manual Q-Q plot (without scipy.stats)
plt.figure(figsize=(10, 7))

# Sort residuals and calculate quantiles
sorted_residuals = np.sort(residuals)
n = len(sorted_residuals)
theoretical_quantiles = np.sort(np.random.standard_normal(n))

plt.scatter(theoretical_quantiles, sorted_residuals, alpha=0.6, s=50)

# Add diagonal line
min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

plt.title('Q-Q Plot - Normality Assessment of Residuals', fontsize=14, fontweight='bold')
plt.xlabel('Theoretical Quantiles', fontsize=12)
plt.ylabel('Sample Quantiles', fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('qq_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Mean Residual: {np.mean(residuals):.4f}")
print(f"Std Dev of Residuals: {np.std(residuals):.4f}")
