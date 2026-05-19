# Bland-Altman Plot - Real Model & Data
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

# Calculate bland-altman data
mean = (y_true + y_pred) / 2
diff = y_pred - y_true

bias = np.mean(diff)
std = np.std(diff)

loa_upper = bias + 1.96 * std
loa_lower = bias - 1.96 * std

# Plot it
plt.figure(figsize=(10, 7))
plt.scatter(mean, diff, alpha=0.6, s=30)
plt.axhline(bias, linestyle='--', linewidth=2, label=f'Bias = {bias:.2f}')
plt.axhline(loa_upper, linestyle='--', color='red', linewidth=2, label=f'+1.96 SD = {loa_upper:.2f}')
plt.axhline(loa_lower, linestyle='--', color='red', linewidth=2, label=f'-1.96 SD = {loa_lower:.2f}')

plt.xlabel('Mean LOS (days)')
plt.ylabel('Prediction Error (days)')
plt.title('Bland–Altman Plot - Model Performance')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('bland_altman_plot.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"Bias: {bias:.4f} | Std Dev: {std:.4f}")
