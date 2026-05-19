# Error distribution plot - Real Model & Data
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

# Calculate errors
errors = y_pred - y_true

# Plot histogram of errors
plt.figure(figsize=(10, 7))
plt.hist(errors, bins=40, alpha=0.7, edgecolor='black', color='steelblue')
plt.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2.5, label=f'Mean = {np.mean(errors):.2f}')
plt.axvline(np.median(errors), color='green', linestyle='--', linewidth=2.5, label=f'Median = {np.median(errors):.2f}')

plt.xlabel('Prediction Error (days)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Print statistics (without scipy)
print(f"Mean Error: {np.mean(errors):.4f}")
print(f"Median Error: {np.median(errors):.4f}")
print(f"Std Dev: {np.std(errors):.4f}")
