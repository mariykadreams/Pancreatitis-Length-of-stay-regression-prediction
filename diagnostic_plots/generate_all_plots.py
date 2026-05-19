# Generate all diagnostic plots from predictions
# This script generates predictions and saves all diagnostic plots
# Run from main project directory: python diagnostic_plots/generate_all_plots.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import re

def sanitize_columns(df):
    """Convert special characters in column names to underscores."""
    new_names = {}
    for col in df.columns:
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', col)
        safe = re.sub(r'_+', '_', safe).strip('_')
        new_names[col] = safe
    return df.rename(columns=new_names)

def engineer_features(df):
    """Apply feature engineering transformations."""
    df = df.copy()
    
    # Temporal deltas
    temporal_groups = [
        ("PCR",        ["PCR_Adm", "PCR_48h", "PCR_72h"]),
        ("Creat",      ["Creat_Adm", "Creat_48h", "Creat_72h"]),
        ("Urea",       ["Urea_Adm", "Urea_48h", "Urea_72h"]),
        ("Leukocytes", ["Leukocytes_Adm", "Leukocytes_48h", "Leukocytes_72h"]),
        ("Hct",        ["Hct_Adm", "Hct_48h", "Hct_72h"]),
    ]
    for base, cols in temporal_groups:
        existing = [c for c in cols if c in df.columns]
        adm_cols = [c for c in existing if "Adm" in c]
        follow_cols = [c for c in existing if "48h" in c or "72h" in c]
        if adm_cols and follow_cols:
            adm = adm_cols[0]
            for fc in follow_cols:
                df[f"{base}_delta_{fc.split('_')[-1]}"] = df[fc] - df[adm]
    
    # Ratios
    if "PCR_Adm" in df.columns and "Albumin" in df.columns:
        df["PCR_Albumin_ratio"] = df["PCR_Adm"] / df["Albumin"].replace(0, np.nan)
    
    if "PMN_Adm" in df.columns and "Lymphocytes_Adm" in df.columns:
        df["NLR_adm"] = df["PMN_Adm"] / df["Lymphocytes_Adm"].replace(0, np.nan)
    
    if "PMN_48h" in df.columns and "Lymph_48h" in df.columns:
        df["NLR_48h"] = df["PMN_48h"] / df["Lymph_48h"].replace(0, np.nan)
    
    # Composite severity
    sev = [c for c in ["BISAP", "Ransom_Adm", "SIRS", "CCI"] if c in df.columns]
    if sev:
        df["severity_composite"] = df[sev].fillna(0).sum(axis=1)
    
    # Missingness flags
    HIGH_MISSING_COLS = [
        "PCR_72h", "Creat_72h", "Hct_72h", "PMN_72h", "Lymph_72h",
        "Leukocytes_72h", "Urea_72h", "Eosinophils_72h", "Mono_72h",
        "PMN_Lymph_72h", "Col_total", "Amilasa", "PCO2",
        "Exceso_Bases", "CO3_H", "Gasometr_a_Ph",
    ]
    for col in HIGH_MISSING_COLS:
        if col in df.columns:
            df[f"{col}_observed"] = df[col].notna().astype(int)
    
    return df

print("Loading data...")
data_path = Path(__file__).parent.parent / 'train.csv'
df = pd.read_csv(data_path, index_col=0)

# Drop ID column if present
if "ID" in df.columns:
    df.drop(columns=["ID"], inplace=True)

# Sanitize column names
df = sanitize_columns(df)

# Engineer features
df = engineer_features(df)

# Separate target
target_col = "Length_of_stay"  # sanitized name
y_true = df[target_col].values
X = df.drop(target_col, axis=1)

print("Loading model... (this may take a moment)")
try:
    model_path = Path(__file__).parent.parent / 'final_model_pipeline.joblib'
    import joblib
    model = joblib.load(model_path)
except Exception as e:
    print(f"Warning: Could not load model with joblib: {e}")
    print("Please run this from the project root: python -c \"from pancreatitis_ml import model; import joblib; joblib.dump(model, 'final_model_pipeline.joblib')\"")
    sys.exit(1)

print("Generating predictions...")
y_pred = model.predict(X)

print("\n" + "="*50)
print("DIAGNOSTIC PLOTS GENERATED")
print("="*50)

# 1. BLAND-ALTMAN PLOT
print("\n[1/6] Creating Bland-Altman Plot...")
mean = (y_true + y_pred) / 2
diff = y_pred - y_true
bias = np.mean(diff)
std = np.std(diff)
loa_upper = bias + 1.96 * std
loa_lower = bias - 1.96 * std

plt.figure(figsize=(10, 7))
plt.scatter(mean, diff, alpha=0.6, s=30)
plt.axhline(bias, linestyle='--', linewidth=2, label=f'Bias = {bias:.2f}')
plt.axhline(loa_upper, linestyle='--', color='red', linewidth=2, label=f'+1.96 SD = {loa_upper:.2f}')
plt.axhline(loa_lower, linestyle='--', color='red', linewidth=2, label=f'-1.96 SD = {loa_lower:.2f}')
plt.xlabel('Mean LOS (days)', fontsize=12)
plt.ylabel('Prediction Error (days)', fontsize=12)
plt.title('Bland–Altman Plot', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('diagnostic_plots/01_bland_altman_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Bias: {bias:.4f} | Std Dev: {std:.4f}")

# 2. ACTUAL VS PREDICTED
print("[2/6] Creating Actual vs Predicted Plot...")
mae = np.mean(np.abs(y_true - y_pred))
mse = np.mean((y_true - y_pred) ** 2)
rmse = np.sqrt(mse)
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - (ss_res / ss_tot)

plt.figure(figsize=(10, 8))
plt.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect Prediction')
plt.xlabel('Actual LOS (days)', fontsize=12)
plt.ylabel('Predicted LOS (days)', fontsize=12)
plt.title('Actual vs Predicted Values', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('diagnostic_plots/02_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ R² Score: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

# 3. RESIDUALS PLOT
print("[3/6] Creating Residuals Plot...")
residuals = y_pred - y_true

plt.figure(figsize=(10, 7))
plt.scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
plt.axhline(0, linestyle='--', linewidth=2.5, color='red', label='Zero Error')
plt.xlabel('Fitted Values (Predicted LOS)', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residual Plot (Homoscedasticity Check)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('diagnostic_plots/03_residuals_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Mean Residual: {np.mean(residuals):.4f} | Std Dev: {np.std(residuals):.4f}")

# 4. ERROR DISTRIBUTION
print("[4/6] Creating Error Distribution Plot...")
errors = y_pred - y_true

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
plt.savefig('diagnostic_plots/04_error_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Mean Error: {np.mean(errors):.4f} | Median: {np.median(errors):.4f}")

# 5. Q-Q PLOT (manual implementation)
print("[5/6] Creating Q-Q Plot...")
sorted_residuals = np.sort(residuals)
n = len(sorted_residuals)
theoretical_quantiles = np.sort(np.random.standard_normal(n))

plt.figure(figsize=(10, 7))
plt.scatter(theoretical_quantiles, sorted_residuals, alpha=0.6, s=50)
min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5)
plt.title('Q-Q Plot - Normality Assessment', fontsize=14, fontweight='bold')
plt.xlabel('Theoretical Quantiles', fontsize=12)
plt.ylabel('Sample Quantiles', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('diagnostic_plots/05_qq_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Q-Q plot created")

# 6. ERROR BY PREDICTION
print("[6/6] Creating Error by Prediction Plot...")
abs_errors = np.abs(y_pred - y_true)

plt.figure(figsize=(10, 7))
plt.scatter(y_pred, abs_errors, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
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
plt.savefig('diagnostic_plots/06_error_by_prediction.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Mean Absolute Error: {np.mean(abs_errors):.4f}")

print("\n" + "="*50)
print("✓ ALL PLOTS SAVED TO: diagnostic_plots/")
print("="*50)
print("\nFiles created:")
for i in range(1, 7):
    print(f"   0{i}_*.png")
