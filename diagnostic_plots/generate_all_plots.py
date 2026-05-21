# Generate all diagnostic plots from predictions
# Run from project root: python diagnostic_plots/generate_all_plots.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys
import re

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})

PLOTS_DIR = Path(__file__).parent
PLOTS_DIR.mkdir(exist_ok=True)


def sanitize_columns(df):
    new_names = {}
    for col in df.columns:
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', col)
        safe = re.sub(r'_+', '_', safe).strip('_')
        new_names[col] = safe
    return df.rename(columns=new_names)


def engineer_features(df):
    df = df.copy()
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
    if "PCR_Adm" in df.columns and "Albumin" in df.columns:
        df["PCR_Albumin_ratio"] = df["PCR_Adm"] / df["Albumin"].replace(0, np.nan)
    if "PMN_Adm" in df.columns and "Lymphocytes_Adm" in df.columns:
        df["NLR_adm"] = df["PMN_Adm"] / df["Lymphocytes_Adm"].replace(0, np.nan)
    if "PMN_48h" in df.columns and "Lymph_48h" in df.columns:
        df["NLR_48h"] = df["PMN_48h"] / df["Lymph_48h"].replace(0, np.nan)
    sev = [c for c in ["BISAP", "Ransom_Adm", "SIRS", "CCI"] if c in df.columns]
    if sev:
        df["severity_composite"] = df[sev].fillna(0).sum(axis=1)
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


# ── load data & model ────────────────────────────────────────────────────────

print("Loading data...")
data_path = Path(__file__).parent.parent / 'train.csv'
df = pd.read_csv(data_path, index_col=0)
if "ID" in df.columns:
    df.drop(columns=["ID"], inplace=True)
df = sanitize_columns(df)
df = engineer_features(df)

target_col = "Length_of_stay"
y_true = df[target_col].values.astype(float)
X = df.drop(target_col, axis=1)

print("Loading model...")
try:
    model_path = Path(__file__).parent.parent / 'final_model_pipeline.joblib'
    import joblib
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ensemble import WeightedEnsemble  # noqa: F401
    model = joblib.load(model_path)
except Exception as e:
    print(f"Could not load model: {e}")
    print("Re-run pancreatitis_ml.py first to regenerate the model file.")
    sys.exit(1)

print("Generating predictions...")
y_pred_log = model.predict(X)
y_pred = np.maximum(np.expm1(y_pred_log), 1.0)

# ── shared metrics ───────────────────────────────────────────────────────────

abs_err   = np.abs(y_true - y_pred)
errors    = y_pred - y_true
mae       = abs_err.mean()
rmse      = np.sqrt(((y_true - y_pred) ** 2).mean())
ss_res    = ((y_true - y_pred) ** 2).sum()
ss_tot    = ((y_true - y_true.mean()) ** 2).sum()
r2        = 1 - ss_res / ss_tot
within    = {n: (abs_err <= n).mean() * 100 for n in [2, 3, 5, 7, 10]}

print(f"\n  MAE={mae:.2f}d  RMSE={rmse:.2f}d  R²={r2:.3f}")
for n, pct in within.items():
    print(f"  Within {n:2d} days: {pct:.1f}%")

# LOS bucket labels (used in several plots)
bucket_edges  = [0, 3, 7, 14, 30, 200]
bucket_labels = ["1–3 d", "4–7 d", "8–14 d", "15–30 d", ">30 d"]
bucket_ids    = pd.cut(y_true, bins=bucket_edges, labels=bucket_labels)

print("\n" + "=" * 55)
print("GENERATING PLOTS")
print("=" * 55)


# ── 1. ACTUAL vs PREDICTED ──────────────────────────────────────────────────
# Coloured by absolute error; ±3d and ±7d accuracy bands shown.
print("\n[1/6] Actual vs Predicted (colour = absolute error)...")

fig, ax = plt.subplots(figsize=(8, 7))

vmax = np.percentile(abs_err, 90)
sc = ax.scatter(y_true, y_pred, c=abs_err, cmap="RdYlGn_r",
                vmin=0, vmax=vmax, s=40, alpha=0.75, edgecolors="none")
plt.colorbar(sc, ax=ax, label="Absolute error (days)")

lim = max(y_true.max(), y_pred.max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1.5, label="Perfect")
for band, color in [(3, "#2196F3"), (7, "#FF9800")]:
    ax.fill_between([0, lim], [0 - band, lim - band], [0 + band, lim + band],
                    alpha=0.08, color=color, label=f"±{band} day band")
    ax.plot([0, lim], [0 + band, lim + band], color=color, lw=0.8, ls="--")
    ax.plot([0, lim], [0 - band, lim - band], color=color, lw=0.8, ls="--")

ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Actual LOS (days)"); ax.set_ylabel("Predicted LOS (days)")
ax.set_title("Actual vs Predicted LOS", fontweight="bold")
ax.legend(fontsize=9, loc="upper left")

metrics_txt = (f"MAE  = {mae:.2f} d\nRMSE = {rmse:.2f} d\nR²   = {r2:.3f}\n"
               f"Within 3d: {within[3]:.1f}%\nWithin 7d: {within[7]:.1f}%")
ax.text(0.97, 0.05, metrics_txt, transform=ax.transAxes, fontsize=9,
        va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

plt.tight_layout()
fig.savefig(PLOTS_DIR / "01_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}")


# ── 2. WITHIN-N-DAYS ACCURACY ───────────────────────────────────────────────
# Horizontal bar chart — the most clinically meaningful accuracy metric.
print("[2/6] Within-N-days accuracy bar chart...")

thresholds = [1, 2, 3, 5, 7, 10, 14]
pcts = [(abs_err <= t).mean() * 100 for t in thresholds]
colors_bar = ["#d32f2f" if p < 50 else "#f57c00" if p < 70 else "#388e3c" for p in pcts]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh([f"Within {t} day{'s' if t > 1 else ''}" for t in thresholds],
               pcts, color=colors_bar, edgecolor="white", height=0.6)
ax.axvline(80, color="grey", lw=1.2, ls="--", label="80% reference")
ax.set_xlim(0, 105)
ax.set_xlabel("% of patients")
ax.set_title("Within-N-Days Prediction Accuracy", fontweight="bold")
for bar, pct in zip(bars, pcts):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%", va="center", fontsize=10)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "02_within_n_days_accuracy.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"   Within 3d: {within[3]:.1f}%  |  Within 7d: {within[7]:.1f}%")


# ── 3. ERROR BOXPLOTS BY LOS BUCKET ─────────────────────────────────────────
# Shows clearly where the model is accurate and where it struggles.
print("[3/6] Error boxplots by LOS bucket...")

groups = [errors[bucket_ids == label] for label in bucket_labels]
counts = [len(g) for g in groups]
non_empty = [(lbl, g, n) for lbl, g, n in zip(bucket_labels, groups, counts) if n > 0]

fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot([g for _, g, _ in non_empty], patch_artist=True, widths=0.5,
                medianprops=dict(color="black", lw=2))
palette = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0"]
for patch, color in zip(bp["boxes"], palette[:len(non_empty)]):
    patch.set_facecolor(color); patch.set_alpha(0.6)

ax.axhline(0, color="red", lw=1.5, ls="--", label="Zero error")
ax.set_xticks(range(1, len(non_empty) + 1))
ax.set_xticklabels([f"{lbl}\n(n={n})" for lbl, _, n in non_empty])
ax.set_ylabel("Prediction error (days)")
ax.set_title("Prediction Error by LOS Bucket\n"
             "(box = IQR, line = median, whiskers = 1.5×IQR)", fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "03_error_by_los_bucket.png", dpi=300, bbox_inches="tight")
plt.close(fig)
for lbl, g, n in non_empty:
    print(f"   {lbl:8s}: median error = {np.median(g):+.1f} d  (n={n})")


# ── 4. BLAND–ALTMAN (improved) ───────────────────────────────────────────────
# Points coloured by LOS bucket; polynomial trend shows heteroscedasticity.
print("[4/6] Bland–Altman plot with trend line...")

mean_vals = (y_true + y_pred) / 2
bias      = errors.mean()
sd        = errors.std()
loa_hi    = bias + 1.96 * sd
loa_lo    = bias - 1.96 * sd
pct_out   = ((errors > loa_hi) | (errors < loa_lo)).mean() * 100

fig, ax = plt.subplots(figsize=(9, 6))

bucket_colors = {"1–3 d": "#4CAF50", "4–7 d": "#2196F3",
                 "8–14 d": "#FF9800", "15–30 d": "#E91E63", ">30 d": "#9C27B0"}
for label in bucket_labels:
    mask = bucket_ids == label
    if mask.sum():
        ax.scatter(mean_vals[mask], errors[mask], s=35, alpha=0.7,
                   color=bucket_colors[label], label=label, edgecolors="none")

# LOA band
ax.axhspan(loa_lo, loa_hi, alpha=0.07, color="grey")
ax.axhline(bias,   ls="--", lw=2,   color="navy",  label=f"Bias = {bias:+.2f} d")
ax.axhline(loa_hi, ls="--", lw=1.5, color="crimson", label=f"+1.96 SD = {loa_hi:.2f} d")
ax.axhline(loa_lo, ls="--", lw=1.5, color="crimson", label=f"−1.96 SD = {loa_lo:.2f} d")

# Trend line (shows growing bias with LOS)
z = np.polyfit(mean_vals, errors, 1)
xfit = np.linspace(mean_vals.min(), mean_vals.max(), 200)
ax.plot(xfit, np.poly1d(z)(xfit), color="darkorange", lw=2, ls="-", label="Trend")

ax.set_xlabel("Mean of Actual & Predicted LOS (days)")
ax.set_ylabel("Prediction error  Predicted − Actual (days)")
ax.set_title(f"Bland–Altman Plot  ({pct_out:.1f}% outside LOA)", fontweight="bold")
ax.legend(fontsize=8, ncol=2, loc="lower left")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "04_bland_altman.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"   Bias={bias:+.2f}d  SD={sd:.2f}d  Outside LOA={pct_out:.1f}%")


# ── 5. ABSOLUTE ERROR DISTRIBUTION ──────────────────────────────────────────
# Histogram of absolute errors with cumulative line; annotates key percentiles.
print("[5/6] Absolute error distribution...")

fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

bins = np.arange(0, abs_err.max() + 2, 1)
ax1.hist(abs_err, bins=bins, color="steelblue", alpha=0.7, edgecolor="white",
         label="Absolute error count")

sorted_ae = np.sort(abs_err)
cdf = np.arange(1, len(sorted_ae) + 1) / len(sorted_ae) * 100
ax2.plot(sorted_ae, cdf, color="darkorange", lw=2.5, label="Cumulative %")

for t, color in [(3, "#388e3c"), (7, "#f57c00"), (14, "#d32f2f")]:
    pct = (abs_err <= t).mean() * 100
    ax2.axvline(t, color=color, lw=1.5, ls="--")
    ax2.text(t + 0.3, pct + 2, f"{pct:.0f}%\n≤{t}d", color=color, fontsize=9, va="bottom")

ax1.set_xlabel("Absolute prediction error (days)")
ax1.set_ylabel("Number of patients", color="steelblue")
ax2.set_ylabel("Cumulative % of patients", color="darkorange")
ax2.set_ylim(0, 110)
ax1.set_title("Absolute Error Distribution", fontweight="bold")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "05_absolute_error_distribution.png", dpi=300, bbox_inches="tight")
plt.close(fig)
p50, p75, p90 = np.percentile(abs_err, [50, 75, 90])
print(f"   p50={p50:.1f}d  p75={p75:.1f}d  p90={p90:.1f}d")


# ── 6. RESIDUALS vs ACTUAL (heteroscedasticity) ──────────────────────────────
# Coloured by absolute error; shows if variance grows with LOS.
print("[6/6] Residuals vs Actual LOS...")

fig, ax = plt.subplots(figsize=(9, 5))
sc = ax.scatter(y_true, errors, c=abs_err, cmap="RdYlGn_r",
                vmin=0, vmax=np.percentile(abs_err, 90),
                s=35, alpha=0.75, edgecolors="none")
plt.colorbar(sc, ax=ax, label="Absolute error (days)")

ax.axhline(0,    color="black",  lw=1.5, ls="--", label="Zero error")
ax.axhline(mae,  color="#FF9800", lw=1.2, ls=":",  label=f"+MAE ({mae:.1f}d)")
ax.axhline(-mae, color="#FF9800", lw=1.2, ls=":",  label=f"−MAE ({mae:.1f}d)")

ax.set_xlabel("Actual LOS (days)")
ax.set_ylabel("Prediction error  Predicted − Actual (days)")
ax.set_title("Residuals vs Actual LOS\n(colour = absolute error magnitude)",
             fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "06_residuals_vs_actual.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"   Mean residual={errors.mean():+.2f}d  Std={errors.std():.2f}d")


print("\n" + "=" * 55)
print("ALL PLOTS SAVED TO: diagnostic_plots/")
print("=" * 55)
files = [
    "01_actual_vs_predicted.png       — scatter coloured by error, ±3/7d bands",
    "02_within_n_days_accuracy.png    — bar chart of clinical accuracy thresholds",
    "03_error_by_los_bucket.png       — boxplots per LOS group",
    "04_bland_altman.png              — bias/LOA + trend line",
    "05_absolute_error_distribution.png — histogram + CDF",
    "06_residuals_vs_actual.png       — heteroscedasticity check",
]
for f in files:
    print(f"  {f}")
