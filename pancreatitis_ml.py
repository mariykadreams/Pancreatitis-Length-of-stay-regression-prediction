"""
Pancreatitis Length-of-Stay Regression – Full ML Pipeline
==========================================================
Improvements over baseline:
  - log1p target transform  (skewness = 5.78 -> handles heavy right tail)
  - LightGBM estimator       (usually outperforms XGBoost on small tabular data)
  - Bland-Altman plot        (reveals systematic bias & heteroscedasticity)
  - Within-N-days accuracy   (clinically meaningful thresholds)
  - Permutation feature importance

Run:
    python pancreatitis_ml.py

Outputs:
    - final_model_pipeline.joblib
    - feature_importance.csv
    - plots/
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

try:
    import joblib
except ImportError:
    import pickle as joblib

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)
TARGET = "Length of stay"


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str = "train.csv") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MISSING DATA AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def missing_data_report(df: pd.DataFrame) -> pd.DataFrame:
    total   = df.shape[0]
    missing = df.isnull().sum()
    pct     = (missing / total * 100).round(2)
    report  = pd.DataFrame({"missing_count": missing, "missing_pct": pct,
                             "dtype": df.dtypes})
    report  = report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)

    print("\n" + "=" * 60)
    print("MISSING DATA REPORT")
    print("=" * 60)
    print(f"Shape : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns with missing values : {len(report)}")
    print(report.to_string())

    # heatmap
    cols = report.index.tolist()
    if cols:
        fig, ax = plt.subplots(figsize=(14, max(4, len(cols) * 0.35)))
        sns.heatmap(df[cols].isnull().astype(int).T, cmap="YlOrRd",
                    cbar=False, ax=ax, xticklabels=False)
        ax.set_title("Missing Data Heatmap")
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "missing_data_heatmap.png", dpi=120)
        plt.close(fig)

    # bar chart
    top = report.head(20)
    if not top.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(top.index[::-1], top["missing_pct"][::-1], color="salmon")
        ax.set_xlabel("Missing (%)")
        ax.set_title("Top Columns by Missing Data %")
        for i, v in enumerate(top["missing_pct"][::-1]):
            ax.text(v + 0.3, i, f"{v}%", va="center", fontsize=8)
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "missing_data_bar.png", dpi=120)
        plt.close(fig)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# TARGET ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def target_analysis(df: pd.DataFrame):
    y = df[TARGET].dropna()
    print("\n" + "=" * 60)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 60)
    print(y.describe().round(2).to_string())
    print(f"Skewness : {y.skew():.3f}  |  Kurtosis : {y.kurtosis():.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(y, bins=30, color="steelblue", edgecolor="white")
    axes[0].set_title("LOS - original")
    axes[0].set_xlabel("Days")

    log_y = np.log1p(y)
    axes[1].hist(log_y, bins=30, color="seagreen", edgecolor="white")
    axes[1].set_title("LOS - log1p transformed")
    axes[1].set_xlabel("log(1 + days)")

    axes[2].boxplot(y, patch_artist=True, boxprops=dict(facecolor="steelblue", alpha=0.6))
    axes[2].set_title("LOS - boxplot")
    axes[2].set_ylabel("Days")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "target_distribution.png", dpi=120)
    plt.close(fig)
    print(f"[plot] {PLOTS_DIR / 'target_distribution.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    temporal_groups = [
        ("PCR",        ["PCR Adm",        "PCR 48h",        "PCR 72h"]),
        ("Creat",      ["Creat Adm",       "Creat 48h",      "Creat 72h"]),
        ("Urea",       ["Urea Adm",        "Urea 48h",       "Urea 72h"]),
        ("Leukocytes", ["Leukocytes Adm",  "Leukocytes 48h", "Leukocytes 72h"]),
        ("Hct",        ["Hct Adm",         "Hct 48h",        "Hct 72h"]),
    ]
    for base, cols in temporal_groups:
        existing    = [c for c in cols if c in df.columns]
        adm_cols    = [c for c in existing if "Adm" in c]
        follow_cols = [c for c in existing if "48h" in c or "72h" in c]
        if adm_cols and follow_cols:
            adm = adm_cols[0]
            for fc in follow_cols:
                df[f"{base}_delta_{fc.split()[-1]}"] = df[fc] - df[adm]

    if "PCR Adm" in df.columns and "Albumin" in df.columns:
        df["PCR_Albumin_ratio"] = df["PCR Adm"] / df["Albumin"].replace(0, np.nan)

    if "PMN Adm" in df.columns and "Lymphocytes Adm" in df.columns:
        df["NLR_adm"] = df["PMN Adm"] / df["Lymphocytes Adm"].replace(0, np.nan)

    if "PMN 48h" in df.columns and "Lymph 48h" in df.columns:
        df["NLR_48h"] = df["PMN 48h"] / df["Lymph 48h"].replace(0, np.nan)

    sev = [c for c in ["BISAP", "Ransom Adm", "SIRS", "CCI"] if c in df.columns]
    if sev:
        df["severity_composite"] = df[sev].fillna(0).sum(axis=1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

def _best_estimator():
    """Return best available gradient boosting estimator."""
    try:
        import lightgbm as lgb
        est = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        return est, "LightGBM"
    except ImportError:
        pass

    try:
        from xgboost import XGBRegressor
        est = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.75,
            min_child_weight=3,
            gamma=0.05,
            reg_alpha=0.3,
            reg_lambda=1.5,
            random_state=42,
            verbosity=0,
        )
        return est, "XGBoost"
    except ImportError:
        pass

    return RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                 random_state=42, n_jobs=-1), "RandomForest"


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_tf = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("sc",  StandardScaler())])
    try:
        cat_tf = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="__missing__")),
                           ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])
    except TypeError:
        cat_tf = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="__missing__")),
                           ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    pre = ColumnTransformer([("num", num_tf, num_cols),
                              ("cat", cat_tf, cat_cols)], remainder="drop")

    est, name = _best_estimator()
    print(f"  Estimator : {name}")
    return Pipeline([("pre", pre), ("model", est)])


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _rmse(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return math.sqrt(mean_squared_error(y_true, y_pred))


def within_n_days(y_true, y_pred, n):
    return (np.abs(np.array(y_true) - np.array(y_pred)) <= n).mean() * 100


def bland_altman_plot(y_true, y_pred, title: str, path: Path):
    mean_vals = (y_true + y_pred) / 2
    diff      = y_pred - y_true
    bias      = diff.mean()
    sd        = diff.std()
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mean_vals, diff, alpha=0.45, s=18, color="steelblue")
    ax.axhline(bias,      linestyle="--", linewidth=2,
               label=f"Bias = {bias:.2f} days")
    ax.axhline(loa_upper, linestyle="--", color="red", linewidth=1.5,
               label=f"+1.96 SD = {loa_upper:.2f}")
    ax.axhline(loa_lower, linestyle="--", color="red", linewidth=1.5,
               label=f"-1.96 SD = {loa_lower:.2f}")
    ax.set_xlabel("Mean LOS (days)")
    ax.set_ylabel("Prediction error (days)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [plot] {path}")
    return bias, sd, loa_lower, loa_upper


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def stratify_bins(y: pd.Series, q: int = 10) -> pd.Series:
    try:
        return pd.qcut(y, q=q, labels=False, duplicates="drop")
    except Exception:
        return pd.cut(y, bins=q, labels=False)


def train_and_evaluate(df: pd.DataFrame) -> dict:
    df = df[df[TARGET].notna()].copy()
    X  = df.drop(columns=[TARGET])
    y  = df[TARGET].astype(float)

    bins = stratify_bins(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=bins)

    # ── log1p transform: target skewness is 5.78, strongly right-tailed ──────
    y_train_log = np.log1p(y_train)

    print(f"\n  Samples  : {len(X_train)} train | {len(X_test)} test")
    print(f"  Features : {X.shape[1]}")

    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train_log)

    # Back-transform predictions to original scale
    preds_log = pipe.predict(X_test)
    preds     = np.expm1(preds_log)
    preds     = np.maximum(preds, 1.0)

    mae  = mean_absolute_error(y_test, preds)
    rmse = _rmse(y_test, preds)
    r2   = r2_score(y_test, preds)

    print(f"\n  Test MAE  : {mae:.4f} days")
    print(f"  Test RMSE : {rmse:.4f} days")
    print(f"  Test R2   : {r2:.4f}")
    for n in [2, 3, 5]:
        print(f"  Within {n} days : {within_n_days(y_test, preds, n):.1f}%")

    # 5-fold CV on log-transformed target
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_raw = cross_val_score(pipe, X_train, y_train_log,
                             cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1)
    # convert CV MAE from log-space back to approximate original-scale MAE
    cv_mae_log = -cv_raw.mean()
    print(f"  CV MAE log-space (5-fold) : {cv_mae_log:.4f} +/- {cv_raw.std():.4f}")

    # ── plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    res = y_test.values - preds
    axes[0].scatter(preds, res, alpha=0.4, s=15, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted LOS")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residual Plot")

    mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    axes[1].scatter(y_test, preds, alpha=0.4, s=15, color="steelblue")
    axes[1].plot([mn, mx], [mn, mx], "r--")
    axes[1].set_xlabel("Actual LOS")
    axes[1].set_ylabel("Predicted LOS")
    axes[1].set_title(f"Actual vs Predicted  (R2={r2:.3f})")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "model_evaluation.png", dpi=120)
    plt.close(fig)
    print(f"  [plot] {PLOTS_DIR / 'model_evaluation.png'}")

    ba = bland_altman_plot(
        y_test.values, preds,
        title="Bland-Altman Plot — LOS Prediction",
        path=PLOTS_DIR / "bland_altman.png",
    )
    print(f"  Bland-Altman  bias={ba[0]:.2f}  SD={ba[1]:.2f}  "
          f"LOA=[{ba[2]:.2f}, {ba[3]:.2f}]")

    return {
        "pipeline": pipe,
        "mae": mae, "rmse": rmse, "r2": r2,
        "X_test": X_test, "y_test": y_test, "preds": preds,
        "feature_names": X.columns.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(results: dict, top_n: int = 20) -> pd.DataFrame:
    pipe       = results["pipeline"]
    X_test     = results["X_test"]
    y_test     = results["y_test"]
    feat_names = results["feature_names"]

    print("\n  Computing permutation importance ...")
    perm = permutation_importance(
        pipe, X_test, np.log1p(y_test),   # importance on log-scale target
        n_repeats=15, random_state=42, n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )

    imp_df = pd.DataFrame({
        "feature":    feat_names,
        "importance": perm.importances_mean,
        "std":        perm.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\n  Top {top_n} features:")
    print(imp_df.head(top_n).to_string(index=False))

    imp_df.to_csv("feature_importance.csv", index=False)

    top = imp_df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"][::-1], top["importance"][::-1],
            xerr=top["std"][::-1], color="steelblue", alpha=0.85,
            ecolor="grey", capsize=3)
    ax.set_xlabel("Mean permutation importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)
    print(f"  [plot] {PLOTS_DIR / 'feature_importance.png'}")

    return imp_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    data_path = Path("train.csv")
    if not data_path.exists():
        raise FileNotFoundError("train.csv not found.")

    print("\n[1/6] Loading data ...")
    df = load_data(str(data_path))
    print(f"      Shape: {df.shape}")

    print("\n[2/6] Missing data audit ...")
    missing_data_report(df)

    print("\n[3/6] Target analysis ...")
    target_analysis(df)

    print("\n[4/6] Feature engineering ...")
    df = engineer_features(df)
    print(f"      Total features after engineering: {df.shape[1] - 1}")

    print("\n[5/6] Training model (log1p target) ...")
    results = train_and_evaluate(df)

    print("\n[6/6] Feature importance ...")
    imp_df = plot_feature_importance(results)

    joblib.dump(results["pipeline"], "final_model_pipeline.joblib")
    print("\n  [saved] final_model_pipeline.joblib")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  MAE  : {results['mae']:.4f} days")
    print(f"  RMSE : {results['rmse']:.4f} days")
    print(f"  R2   : {results['r2']:.4f}")
    print(f"\n  Top 5 predictors:")
    for _, row in imp_df.head(5).iterrows():
        print(f"    {row['feature']:35s}  {row['importance']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
