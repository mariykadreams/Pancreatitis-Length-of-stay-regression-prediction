"""
Pancreatitis Length-of-Stay Regression – Full ML Pipeline
==========================================================
Improvements over baseline:
  - log1p target transform      (skewness = 5.78 -> handles heavy right tail)
  - Missingness indicator flags  (>30% missing columns signal clinical severity)
  - Optuna with RepeatedKFold    (5x3=15 evals/trial — much harder to overfit CV)
  - Stacking ensemble            (XGBoost + LightGBM + CatBoost -> Ridge)
  - Bland-Altman plot            (reveals systematic bias & heteroscedasticity)
  - Within-N-days accuracy       (clinically meaningful thresholds)
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
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

try:
    import joblib
except ImportError:
    import pickle as joblib

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)
TARGET = "Length_of_stay"  # sanitized form of "Length of stay"

# Columns with >30% missing — whether the value was observed at all is
# clinically meaningful (mild cases are often discharged before 72h labs).
# Listed using ORIGINAL names (before sanitization) so engineer_features
# can create the _observed flags before renaming.
HIGH_MISSING_COLS = [
    "PCR_72h", "Creat_72h", "Hct_72h", "PMN_72h", "Lymph_72h",
    "Leukocytes_72h", "Urea_72h", "Eosinophils_72h", "Mono_72h",
    "PMN_Lymph_72h", "Col_total", "Amilasa", "PCO2",
    "Exceso_Bases", "CO3_H", "Gasometr_a_Ph",
]


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace special characters in column names so LightGBM accepts them.

    LightGBM rejects names containing JSON special chars (spaces, /, :, accents).
    Replaces any non-alphanumeric/underscore character with '_', then collapses
    consecutive underscores and strips leading/trailing ones.
    """
    new_names = {}
    for col in df.columns:
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', col)
        safe = re.sub(r'_+', '_', safe).strip('_')
        new_names[col] = safe
    return df.rename(columns=new_names)


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

    cols = report.index.tolist()
    if cols:
        fig, ax = plt.subplots(figsize=(14, max(4, len(cols) * 0.35)))
        sns.heatmap(df[cols].isnull().astype(int).T, cmap="YlOrRd",
                    cbar=False, ax=ax, xticklabels=False)
        ax.set_title("Missing Data Heatmap")
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "missing_data_heatmap.png", dpi=120)
        plt.close(fig)

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

    # Temporal deltas (admission → 48h / 72h)
    # Column names are already sanitized (spaces→_, /→_) when this runs
    temporal_groups = [
        ("PCR",        ["PCR_Adm",        "PCR_48h",        "PCR_72h"]),
        ("Creat",      ["Creat_Adm",       "Creat_48h",      "Creat_72h"]),
        ("Urea",       ["Urea_Adm",        "Urea_48h",       "Urea_72h"]),
        ("Leukocytes", ["Leukocytes_Adm",  "Leukocytes_48h", "Leukocytes_72h"]),
        ("Hct",        ["Hct_Adm",         "Hct_48h",        "Hct_72h"]),
    ]
    for base, cols in temporal_groups:
        existing    = [c for c in cols if c in df.columns]
        adm_cols    = [c for c in existing if "Adm" in c]
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

    # Missingness indicator flags — 1 if value was observed, 0 if missing
    for col in HIGH_MISSING_COLS:
        if col in df.columns:
            df[f"{col}_observed"] = df[col].notna().astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────

def tune_xgboost_params(X_train: pd.DataFrame, y_train: pd.Series,
                        n_trials: int = 100) -> dict:
    """Tune XGBoost with Optuna using RepeatedKFold (5 splits × 3 repeats = 15 evals/trial).

    RepeatedKFold prevents Optuna from exploiting lucky single-split variance
    over many trials — critical for small datasets like this one (~500 samples).
    reg_lambda lower bound is 0.1 (not 1e-3) to prevent under-regularized models.
    """
    if optuna is None:
        print("  [warning] Optuna not installed; using default XGBoost params")
        return {}

    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("  [warning] XGBoost not available; skipping tuning")
        return {}

    print(f"\n  Tuning XGBoost params ({n_trials} trials, 5x3 RepeatedKFold) ...")

    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        }
        model = XGBRegressor(**params, random_state=42, verbosity=0)
        scores = cross_val_score(
            model, X_train, y_train, cv=rkf,
            scoring="neg_mean_absolute_error", n_jobs=-1
        )
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"    Best CV MAE (log-space) : {-study.best_value:.4f}")
    print(f"    Best params : {study.best_params}")

    return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# STACKING ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

def build_stacking_pipeline(xgb_params: dict = None) -> Pipeline:
    """Build XGBoost + LightGBM + CatBoost stacking ensemble with Ridge meta-learner.

    All three handle NaN natively so no imputation preprocessing is needed.
    Falls back to a single estimator if libraries are missing.
    """
    base_estimators = []

    # XGBoost — use Optuna-tuned params when provided
    try:
        from xgboost import XGBRegressor
        defaults = {
            "n_estimators": 500, "max_depth": 5, "learning_rate": 0.03,
            "subsample": 0.8, "colsample_bytree": 0.75, "min_child_weight": 3,
            "gamma": 0.05, "reg_alpha": 0.3, "reg_lambda": 1.5,
        }
        if xgb_params:
            defaults.update(xgb_params)
        base_estimators.append(("xgb", XGBRegressor(**defaults, random_state=42, verbosity=0)))
    except ImportError:
        pass

    # LightGBM
    try:
        import lightgbm as lgb
        base_estimators.append(("lgbm", lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.03, num_leaves=31,
            min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1,
        )))
    except ImportError:
        pass

    # CatBoost
    try:
        from catboost import CatBoostRegressor
        base_estimators.append(("cat", CatBoostRegressor(
            iterations=800, depth=6, learning_rate=0.03, loss_function="MAE",
            random_seed=42, verbose=False, allow_writing_files=False,
        )))
    except ImportError:
        pass

    if len(base_estimators) >= 2:
        names = " + ".join(n for n, _ in base_estimators)
        print(f"  Stacking : {names} → Ridge")
        stacker = StackingRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=5,
            n_jobs=1,  # avoid nested parallelism with base estimator n_jobs
        )
        return Pipeline([("model", stacker)])

    # Fallback: single estimator
    print("  [warning] < 2 estimators available; falling back to single model")
    if base_estimators:
        name, est = base_estimators[0]
    else:
        name, est = "RandomForest", RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, random_state=42, n_jobs=-1)
    print(f"  Estimator : {name}")
    return Pipeline([("model", est)])


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

    y_train_log = np.log1p(y_train)

    print(f"\n  Samples  : {len(X_train)} train | {len(X_test)} test")
    print(f"  Features : {X.shape[1]}")

    xgb_params = tune_xgboost_params(X_train, y_train_log, n_trials=100)

    pipe = build_stacking_pipeline(xgb_params=xgb_params)
    pipe.fit(X_train, y_train_log)

    preds_log = pipe.predict(X_test)
    preds     = np.maximum(np.expm1(preds_log), 1.0)

    mae  = mean_absolute_error(y_test, preds)
    rmse = _rmse(y_test, preds)
    r2   = r2_score(y_test, preds)

    print(f"\n  Test MAE  : {mae:.4f} days")
    print(f"  Test RMSE : {rmse:.4f} days")
    print(f"  Test R2   : {r2:.4f}")
    for n in [2, 3, 5]:
        print(f"  Within {n} days : {within_n_days(y_test, preds, n):.1f}%")

    # ── plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    res = y_test.values - preds
    axes[0].scatter(preds, res, alpha=0.4, s=15, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted LOS")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residual Plot")

    mn = min(y_test.min(), preds.min())
    mx = max(y_test.max(), preds.max())
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
        title="Bland-Altman Plot — LOS Prediction (Stacking Ensemble)",
        path=PLOTS_DIR / "bland_altman.png",
    )
    print(f"  Bland-Altman  bias={ba[0]:.2f}  SD={ba[1]:.2f}  "
          f"LOA=[{ba[2]:.2f}, {ba[3]:.2f}]")

    return {
        "pipeline": pipe,
        "xgb_params": xgb_params,
        "mae": mae, "rmse": rmse, "r2": r2,
        "X_test": X_test, "y_test": y_test, "preds": preds,
        "feature_names": X.columns.tolist(),
    }


def train_final_model(df: pd.DataFrame, xgb_params: dict = None) -> Pipeline:
    df = df[df[TARGET].notna()].copy()
    X     = df.drop(columns=[TARGET])
    y_log = np.log1p(df[TARGET].astype(float))

    print(f"\n  Refitting final stacking ensemble on all {len(X)} labeled rows ...")
    final_pipeline = build_stacking_pipeline(xgb_params=xgb_params)
    final_pipeline.fit(X, y_log)
    return final_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(results: dict, top_n: int = 20) -> pd.DataFrame:
    pipe       = results["pipeline"]
    X_test     = results["X_test"]
    y_test     = results["y_test"]
    feat_names = results["feature_names"]

    print("\n  Computing permutation importance (stacking — may take a few minutes) ...")
    perm = permutation_importance(
        pipe, X_test, np.log1p(y_test),
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
    ax.set_title(f"Top {top_n} Feature Importances — Stacking Ensemble")
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
    df = sanitize_columns(df)  # sanitize early so all steps see clean names
    print(f"      Shape: {df.shape}")

    print("\n[2/6] Missing data audit ...")
    missing_data_report(df)

    print("\n[3/6] Target analysis ...")
    target_analysis(df)

    print("\n[4/6] Feature engineering ...")
    df = engineer_features(df)
    print(f"      Total features after engineering: {df.shape[1] - 1}")

    print("\n[5/6] Training stacking ensemble (log1p target) ...")
    results = train_and_evaluate(df)

    print("\n[6/6] Feature importance ...")
    imp_df = plot_feature_importance(results)

    # Refit final model on all data using the tuned XGBoost params
    final_pipeline = train_final_model(df, xgb_params=results["xgb_params"])
    joblib.dump(final_pipeline, "final_model_pipeline.joblib")
    print("\n  [saved] final_model_pipeline.joblib (trained on 100% of train.csv)")

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
