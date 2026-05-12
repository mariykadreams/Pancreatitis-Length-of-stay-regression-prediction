"""
Pancreatitis Length-of-Stay Regression – Full ML Pipeline
==========================================================
Steps:
  1. Load & inspect data
  2. Missing-data audit (counts, percentages, heatmap)
  3. Target distribution analysis
  4. Key clinical feature analysis
  5. Correlation analysis
  6. Feature engineering
  7. Model training (XGBoost / RandomForest) with CV
  8. Feature importance
  9. Save pipeline + plots

Run:
    python pancreatitis_ml.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless – saves PNGs without a display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

try:
    import joblib
except ImportError:
    import pickle as joblib

# ── output folder for plots ──────────────────────────────────────────────────
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

TARGET = "Length of stay"

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str = "train.csv") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    # drop pure row-id column when present
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. MISSING DATA AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def missing_data_report(df: pd.DataFrame) -> pd.DataFrame:
    """Print and return a DataFrame with missing-value statistics."""
    total = df.shape[0]
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)
    dtype = df.dtypes

    report = pd.DataFrame({
        "missing_count": missing,
        "missing_pct":   pct,
        "dtype":         dtype,
    }).sort_values("missing_pct", ascending=False)

    report = report[report["missing_count"] > 0]

    print("\n" + "=" * 60)
    print("MISSING DATA REPORT")
    print("=" * 60)
    print(f"Dataset shape : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns with missing values : {len(report)}")
    print()
    print(report.to_string())

    # ── heatmap of missing values ────────────────────────────────────────────
    cols_with_missing = report.index.tolist()
    if cols_with_missing:
        fig, ax = plt.subplots(figsize=(14, max(4, len(cols_with_missing) * 0.35)))
        missing_matrix = df[cols_with_missing].isnull().astype(int)
        sns.heatmap(
            missing_matrix.T,
            cmap="YlOrRd",
            cbar=False,
            ax=ax,
            xticklabels=False,
        )
        ax.set_title("Missing Data Heatmap\n(each column = feature, each row = sample)")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "missing_data_heatmap.png", dpi=120)
        plt.close(fig)
        print(f"\n[plot saved] {PLOTS_DIR / 'missing_data_heatmap.png'}")

    # ── bar chart: top missing columns ──────────────────────────────────────
    top_missing = report.head(20)
    if not top_missing.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(top_missing.index[::-1], top_missing["missing_pct"][::-1], color="salmon")
        ax.set_xlabel("Missing (%)")
        ax.set_title("Top Columns by Missing Data %")
        for i, v in enumerate(top_missing["missing_pct"][::-1]):
            ax.text(v + 0.3, i, f"{v}%", va="center", fontsize=8)
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "missing_data_bar.png", dpi=120)
        plt.close(fig)
        print(f"[plot saved] {PLOTS_DIR / 'missing_data_bar.png'}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# 3. TARGET VARIABLE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def target_analysis(df: pd.DataFrame):
    y = df[TARGET].dropna()
    print("\n" + "=" * 60)
    print("TARGET VARIABLE ANALYSIS — 'Length of stay'")
    print("=" * 60)
    print(y.describe().round(2).to_string())
    print(f"\nSkewness : {y.skew():.3f}")
    print(f"Kurtosis : {y.kurtosis():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # histogram
    axes[0].hist(y, bins=30, color="steelblue", edgecolor="white")
    axes[0].set_title("Length of Stay – Distribution")
    axes[0].set_xlabel("Days")
    axes[0].set_ylabel("Count")

    # box-plot
    axes[1].boxplot(y, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.6))
    axes[1].set_title("Length of Stay – Boxplot")
    axes[1].set_ylabel("Days")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "target_distribution.png", dpi=120)
    plt.close(fig)
    print(f"[plot saved] {PLOTS_DIR / 'target_distribution.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. KEY CLINICAL FEATURE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

# Clinically important feature groups for pancreatitis LOS
IMPORTANT_FEATURES = {
    "Severity Scores":        ["BISAP", "Ransom Adm", "Ransom 48h", "SIRS", "CCI"],
    "Inflammatory Markers":   ["PCR Adm", "PCR 48h", "PCR 72h"],
    "Renal Function":         ["Urea Adm", "Urea 48h", "Urea 72h",
                               "Creat Adm", "Creat 48h", "Creat 72h"],
    "Blood Count":            ["Leukocytes Adm", "PMN Adm", "Lymphocytes Adm"],
    "Demographics / Anthro":  ["Age", "BMI", "Weight"],
    "Liver / Pancreas Enzymes": ["Amilasa", "Lipase", "GOT", "GPT"],
}


def feature_analysis(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("KEY CLINICAL FEATURE ANALYSIS")
    print("=" * 60)

    for group, cols in IMPORTANT_FEATURES.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            continue

        print(f"\n-- {group} --")
        print(df[present].describe().round(2).to_string())

        # scatter vs target
        n = len(present)
        ncols_plot = min(3, n)
        nrows_plot = math.ceil(n / ncols_plot)
        fig, axes = plt.subplots(nrows_plot, ncols_plot,
                                  figsize=(5 * ncols_plot, 4 * nrows_plot))
        axes = np.array(axes).flatten()

        for i, col in enumerate(present):
            subset = df[[col, TARGET]].dropna()
            axes[i].scatter(subset[col], subset[TARGET], alpha=0.4,
                            s=15, color="steelblue")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("LOS (days)")
            axes[i].set_title(f"{col} vs LOS")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"{group} vs Length of Stay", fontsize=12, y=1.01)
        plt.tight_layout()
        safe_name = group.lower().replace(" ", "_").replace("/", "_")
        path = PLOTS_DIR / f"features_{safe_name}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  [plot saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def correlation_analysis(df: pd.DataFrame, top_n: int = 20):
    print("\n" + "=" * 60)
    print("CORRELATION WITH TARGET (top features)")
    print("=" * 60)

    numeric_df = df.select_dtypes(include=[np.number])
    if TARGET not in numeric_df.columns:
        return

    corr = numeric_df.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    print(corr.head(top_n).round(3).to_string())

    top_corr = corr.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["salmon" if v < 0 else "steelblue" for v in top_corr.values]
    ax.barh(top_corr.index[::-1], top_corr.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with LOS")
    ax.set_title(f"Top {top_n} Features Correlated with Length of Stay")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "correlation_with_target.png", dpi=120)
    plt.close(fig)
    print(f"[plot saved] {PLOTS_DIR / 'correlation_with_target.png'}")

    # correlation heatmap for top correlated features only
    top_cols = corr.head(15).index.tolist() + [TARGET]
    sub = df[top_cols].dropna(how="all")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        sub.corr().round(2),
        annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.4, ax=ax, annot_kws={"size": 7},
    )
    ax.set_title("Correlation Heatmap – Top 15 Features + Target")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=120)
    plt.close(fig)
    print(f"[plot saved] {PLOTS_DIR / 'correlation_heatmap.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── temporal deltas (value at later time point minus admission) ──────────
    temporal_groups = [
        ("PCR",        ["PCR Adm",        "PCR 48h",        "PCR 72h"]),
        ("Creat",      ["Creat Adm",       "Creat 48h",      "Creat 72h"]),
        ("Urea",       ["Urea Adm",        "Urea 48h",       "Urea 72h"]),
        ("Leukocytes", ["Leukocytes Adm",  "Leukocytes 48h", "Leukocytes 72h"]),
        ("Hct",        ["Hct Adm",         "Hct 48h",        "Hct 72h"]),
    ]
    for base, cols in temporal_groups:
        existing = [c for c in cols if c in df.columns]
        adm_cols = [c for c in existing if "Adm" in c]
        follow_cols = [c for c in existing if "48h" in c or "72h" in c]
        if adm_cols and follow_cols:
            adm = adm_cols[0]
            for fc in follow_cols:
                tag = fc.split()[-1]
                df[f"{base}_delta_{tag}"] = df[fc] - df[adm]

    # ── clinically-derived ratios ─────────────────────────────────────────────
    if "PCR Adm" in df.columns and "Albumin" in df.columns:
        df["PCR_Albumin_ratio"] = df["PCR Adm"] / (df["Albumin"].replace(0, np.nan))

    if "PMN Adm" in df.columns and "Lymphocytes Adm" in df.columns:
        df["NLR_adm"] = df["PMN Adm"] / (df["Lymphocytes Adm"].replace(0, np.nan))

    if "PMN 48h" in df.columns and "Lymph 48h" in df.columns:
        df["NLR_48h"] = df["PMN 48h"] / (df["Lymph 48h"].replace(0, np.nan))

    # ── severity composite ───────────────────────────────────────────────────
    sev_cols = [c for c in ["BISAP", "Ransom Adm", "SIRS", "CCI"] if c in df.columns]
    if sev_cols:
        df["severity_composite"] = df[sev_cols].fillna(0).sum(axis=1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. PREPROCESSING PIPELINE + MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols     = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    try:
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ])
    except TypeError:
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, numeric_cols),
        ("cat", cat_transformer, categorical_cols),
    ], remainder="drop")

    try:
        from xgboost import XGBRegressor
        estimator = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.75,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.5,
            reg_lambda=1.5,
            random_state=42,
            verbosity=0,
        )
        model_name = "XGBoost"
    except ImportError:
        estimator = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        model_name = "RandomForest"

    print(f"  Estimator : {model_name}")
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    return pipeline


def stratify_bins(y: pd.Series, q: int = 10) -> pd.Series:
    try:
        return pd.qcut(y, q=q, labels=False, duplicates="drop")
    except Exception:
        return pd.cut(y, bins=q, labels=False)


def train_and_evaluate(df: pd.DataFrame) -> dict:
    df = df[df[TARGET].notna()].copy()
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(float)

    bins = stratify_bins(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=bins
    )

    print(f"\n  Samples  : {len(X_train)} train | {len(X_test)} test")
    print(f"  Features : {X.shape[1]}")

    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    try:
        rmse = mean_squared_error(y_test, preds, squared=False)
    except TypeError:
        rmse = math.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    print(f"\n  Test MAE  : {mae:.4f} days")
    print(f"  Test RMSE : {rmse:.4f} days")
    print(f"  Test R²   : {r2:.4f}")

    # 5-fold CV on training set
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train,
                                cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_mae = -cv_scores.mean()
    print(f"  CV MAE (5-fold) : {cv_mae:.4f} ± {cv_scores.std():.4f}")

    # ── residual plot ────────────────────────────────────────────────────────
    residuals = y_test.values - preds
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(preds, residuals, alpha=0.4, s=15, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted LOS (days)")
    axes[0].set_ylabel("Residual (actual − predicted)")
    axes[0].set_title("Residual Plot")

    axes[1].scatter(y_test, preds, alpha=0.4, s=15, color="steelblue")
    mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    axes[1].plot([mn, mx], [mn, mx], "r--", linewidth=1)
    axes[1].set_xlabel("Actual LOS (days)")
    axes[1].set_ylabel("Predicted LOS (days)")
    axes[1].set_title(f"Actual vs Predicted  (R²={r2:.3f})")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "model_evaluation.png", dpi=120)
    plt.close(fig)
    print(f"  [plot saved] {PLOTS_DIR / 'model_evaluation.png'}")

    return {
        "pipeline": pipe,
        "mae": mae, "rmse": rmse, "r2": r2, "cv_mae": cv_mae,
        "X_test": X_test, "y_test": y_test, "preds": preds,
        "feature_names": X.columns.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(results: dict, top_n: int = 20):
    pipe         = results["pipeline"]
    X_test       = results["X_test"]
    y_test       = results["y_test"]
    feat_names   = results["feature_names"]

    print("\n  Computing permutation importance …")
    perm = permutation_importance(
        pipe, X_test, y_test, n_repeats=15, random_state=42, n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )

    imp_df = pd.DataFrame({
        "feature":    feat_names,
        "importance": perm.importances_mean,
        "std":        perm.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\n  Top {top_n} features by permutation importance:")
    print(imp_df.head(top_n).to_string(index=False))

    imp_df.to_csv("feature_importance.csv", index=False)
    print("  [saved] feature_importance.csv")

    top = imp_df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"][::-1], top["importance"][::-1],
            xerr=top["std"][::-1], align="center",
            color="steelblue", alpha=0.85, ecolor="grey", capsize=3)
    ax.set_xlabel("Mean permutation importance (MAE reduction)")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)
    print(f"  [plot saved] {PLOTS_DIR / 'feature_importance.png'}")

    return imp_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    data_path = Path("train.csv")
    if not data_path.exists():
        raise FileNotFoundError("train.csv not found in current directory.")

    # 1. Load ────────────────────────────────────────────────────────────────
    print("\n[1/8] Loading data …")
    df = load_data(str(data_path))
    print(f"      Shape: {df.shape}")

    # 2. Missing data audit ──────────────────────────────────────────────────
    print("\n[2/8] Missing data audit …")
    missing_report = missing_data_report(df)

    # 3. Target analysis ─────────────────────────────────────────────────────
    print("\n[3/8] Target variable analysis …")
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")
    target_analysis(df)

    # 4. Key feature analysis ────────────────────────────────────────────────
    print("\n[4/8] Key clinical feature analysis …")
    feature_analysis(df)

    # 5. Correlation analysis ────────────────────────────────────────────────
    print("\n[5/8] Correlation analysis …")
    correlation_analysis(df, top_n=20)

    # 6. Feature engineering ─────────────────────────────────────────────────
    print("\n[6/8] Feature engineering …")
    df_eng = engineer_features(df)
    new_cols = [c for c in df_eng.columns if c not in df.columns]
    print(f"      {len(new_cols)} new features created: {new_cols}")

    # 7. Train & evaluate ────────────────────────────────────────────────────
    print("\n[7/8] Training model …")
    results = train_and_evaluate(df_eng)

    # 8. Feature importance ──────────────────────────────────────────────────
    print("\n[8/8] Feature importance …")
    imp_df = plot_feature_importance(results, top_n=20)

    # Save pipeline ──────────────────────────────────────────────────────────
    joblib.dump(results["pipeline"], "final_model_pipeline.joblib")
    print("\n  [saved] final_model_pipeline.joblib")

    # Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  MAE  (test)    : {results['mae']:.4f} days")
    print(f"  RMSE (test)    : {results['rmse']:.4f} days")
    print(f"  R²   (test)    : {results['r2']:.4f}")
    print(f"  MAE  (5-fold CV): {results['cv_mae']:.4f} days")
    print(f"\n  Top 5 predictors:")
    for _, row in imp_df.head(5).iterrows():
        print(f"    {row['feature']:35s}  importance={row['importance']:.4f}")
    print("\n  Plots saved to:", PLOTS_DIR.resolve())
    print("=" * 60)


if __name__ == "__main__":
    main()
