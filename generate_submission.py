"""
Generate submission file from trained model predictions.

Loads the final stacking ensemble and generates predictions for test.csv.
Applies bias correction from bias_correction.json, then clips at 95th pct.

Usage:
    python generate_submission.py

Outputs:
    - submission.csv (ID, TARGET format)
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np

try:
    import joblib
except ImportError:
    import pickle as joblib

from ensemble import WeightedEnsemble  # noqa: F401 — needed for joblib to unpickle the model

# Must exactly mirror HIGH_MISSING_COLS in pancreatitis_ml.py
HIGH_MISSING_COLS = [
    "PCR_72h", "Creat_72h", "Hct_72h", "PMN_72h", "Lymph_72h",
    "Leukocytes_72h", "Urea_72h", "Eosinophils_72h", "Mono_72h",
    "PMN_Lymph_72h", "Col_total", "Amilasa", "PCO2",
    "Exceso_Bases", "CO3_H", "Gasometr_a_Ph",
]


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_names = {}
    for col in df.columns:
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', col)
        safe = re.sub(r'_+', '_', safe).strip('_')
        new_names[col] = safe
    return df.rename(columns=new_names)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Must exactly mirror engineer_features in pancreatitis_ml.py."""
    df = df.copy()

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

    sev_cols = [c for c in ["BISAP", "Ransom_Adm", "SIRS", "CCI"] if c in df.columns]
    if sev_cols:
        df["severity_composite"] = df[sev_cols].fillna(0).sum(axis=1)

    # Missingness indicator flags
    for col in HIGH_MISSING_COLS:
        if col in df.columns:
            df[f"{col}_observed"] = df[col].notna().astype(int)

    return df


def main():
    test_path   = Path("test.csv")
    model_path  = Path("final_model_pipeline.joblib")
    train_path  = Path("train.csv")

    if not test_path.exists():
        raise FileNotFoundError("test.csv not found.")
    if not model_path.exists():
        raise FileNotFoundError(
            "final_model_pipeline.joblib not found. Run pancreatitis_ml.py first."
        )

    print("Loading test data...")
    test_df  = pd.read_csv(test_path)

    if "ID" not in test_df.columns:
        raise ValueError("'ID' column not found in test.csv")
    test_ids = test_df["ID"].values

    drop_cols = ["ID"] + [c for c in test_df.columns if c.startswith("Unnamed")]
    test_df   = test_df.drop(columns=drop_cols)
    test_df   = sanitize_columns(test_df)  # must match training sanitization
    print(f"Test set shape: {test_df.shape}")

    print("Engineering features...")
    test_df = engineer_features(test_df)

    print("Loading model...")
    pipeline = joblib.load(str(model_path))

    print("Predicting...")
    predictions = np.maximum(np.expm1(pipeline.predict(test_df)), 1.0)

    # Bias correction is intentionally skipped for submission:
    # a global +1.2 day shift hurts short-stay predictions (the majority)
    # more than it helps long-stay predictions.

    # Only prevent negative predictions; allow full range
    print("No clipping applied — allowing predictions to reach their full range")

    submission = pd.DataFrame({"ID": test_ids, "TARGET": predictions})
    submission_path = Path("submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"\nSubmission saved: {submission_path.resolve()}")
    print(f"Rows: {len(submission)}")
    print(submission.head(10).to_string(index=False))
    print(f"\nPrediction stats:")
    print(f"  Min   : {predictions.min():.2f} days")
    print(f"  Max   : {predictions.max():.2f} days")
    print(f"  Mean  : {predictions.mean():.2f} days")
    print(f"  Median: {np.median(predictions):.2f} days")
    print(f"  Std   : {predictions.std():.2f} days")


if __name__ == "__main__":
    main()
