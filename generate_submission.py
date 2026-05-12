"""
Generate submission file from trained model predictions.

Loads the final model pipeline and generates predictions for test.csv.
Outputs submission.csv with ID and predicted TARGET (LOS in days).

Usage:
    python generate_submission.py

Outputs:
    - submission.csv (ID, TARGET format)
"""

from pathlib import Path
import pandas as pd
import numpy as np

try:
    import joblib
except ImportError:
    import pickle as joblib


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Must exactly mirror the engineering in pancreatitis_ml.py."""
    df = df.copy()

    temporal_groups = [
        ("PCR",        ["PCR Adm",        "PCR 48h",        "PCR 72h"]),
        ("Creat",      ["Creat Adm",       "Creat 48h",      "Creat 72h"]),
        ("Urea",       ["Urea Adm",        "Urea 48h",       "Urea 72h"]),
        ("Leukocytes", ["Leukocytes Adm",  "Leukocytes 48h", "Leukocytes 72h"]),
        ("Hct",        ["Hct Adm",         "Hct 48h",        "Hct 72h"]),
    ]
    for base, cols in temporal_groups:
        existing   = [c for c in cols if c in df.columns]
        adm_cols   = [c for c in existing if "Adm" in c]
        follow_cols = [c for c in existing if "48h" in c or "72h" in c]
        if adm_cols and follow_cols:
            adm = adm_cols[0]
            for fc in follow_cols:
                tag = fc.split()[-1]
                df[f"{base}_delta_{tag}"] = df[fc] - df[adm]

    if "PCR Adm" in df.columns and "Albumin" in df.columns:
        df["PCR_Albumin_ratio"] = df["PCR Adm"] / df["Albumin"].replace(0, np.nan)

    if "PMN Adm" in df.columns and "Lymphocytes Adm" in df.columns:
        df["NLR_adm"] = df["PMN Adm"] / df["Lymphocytes Adm"].replace(0, np.nan)

    if "PMN 48h" in df.columns and "Lymph 48h" in df.columns:
        df["NLR_48h"] = df["PMN 48h"] / df["Lymph 48h"].replace(0, np.nan)

    sev_cols = [c for c in ["BISAP", "Ransom Adm", "SIRS", "CCI"] if c in df.columns]
    if sev_cols:
        df["severity_composite"] = df[sev_cols].fillna(0).sum(axis=1)

    return df


def main():
    test_path  = Path("test.csv")
    model_path = Path("final_model_pipeline.joblib")

    if not test_path.exists():
        raise FileNotFoundError("test.csv not found.")
    if not model_path.exists():
        raise FileNotFoundError(
            "final_model_pipeline.joblib not found. Run pancreatitis_ml.py first."
        )

    print("Loading test data...")
    # Read without treating any column as an index so we can extract ID cleanly
    test_df = pd.read_csv(test_path)

    # The real submission ID is the 'ID' column
    if "ID" not in test_df.columns:
        raise ValueError("'ID' column not found in test.csv")
    test_ids = test_df["ID"].values

    # Drop both the unnamed row-index column and the ID column before predicting
    drop_cols = ["ID"] + [c for c in test_df.columns if c.startswith("Unnamed")]
    test_df = test_df.drop(columns=drop_cols)

    print(f"Test set shape: {test_df.shape}")

    print("Engineering features...")
    test_df = engineer_features(test_df)

    print("Loading model...")
    pipeline = joblib.load(str(model_path))

    print("Predicting...")
    predictions = np.maximum(pipeline.predict(test_df), 1.0)   # LOS >= 1 day

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
