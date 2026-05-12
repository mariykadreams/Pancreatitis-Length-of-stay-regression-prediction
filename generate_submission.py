"""
Generate submission file from trained model predictions.

Loads the improved model pipeline and generates predictions for test.csv.
Outputs submission.csv with ID and predicted TARGET (LOS in days).

Usage:
    python generate_submission.py

Outputs:
    - submission.csv (ID, TARGET format)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pickle

try:
    import joblib
except Exception:
    class _PickleJoblib:
        @staticmethod
        def dump(obj, path):
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
        @staticmethod
        def load(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
    joblib = _PickleJoblib()


def engineer_features(df):
    """Apply same feature engineering as training."""
    df = df.copy()
    
    temporal_pairs = [
        ('PCR', ['PCR Adm', 'PCR 48h', 'PCR 72h']),
        ('Creat', ['Creat Adm', 'Creat 48h', 'Creat 72h']),
        ('Urea', ['Urea Adm', 'Urea 48h', 'Urea 72h']),
        ('Leukocytes', ['Leukocytes Adm', 'Leukocytes 48h', 'Leukocytes 72h']),
        ('Hct', ['Hct Adm', 'Hct 48h', 'Hct 72h']),
    ]
    
    for base_name, time_cols in temporal_pairs:
        existing_cols = [c for c in time_cols if c in df.columns]
        if len(existing_cols) >= 2:
            adm_col = [c for c in existing_cols if 'Adm' in c]
            adm_72h_cols = [c for c in existing_cols if '72h' in c or '48h' in c]
            
            if adm_col and adm_72h_cols:
                adm_val = adm_col[0]
                for other_col in adm_72h_cols:
                    delta_name = f'{base_name}_delta_{other_col.split()[-1]}'
                    df[delta_name] = df[other_col] - df[adm_val]
    
    if 'Albumin' in df.columns and 'PCR Adm' in df.columns:
        df['PCR_Albumin_adm'] = df['PCR Adm'] / (df['Albumin'] + 1e-3)
    
    if 'PMN Adm' in df.columns and 'Lymphocytes Adm' in df.columns:
        df['PMN_Lymph_adm'] = df['PMN Adm'] / (df['Lymphocytes Adm'] + 1e-3)
    
    if 'Weight' in df.columns and 'Height' in df.columns:
        df['BMI_calc'] = df['Weight'] / ((df['Height'] / 100) ** 2 + 1e-3)
    
    severity_cols = ['BISAP', 'Ransom Adm', 'SIRS']
    existing_severity = [c for c in severity_cols if c in df.columns]
    if existing_severity:
        df['severity_composite'] = df[existing_severity].fillna(0).sum(axis=1)
    
    return df


def main():
    # Check for test.csv
    test_path = Path('test.csv')
    if not test_path.exists():
        print('test.csv not found in current directory.')
        return
    
    # Check for trained model
    model_path = Path('improved_model_pipeline.joblib')
    if not model_path.exists():
        print('improved_model_pipeline.joblib not found.')
        print('Run improved_model.py first to train the model.')
        return
    
    print('Loading test data...')
    test_df = pd.read_csv(test_path, index_col=0)
    
    # Store IDs before engineering
    test_ids = test_df.index.copy()
    
    # Drop ID column if present
    if 'ID' in test_df.columns:
        test_df = test_df.drop(columns=['ID'])
    
    print(f'Test set shape: {test_df.shape}')
    
    print('Engineering features on test data...')
    test_df = engineer_features(test_df)
    
    print('Loading trained model...')
    pipeline = joblib.load(str(model_path))
    
    print('Generating predictions...')
    predictions = pipeline.predict(test_df)
    
    # Ensure predictions are non-negative (LOS cannot be negative)
    predictions = np.maximum(predictions, 0)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'ID': test_ids,
        'TARGET': predictions
    })
    
    # Save to CSV
    submission_path = Path('submission.csv')
    submission.to_csv(submission_path, index=False)
    
    print(f'\nSubmission saved: {submission_path.resolve()}')
    print(f'Shape: {submission.shape}')
    print(f'\nFirst 10 rows:')
    print(submission.head(10))
    print(f'\nPrediction statistics:')
    print(f'  Min: {predictions.min():.2f} days')
    print(f'  Max: {predictions.max():.2f} days')
    print(f'  Mean: {predictions.mean():.2f} days')
    print(f'  Median: {np.median(predictions):.2f} days')
    print(f'  Std: {predictions.std():.2f} days')


if __name__ == '__main__':
    main()
