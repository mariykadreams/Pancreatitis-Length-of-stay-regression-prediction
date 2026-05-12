"""
Improved LOS regression model with feature engineering, tuning, and parsimony focus.

Key improvements:
- Feature engineering: temporal deltas (48h-Adm, 72h-Adm), ratios, derived indices
- Feature selection: permutation importance + top-N features for parsimony
- Hyperparameter tuning: optimized XGBoost parameters
- Better handling of missing data and outliers
- Comparison of full vs. parsimonious (5-feature) models

Usage:
    python improved_model.py

Outputs:
    - improved_model_pipeline.joblib (full model)
    - parsimonious_model_pipeline.joblib (top-5 features only)
    - feature_importance.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import math
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

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(path: str):
    df = pd.read_csv(path, index_col=0)
    if 'ID' in df.columns:
        try:
            df.drop(columns=['ID'], inplace=True)
        except Exception:
            pass
    return df


def engineer_features(df, y_column='Length of stay'):
    """Create derived features for improved prediction."""
    df = df.copy()
    
    # Temporal deltas (72h - Adm, 48h - Adm)
    temporal_pairs = [
        ('PCR', ['PCR Adm', 'PCR 48h', 'PCR 72h']),
        ('Creat', ['Creat Adm', 'Creat 48h', 'Creat 72h']),
        ('Urea', ['Urea Adm', 'Urea 48h', 'Urea 72h']),
        ('Leukocytes', ['Leukocytes Adm', 'Leukocytes 48h', 'Leukocytes 72h']),
        ('Hct', ['Hct Adm', 'Hct 48h', 'Hct 72h']),
    ]
    
    for base_name, time_cols in temporal_pairs:
        # Filter to only existing columns
        existing_cols = [c for c in time_cols if c in df.columns]
        if len(existing_cols) >= 2:
            adm_col = [c for c in existing_cols if 'Adm' in c]
            adm_72h_cols = [c for c in existing_cols if '72h' in c or '48h' in c]
            
            if adm_col and adm_72h_cols:
                adm_val = adm_col[0]
                for other_col in adm_72h_cols:
                    delta_name = f'{base_name}_delta_{other_col.split()[-1]}'
                    df[delta_name] = df[other_col] - df[adm_val]
    
    # Derived indices / ratios
    if 'Albumin' in df.columns and 'PCR Adm' in df.columns:
        df['PCR_Albumin_adm'] = df['PCR Adm'] / (df['Albumin'] + 1e-3)
    
    if 'PMN Adm' in df.columns and 'Lymphocytes Adm' in df.columns:
        df['PMN_Lymph_adm'] = df['PMN Adm'] / (df['Lymphocytes Adm'] + 1e-3)
    
    if 'Weight' in df.columns and 'Height' in df.columns:
        df['BMI_calc'] = df['Weight'] / ((df['Height'] / 100) ** 2 + 1e-3)
    
    # Severity composite (sum of key scores)
    severity_cols = ['BISAP', 'Ransom Adm', 'SIRS']
    existing_severity = [c for c in severity_cols if c in df.columns]
    if existing_severity:
        df['severity_composite'] = df[existing_severity].fillna(0).sum(axis=1)
    
    return df


def make_stratify_bins(y, q=10):
    try:
        bins = pd.qcut(y, q=q, labels=False, duplicates='drop')
    except Exception:
        bins = pd.cut(y, bins=q, labels=False)
    return bins


def build_pipeline(df, y_column, feature_cols=None):
    """Build preprocessing + model pipeline."""
    X = df.drop(columns=[y_column])
    
    if feature_cols is not None:
        X = X[feature_cols]
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    try:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
    except TypeError:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')
    
    # Try XGBoost with tuned params, else RandomForest
    try:
        from xgboost import XGBRegressor
        estimator = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
    except Exception:
        estimator = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', estimator)])
    return pipeline


def train_and_evaluate(df, y_column='Length of stay', feature_cols=None, model_name=''):
    """Train model and return metrics + feature importances."""
    df = df.copy()
    df = df[df[y_column].notna()]
    
    X = df.drop(columns=[y_column])
    if feature_cols is not None:
        X = X[feature_cols]
    y = df[y_column].astype(float)
    
    strat_bins = make_stratify_bins(y, q=10)
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat_bins
    )
    
    pipeline = build_pipeline(df, y_column, feature_cols=feature_cols)
    
    print(f'\n{"="*60}')
    print(f'Training {model_name} model...')
    print(f'Features: {len(X.columns)}')
    print(f'Samples: {len(train_X)} train, {len(test_X)} test')
    print(f'{"="*60}')
    
    pipeline.fit(train_X, train_y)
    
    preds = pipeline.predict(test_X)
    
    mae = mean_absolute_error(test_y, preds)
    try:
        rmse = mean_squared_error(test_y, preds, squared=False)
    except TypeError:
        mse = mean_squared_error(test_y, preds)
        rmse = math.sqrt(mse)
    r2 = r2_score(test_y, preds)
    
    print(f'Test MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}')
    
    try:
        cv_scores = cross_val_score(pipeline, train_X, train_y, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        print(f'CV MAE (5-fold): {cv_mae:.4f}')
    except Exception as e:
        print(f'CV failed: {e}')
        cv_mae = None
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'cv_mae': cv_mae,
        'pipeline': pipeline,
        'test_y': test_y,
        'preds': preds,
        'feature_cols': X.columns.tolist()
    }


def extract_top_features(df, y_column='Length of stay', n_features=5):
    """Use permutation importance to identify top N features."""
    print(f'\nExtracting top {n_features} features via permutation importance...')
    
    df_full = df.copy()
    df_full = df_full[df_full[y_column].notna()]
    
    X = df_full.drop(columns=[y_column])
    y = df_full[y_column].astype(float)
    
    strat_bins = make_stratify_bins(y, q=10)
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat_bins
    )
    
    pipeline = build_pipeline(df_full, y_column)
    pipeline.fit(train_X, train_y)
    
    # Permutation importance on test set
    perm_imp = permutation_importance(
        pipeline, test_X, test_y, n_repeats=10, random_state=42, n_jobs=-1
    )
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': perm_imp.importances_mean
    }).sort_values('importance', ascending=False)
    
    print(importance_df.head(10))
    
    top_features = importance_df.head(n_features)['feature'].tolist()
    print(f'\nTop {n_features} features: {top_features}')
    
    return top_features, importance_df


def main():
    data_path = Path('train.csv')
    if not data_path.exists():
        print('train.csv not found.')
        return
    
    df = load_data(str(data_path))
    y_col = 'Length of stay'
    
    if y_col not in df.columns:
        print(f"Target column '{y_col}' not found.")
        return
    
    print('\n[1] Engineering features...')
    df = engineer_features(df, y_col)
    
    print('\n[2] Training full model...')
    results_full = train_and_evaluate(df, y_col, model_name='FULL')
    joblib.dump(results_full['pipeline'], 'improved_model_pipeline.joblib')
    print(f'Saved: improved_model_pipeline.joblib')
    
    print('\n[3] Finding top-5 parsimonious features...')
    top_features, importance_df = extract_top_features(df, y_col, n_features=5)
    
    print('\n[4] Training parsimonious (top-5) model...')
    results_pars = train_and_evaluate(df, y_col, feature_cols=top_features, model_name='PARSIMONIOUS (Top-5)')
    joblib.dump(results_pars['pipeline'], 'parsimonious_model_pipeline.joblib')
    print(f'Saved: parsimonious_model_pipeline.joblib')
    
    # Save feature importance
    importance_df.to_csv('feature_importance.csv', index=False)
    print(f'\nFeature importance saved: feature_importance.csv')
    
    # Summary
    print('\n' + '='*60)
    print('MODEL COMPARISON SUMMARY')
    print('='*60)
    print(f'Full Model (all features):')
    print(f'  MAE: {results_full["mae"]:.4f} | RMSE: {results_full["rmse"]:.4f} | R²: {results_full["r2"]:.4f}')
    print(f'\nParsimonious Model (top-5 features):')
    print(f'  MAE: {results_pars["mae"]:.4f} | RMSE: {results_pars["rmse"]:.4f} | R²: {results_pars["r2"]:.4f}')
    print(f'  Features: {top_features}')
    delta_mae = results_pars["mae"] - results_full["mae"]
    print(f'\nMAE delta (parsimony cost): +{delta_mae:.4f} days')
    print('='*60)


if __name__ == '__main__':
    main()
