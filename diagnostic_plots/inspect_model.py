from pathlib import Path
import pandas as pd
import joblib
import sys
import numpy as np

ROOT = Path(__file__).parent.parent
train_path = ROOT / 'train.csv'
final_model_path = ROOT / 'final_model_pipeline.joblib'
eval_model_path = ROOT / 'eval_model_pipeline.joblib'

print('Reading dataset:', train_path)
df = pd.read_csv(train_path, index_col=0)
print('Rows, cols:', df.shape)
print('Columns:')
for i, c in enumerate(df.columns[:200], 1):
    print(f'  {i:03d}: {c}')

print('\nSample rows:')
print(df.head().to_string())

model_path = eval_model_path if eval_model_path.exists() else final_model_path
print('\nLoading model from:', model_path)
model = joblib.load(model_path)
print('Model type:', type(model))

def try_print_feature_names(obj, name='model'):
    printed = False
    if hasattr(obj, 'feature_names_in_'):
        print(f"{name}.feature_names_in_ (len={len(obj.feature_names_in_)}):")
        print(list(getattr(obj, 'feature_names_in_')))
        printed = True
    if hasattr(obj, 'get_feature_names_out'):
        try:
            out = obj.get_feature_names_out()
            print(f"{name}.get_feature_names_out() (len={len(out)}):")
            print(list(out))
            printed = True
        except Exception as e:
            print(f"{name}.get_feature_names_out() raised: {e}")
    return printed

printed_any = try_print_feature_names(model, 'model')

# If it's a pipeline, inspect named_steps
if hasattr(model, 'named_steps'):
    print('\nPipeline steps:')
    for step_name, step in model.named_steps.items():
        print(f' - {step_name}: {type(step)}')
        try_print_feature_names(step, f'step:{step_name}')

# Try to find a ColumnTransformer / preprocessor inside the pipeline
from sklearn.compose import ColumnTransformer
def find_column_transformer(obj):
    if isinstance(obj, ColumnTransformer):
        return obj
    if hasattr(obj, 'transformers_'):
        return obj
    if hasattr(obj, 'named_steps'):
        for s in obj.named_steps.values():
            ct = find_column_transformer(s)
            if ct is not None:
                return ct
    return None

ct = find_column_transformer(model)
if ct is None:
    print('\nNo ColumnTransformer/preprocessor found inside model pipeline.')
else:
    print('\nFound ColumnTransformer / preprocessor. Transformers:')
    for name, trans, cols in ct.transformers:
        print(f' - {name}: {type(trans)}  cols={cols}')
    # try to get output names if possible
    if hasattr(ct, 'get_feature_names_out'):
        try:
            fnames = ct.get_feature_names_out()
            print(f'Preprocessor output feature names (len={len(fnames)}):')
            print(list(fnames)[:200])
        except Exception as e:
            print('Could not get preprocessor feature names:', e)

print('\nDone.')

# Compare model expected features with engineered dataframe columns
try:
    # reuse sanitize/engineer functions from generate_all_plots if available
    from diagnostic_plots.generate_all_plots import sanitize_columns, engineer_features
    df2 = sanitize_columns(df)
    df2 = engineer_features(df2)
    X_cols = set(df2.drop(columns=['Length_of_stay'], errors='ignore').columns)
except Exception:
    # fallback: use sanitized names only
    X_cols = set([c.replace(' ', '_') for c in df.columns if c != 'Length of stay'])

model_features = set([str(x) for x in getattr(model, 'feature_names_in_', [])])

missing = sorted(model_features - X_cols)
extra = sorted(X_cols - model_features)
print('\nFeature comparison:')
print(f'  dataset columns (after engineering): {len(X_cols)}')
print(f'  model expected features: {len(model_features)}')
print(f'  missing (in model but not in data): {len(missing)}')
if missing:
    for m in missing[:200]:
        print('   -', m)
print(f'  extra (in data but not expected by model): {len(extra)}')
if extra:
    for e in extra[:200]:
        print('   -', e)

