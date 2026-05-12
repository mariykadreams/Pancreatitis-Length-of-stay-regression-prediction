"""
Simple training script for Length of Stay regression.

Creates a preprocessing pipeline, trains a tree-based regressor,
evaluates on a hold-out test set, and saves the fitted pipeline.

Usage:
	python machinelearningmodele.py

Outputs:
	- model_pipeline.joblib

"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(path: str):
	df = pd.read_csv(path, index_col=0)
	if 'ID' in df.columns:
		try:
			df.drop(columns=['ID'], inplace=True)
		except Exception:
			pass
	return df


def make_stratify_bins(y, q=10):
	try:
		bins = pd.qcut(y, q=q, labels=False, duplicates='drop')
	except Exception:
		# fallback: simple equal-width bins
		bins = pd.cut(y, bins=q, labels=False)
	return bins


def build_pipeline(df, y_column):
	X = df.drop(columns=[y_column])

	numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
	categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

	# numeric transformer: median impute + scale
	numeric_transformer = Pipeline(steps=[
		('imputer', SimpleImputer(strategy='median')),
		('scaler', StandardScaler())
	])

	# categorical transformer: constant impute + one-hot
	# OneHotEncoder parameter name changed across sklearn versions
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

	# estimator: try to use XGBoost if available, otherwise RandomForest
	try:
		from xgboost import XGBRegressor
		estimator = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
	except Exception:
		estimator = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

	pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', estimator)])
	return pipeline


def train_and_evaluate(df, y_column='Length of stay'):
	df = df.copy()
	# drop rows without target
	df = df[df[y_column].notna()]

	X = df.drop(columns=[y_column])
	y = df[y_column].astype(float)

	strat_bins = make_stratify_bins(y, q=10)

	train_X, test_X, train_y, test_y = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=strat_bins
	)

	pipeline = build_pipeline(df, y_column)

	print('Training model...')
	pipeline.fit(train_X, train_y)

	print('Predicting on test set...')
	preds = pipeline.predict(test_X)

	mae = mean_absolute_error(test_y, preds)
	try:
		rmse = mean_squared_error(test_y, preds, squared=False)
	except TypeError:
		import math
		mse = mean_squared_error(test_y, preds)
		rmse = math.sqrt(mse)
	r2 = r2_score(test_y, preds)
	
    

	print(f'MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}')

	# Cross-validated MAE on training data (negative sign for sklearn convention)
	try:
		cv_scores = cross_val_score(pipeline, train_X, train_y, cv=5, scoring='neg_mean_absolute_error')
		cv_mae = -cv_scores.mean()
		print(f'CV MAE (5-fold): {cv_mae:.4f}')
	except Exception:
		pass

	# Persist pipeline
	out_path = Path('model_pipeline.joblib')
	joblib.dump(pipeline, out_path)
	print(f'Model pipeline saved to: {out_path.resolve()}')

	return {'mae': mae, 'rmse': rmse, 'r2': r2}


def main():
	data_path = Path('train.csv')
	if not data_path.exists():
		print('train.csv not found in current directory. Place the dataset next to this script.')
		return

	df = load_data(str(data_path))
	y_col = 'Length of stay'
	if y_col not in df.columns:
		print(f"Target column '{y_col}' not found. Columns: {list(df.columns)[:10]}")
		return

	metrics = train_and_evaluate(df, y_column=y_col)
	print('Done.')


if __name__ == '__main__':
	main()

