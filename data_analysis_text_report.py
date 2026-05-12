"""
Text-Based Data Analysis Report Generator
==========================================
This module provides functions to generate comprehensive text-based analysis reports
of the pancreatitis dataset. All output is in text format for easy sharing with AI.

Usage:
    python data_analysis_text_report.py
    or
    from data_analysis_text_report import *
    generate_full_analysis_report("train.csv")
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import random
import string


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & BASIC INFO
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str = "train.csv") -> pd.DataFrame:
    """Load CSV data and return DataFrame."""
    df = pd.read_csv(path, index_col=0)
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)
    return df


def basic_dataset_info(df: pd.DataFrame) -> str:
    """Generate basic information about the dataset."""
    report = []
    report.append("=" * 80)
    report.append("BASIC DATASET INFORMATION")
    report.append("=" * 80)
    report.append(f"Total Rows: {df.shape[0]}")
    report.append(f"Total Columns: {df.shape[1]}")
    report.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    report.append(f"Duplicate Rows: {df.duplicated().sum()}")
    
    report.append("\n--- Column Types ---")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        report.append(f"{str(dtype):20} : {count:3} columns")
    
    return "\n".join(report)


# ─────────────────────────────────────────────────────────────────────────────
# MISSING DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def missing_data_analysis(df: pd.DataFrame) -> str:
    """Generate comprehensive missing data analysis."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("MISSING DATA ANALYSIS")
    report.append("=" * 80)
    
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    total_missing_pct = (missing_cells / total_cells * 100)
    
    report.append(f"Total Missing Cells: {missing_cells} out of {total_cells} ({total_missing_pct:.2f}%)")
    
    # Missing per column
    missing_by_col = df.isnull().sum()
    missing_pct_by_col = (missing_by_col / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Column": missing_by_col.index,
        "Missing_Count": missing_by_col.values,
        "Missing_%": missing_pct_by_col.values
    }).sort_values("Missing_%", ascending=False)
    
    cols_with_missing = missing_df[missing_df["Missing_Count"] > 0]
    report.append(f"\nColumns with Missing Data: {len(cols_with_missing)}")
    report.append(f"Columns with Complete Data: {len(df.columns) - len(cols_with_missing)}")
    
    if len(cols_with_missing) > 0:
        report.append("\n--- Top 20 Columns by Missing Data ---")
        for _, row in cols_with_missing.head(20).iterrows():
            report.append(f"  {row['Column']:30} : {int(row['Missing_Count']):5} ({row['Missing_%']:6.2f}%)")
    
    report.append(f"\n--- Missing Data Ranges ---")
    report.append(f"  Min Missing: {missing_pct_by_col[missing_pct_by_col > 0].min():.2f}%")
    report.append(f"  Max Missing: {missing_pct_by_col.max():.2f}%")
    report.append(f"  Median Missing: {missing_pct_by_col[missing_pct_by_col > 0].median():.2f}%")
    
    return "\n".join(report)


# ─────────────────────────────────────────────────────────────────────────────
# NUMERICAL FEATURES ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def numerical_features_analysis(df: pd.DataFrame) -> str:
    """Generate detailed analysis of numerical features."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("NUMERICAL FEATURES ANALYSIS")
    report.append("=" * 80)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    report.append(f"Total Numerical Columns: {len(numerical_cols)}\n")
    
    # Summary statistics table
    report.append("--- Summary Statistics ---")
    report.append(f"{'Feature':<30} {'Count':>8} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    report.append("-" * 95)
    
    for col in numerical_cols:
        count = df[col].count()
        mean_val = df[col].mean()
        std_val = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        
        report.append(f"{col:<30} {count:>8} {mean_val:>12.3f} {std_val:>12.3f} {min_val:>12.3f} {max_val:>12.3f}")
    
    # Skewness and kurtosis
    report.append("\n--- Distribution Shape (Skewness & Kurtosis) ---")
    report.append(f"{'Feature':<30} {'Skewness':>12} {'Kurtosis':>12} {'Interpretation':<30}")
    report.append("-" * 85)
    
    for col in numerical_cols:
        skew = df[col].skew()
        kurt = df[col].kurtosis()
        
        if abs(skew) < 0.5:
            interp = "Fairly Symmetrical"
        elif skew > 0.5:
            interp = "Right Skewed"
        else:
            interp = "Left Skewed"
        
        report.append(f"{col:<30} {skew:>12.3f} {kurt:>12.3f}  {interp:<30}")
    
    return "\n".join(report)


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORICAL FEATURES ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def categorical_features_analysis(df: pd.DataFrame) -> str:
    """Generate detailed analysis of categorical features."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("CATEGORICAL FEATURES ANALYSIS")
    report.append("=" * 80)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(categorical_cols) == 0:
        # Check for columns with few unique values
        categorical_cols = [col for col in df.columns if df[col].nunique() < 20]
    
    report.append(f"Total Categorical Columns: {len(categorical_cols)}\n")
    
    for col in categorical_cols:
        report.append(f"\n--- {col} ---")
        value_counts = df[col].value_counts(dropna=False)
        total = len(df)
        
        report.append(f"Unique Values: {df[col].nunique()}")
        report.append(f"Missing: {df[col].isnull().sum()}")
        report.append("\nValue Distribution:")
        
        for value, count in value_counts.head(10).items():
            pct = (count / total * 100)
            bar = "█" * int(pct / 2)
            report.append(f"  {str(value):<20} : {count:>5} ({pct:>6.2f}%) {bar}")
    
    return "\n".join(report)


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def correlation_analysis(df: pd.DataFrame, target_col: str = "Length of stay") -> str:
    """Generate correlation analysis, especially with target variable."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("CORRELATION ANALYSIS")
    report.append("=" * 80)
    
    numerical_df = df.select_dtypes(include=[np.number])
    
    if target_col not in numerical_df.columns:
        report.append(f"Target column '{target_col}' not found or not numerical.")
        return "\n".join(report)
    
    # Correlation with target
    report.append(f"\n--- Correlation with Target Variable: '{target_col}' ---")
    correlations = numerical_df.corr()[target_col].sort_values(ascending=False)
    correlations = correlations[correlations.index != target_col]
    
    report.append(f"{'Feature':<30} {'Correlation':>12} {'Strength':<15}")
    report.append("-" * 60)
    
    for feature, corr in correlations.items():
        abs_corr = abs(corr)
        if abs_corr > 0.7:
            strength = "Very Strong"
        elif abs_corr > 0.5:
            strength = "Strong"
        elif abs_corr > 0.3:
            strength = "Moderate"
        elif abs_corr > 0.1:
            strength = "Weak"
        else:
            strength = "Very Weak"
        
        report.append(f"{feature:<30} {corr:>12.4f}  {strength:<15}")
    
    # Top positive and negative correlations
    report.append(f"\n--- Top 10 Positive Correlations ---")
    for feature, corr in correlations.head(10).items():
        report.append(f"  {feature:<30} : {corr:>8.4f}")
    
    report.append(f"\n--- Top 10 Negative Correlations ---")
    for feature, corr in correlations.tail(10).items():
        report.append(f"  {feature:<30} : {corr:>8.4f}")
    
    # Inter-feature correlations (highly correlated pairs)
    report.append(f"\n--- Highly Correlated Feature Pairs (|r| > 0.8) ---")
    corr_matrix = numerical_df.corr()
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    if high_corr_pairs:
        for col1, col2, corr in high_corr_pairs[:20]:
            report.append(f"  {col1:<30} <-> {col2:<30} : {corr:>8.4f}")
    else:
        report.append("  No highly correlated pairs found.")
    
    return "\n".join(report)


# ─────────────────────────────────────────────────────────────────────────────
# OUTLIER ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def outlier_analysis(df: pd.DataFrame) -> str:
    """Detect and report outliers using IQR method."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("OUTLIER ANALYSIS (IQR Method)")
    report.append("=" * 80)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    report.append(f"{'Feature':<30} {'Outliers':>10} {'%':>8} {'Min_Outlier':>12} {'Max_Outlier':>12}")
    report.append("-" * 85)
    
    total_outliers = 0
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outliers_mask.sum()
        outlier_pct = (outlier_count / len(df) * 100)
        
        if outlier_count > 0:
            outlier_values = df[col][outliers_mask]
            min_outlier = outlier_values.min()
            max_outlier = outlier_values.max()
            total_outliers += outlier_count
            
            report.append(f"{col:<30} {outlier_count:>10} {outlier_pct:>7.2f}% {min_outlier:>12.3f} {max_outlier:>12.3f}")
    
    report.append("-" * 85)
    report.append(f"Total Outlier Instances: {total_outliers}")
    
    return "\n".join(report)


# ─────────────────────────────────────────────────────────────────────────────
# TARGET VARIABLE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def target_analysis(df: pd.DataFrame, target_col: str = "Length of stay") -> str:
    """Detailed analysis of target variable."""
    report = []
    report.append("\n" + "=" * 80)
    report.append(f"TARGET VARIABLE ANALYSIS: '{target_col}'")
    report.append("=" * 80)
    
    if target_col not in df.columns:
        report.append(f"Target column '{target_col}' not found.")
        return "\n".join(report)
    
    y = df[target_col].dropna()
    
    report.append(f"\nBasic Statistics:")
    report.append(f"  Count       : {len(y)}")
    report.append(f"  Missing     : {df[target_col].isnull().sum()}")
    report.append(f"  Mean        : {y.mean():.3f}")
    report.append(f"  Median      : {y.median():.3f}")
    report.append(f"  Std Dev     : {y.std():.3f}")
    report.append(f"  Min         : {y.min():.3f}")
    report.append(f"  Max         : {y.max():.3f}")
    report.append(f"  Q1 (25%)    : {y.quantile(0.25):.3f}")
    report.append(f"  Q3 (75%)    : {y.quantile(0.75):.3f}")
    report.append(f"  IQR         : {y.quantile(0.75) - y.quantile(0.25):.3f}")
    
    report.append(f"\nDistribution Characteristics:")
    report.append(f"  Skewness    : {y.skew():.3f}")
    report.append(f"  Kurtosis    : {y.kurtosis():.3f}")
    report.append(f"  Range       : {y.max() - y.min():.3f}")
    report.append(f"  Coefficient of Variation : {(y.std() / y.mean()):.3f}")
    
    # Log transformation effect
    log_y = np.log1p(y)
    report.append(f"\nLog Transformation Effect (log1p):")
    report.append(f"  Original Skewness : {y.skew():.3f}")
    report.append(f"  Log Skewness      : {log_y.skew():.3f}")
    report.append(f"  Skewness Reduction: {abs(y.skew()) - abs(log_y.skew()):.3f}")
    
    # Percentile breakdown
    report.append(f"\nPercentile Breakdown:")
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    for p in percentiles:
        val = y.quantile(p / 100)
        report.append(f"  {p:>2}th percentile : {val:>8.3f}")
    
    return "\n".join(report)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def feature_groups_analysis(df: pd.DataFrame) -> str:
    """Analyze features by groups (temporal, lab, etc.)."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("FEATURE GROUP ANALYSIS")
    report.append("=" * 80)
    
    # Define feature groups based on naming patterns
    feature_groups = {
        "Demographic": ["Age", "Sex", "Weight", "Height", "BMI"],
        "Lifestyle": ["Drinker", "Grams/Day", "Smoker", "Packs/Year"],
        "Medical History": ["Etiology", "Previous AP", "CCI", "HBD", "DM", "Dyslipidemias"],
        "Clinical Exam": ["Abdominal Exam", "Onset pain", "Pain Intensity"],
        "PCR (Procalcitonin)": ["PCR Adm", "PCR 48h", "PCR 72h", "PCR/Alb ratio"],
        "Ransom Score": ["Ransom Adm", "Ransom 48h"],
        "BISAP Score": ["BISAP"],
        "Kidney Function": ["Urea Adm", "Urea 48h", "Urea 72h", "Creat Adm", "Creat 48h", "Creat 72h"],
        "Blood Cells": ["Hct Adm", "Hct 48h", "Hct 72h", "Leukocytes Adm", "Leukocytes 48h", "Leukocytes 72h"],
        "White Blood Cells": ["PMN Adm", "PMN 48h", "PMN 72h", "Lymphocytes Adm", "Lymph 48h", "Lymph 72h"],
        "Liver Function": ["GOT", "GPT", "F alc", "Albumin", "Bilirubin"],
        "Enzymes & Markers": ["Amilasa", "Lipase"],
        "Time Measurements": ["Adm", "48h", "72h"],
    }
    
    for group_name, keywords in feature_groups.items():
        # Find columns matching this group
        matching_cols = [col for col in df.columns if any(kw in col for kw in keywords)]
        
        if matching_cols:
            report.append(f"\n--- {group_name} ({len(matching_cols)} features) ---")
            
            for col in matching_cols:
                if col in df.columns:
                    missing_count = df[col].isnull().sum()
                    missing_pct = (missing_count / len(df) * 100)
                    
                    if df[col].dtype in [np.float64, np.int64]:
                        mean_val = df[col].mean()
                        std_val = df[col].std()
                        report.append(f"  {col:<30} : Mean={mean_val:>10.2f}, Std={std_val:>8.2f}, Missing={missing_pct:>6.2f}%")
                    else:
                        unique = df[col].nunique()
                        report.append(f"  {col:<30} : Unique={unique:>3}, Missing={missing_pct:>6.2f}%")
    
    return "\n".join(report)


# ─────────────────────────────────────────────────────────────────────────────
# DATA QUALITY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def data_quality_report(df: pd.DataFrame) -> str:
    """Generate an overall data quality assessment."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("DATA QUALITY ASSESSMENT")
    report.append("=" * 80)
    
    # Completeness
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells * 100)
    
    report.append(f"\n1. COMPLETENESS")
    report.append(f"   Overall Data Completeness: {completeness:.2f}%")
    report.append(f"   Complete Rows (no missing): {(~df.isnull().any(axis=1)).sum()}")
    report.append(f"   Rows with at least 1 missing: {(df.isnull().any(axis=1)).sum()}")
    
    # Consistency
    report.append(f"\n2. CONSISTENCY")
    report.append(f"   Duplicate Rows: {df.duplicated().sum()}")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    negative_counts = {col: (df[col] < 0).sum() for col in numerical_cols}
    unexpected_negatives = {k: v for k, v in negative_counts.items() if v > 0 and 'negative' not in k.lower()}
    
    if unexpected_negatives:
        report.append(f"   Columns with Unexpected Negative Values:")
        for col, count in unexpected_negatives.items():
            report.append(f"     - {col}: {count} values")
    else:
        report.append(f"   No unexpected negative values detected")
    
    # Validity
    report.append(f"\n3. VALIDITY")
    report.append(f"   Numeric Columns: {len(numerical_cols)}")
    
    # Check for extreme values
    report.append(f"   Features with Extreme Ranges (max/min > 1000):")
    extreme_features = []
    for col in numerical_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val != 0:
            range_ratio = abs(max_val / min_val) if min_val != 0 else max_val
            if range_ratio > 1000:
                extreme_features.append(col)
                report.append(f"     - {col}: min={min_val:.2f}, max={max_val:.2f}, ratio={range_ratio:.0f}")
    
    if not extreme_features:
        report.append("     None found")
    
    # Accuracy (data distribution quality)
    report.append(f"\n4. ACCURACY / DISTRIBUTION QUALITY")
    skewed_features = []
    for col in numerical_cols:
        skew_val = df[col].skew()
        if abs(skew_val) > 2:
            skewed_features.append((col, skew_val))
    
    if skewed_features:
        report.append(f"   Highly Skewed Features (|skewness| > 2):")
        for col, skew_val in sorted(skewed_features, key=lambda x: abs(x[1]), reverse=True)[:10]:
            report.append(f"     - {col}: {skew_val:.2f}")
    
    # Overall Quality Score
    report.append(f"\n5. OVERALL QUALITY SCORE")
    quality_score = completeness
    report.append(f"   Completeness Score: {quality_score:.1f}/100")
    
    if quality_score >= 90:
        rating = "EXCELLENT"
    elif quality_score >= 80:
        rating = "GOOD"
    elif quality_score >= 70:
        rating = "FAIR"
    else:
        rating = "POOR"
    
    report.append(f"   Rating: {rating}")
    
    return "\n".join(report)


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON ANALYSIS (Train vs Test)
# ─────────────────────────────────────────────────────────────────────────────

def train_test_comparison(train_path: str = "train.csv", test_path: str = "test.csv") -> str:
    """Compare distributions between train and test sets."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("TRAIN vs TEST SET COMPARISON")
    report.append("=" * 80)
    
    try:
        train_df = load_data(train_path)
        test_df = load_data(test_path)
    except FileNotFoundError:
        report.append("Train or test file not found.")
        return "\n".join(report)
    
    report.append(f"\nDataset Sizes:")
    report.append(f"  Train: {train_df.shape[0]} rows × {train_df.shape[1]} columns")
    report.append(f"  Test:  {test_df.shape[0]} rows × {test_df.shape[1]} columns")
    
    common_cols = set(train_df.columns) & set(test_df.columns)
    report.append(f"\nCommon Columns: {len(common_cols)}")
    
    train_only = set(train_df.columns) - set(test_df.columns)
    test_only = set(test_df.columns) - set(train_df.columns)
    
    if train_only:
        report.append(f"Train-only Columns: {', '.join(sorted(train_only))}")
    if test_only:
        report.append(f"Test-only Columns: {', '.join(sorted(test_only))}")
    
    # Compare numerical distributions
    report.append(f"\n--- Numerical Feature Distribution Comparison ---")
    numerical_cols = [col for col in common_cols if train_df[col].dtype in [np.float64, np.int64]]
    
    report.append(f"{'Feature':<30} {'Train Mean':>12} {'Test Mean':>12} {'Diff':>12} {'Train Std':>12} {'Test Std':>12}")
    report.append("-" * 95)
    
    for col in sorted(numerical_cols):
        train_mean = train_df[col].mean()
        test_mean = test_df[col].mean()
        diff = test_mean - train_mean
        train_std = train_df[col].std()
        test_std = test_df[col].std()
        
        report.append(f"{col:<30} {train_mean:>12.3f} {test_mean:>12.3f} {diff:>12.3f} {train_std:>12.3f} {test_std:>12.3f}")
    
    return "\n".join(report)


# ─────────────────────────────────────────────────────────────────────────────
# COMPLETE ANALYSIS REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_full_analysis_report(data_path: str = "train.csv", output_file: str = None) -> str:
    """Generate complete text-based analysis report and save to file."""
    
    # Generate filename with random number if not provided
    if output_file is None:
        random_num = ''.join(random.choices(string.digits, k=6))
        output_file = f"data_analysis_report_{random_num}.txt"
    
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    
    print("Generating analysis report...")
    
    # Compile all sections
    sections = [
        "DATA ANALYSIS REPORT - TEXT FORMAT",
        "=" * 80,
        "This report contains comprehensive text-based analysis suitable for AI analysis.",
        "",
        basic_dataset_info(df),
        missing_data_analysis(df),
        numerical_features_analysis(df),
        categorical_features_analysis(df),
        target_analysis(df),
        correlation_analysis(df),
        outlier_analysis(df),
        feature_groups_analysis(df),
        data_quality_report(df),
        train_test_comparison(),
    ]
    
    full_report = "\n".join(sections)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_report)
    
    print(f"Report saved to: {output_file}")
    
    return full_report


def print_section(section_func, df: pd.DataFrame = None, *args, **kwargs):
    """Print a specific analysis section."""
    if df is None:
        df = load_data()
    
    result = section_func(df, *args, **kwargs)
    print(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    # Generate report with random filename
    report = generate_full_analysis_report("train.csv")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("Report generation completed successfully!")
