"""
Visualize feature importance as a bar chart.

Reads feature_importance.csv and creates publication-quality plots.

Outputs:
    - feature_importance_top20.png (top 20 positive features)
    - feature_importance_all.png (all features sorted)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance():
    """Create feature importance visualizations."""
    
    csv_path = Path('feature_importance.csv')
    if not csv_path.exists():
        print('feature_importance.csv not found.')
        return
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f'Loaded {len(df)} features')
    
    # Plot 1: Top 20 features (most positive)
    print('\n[1] Creating top 20 features plot...')
    top20 = df.nlargest(20, 'importance')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top20['importance']]
    bars = ax.barh(range(len(top20)), top20['importance'], color=colors)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['feature'])
    ax.set_xlabel('Permutation Importance', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Most Important Features for LOS Prediction', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top20['importance'])):
        ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
                va='center', ha='left', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('feature_importance_top20.png', dpi=300, bbox_inches='tight')
    print('Saved: feature_importance_top20.png')
    plt.close()
    
    # Plot 2: All features (separated by positive/negative)
    print('\n[2] Creating all features plot...')
    df_sorted = df.sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 16))
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in df_sorted['importance']]
    bars = ax.barh(range(len(df_sorted)), df_sorted['importance'], color=colors)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['feature'], fontsize=8)
    ax.set_xlabel('Permutation Importance', fontsize=12, fontweight='bold')
    ax.set_title('All Features: Permutation Importance for LOS Prediction', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-')
    
    plt.tight_layout()
    plt.savefig('feature_importance_all.png', dpi=300, bbox_inches='tight')
    print('Saved: feature_importance_all.png')
    plt.close()
    
    # Plot 3: Clinical vs. Derived features breakdown
    print('\n[3] Creating clinical vs. derived features analysis...')
    
    clinical_keywords = ['Age', 'Sex', 'BMI', 'Weight', 'Height', 'Smoker', 'Drinker', 
                        'CCI', 'DM', 'Dyslipidemias', 'Etiology', 'Previous AP',
                        'PCR', 'Creat', 'Urea', 'Leukocytes', 'PMN', 'Lymph', 'Mono',
                        'Eosinophils', 'Platelets', 'Hct', 'Albumin', 'Lipase', 'Amilasa',
                        'GOT', 'GPT', 'F alc', 'Ca', 'Phosphate', 'Glu', 'TG', 'Col',
                        'Ransom', 'BISAP', 'SIRS', 'Inflammatory', 'Pain', 'Onset',
                        'Abdominal', 'Chest X-Ray', 'Gasometría', 'CO3', 'PCO2', 'Ph',
                        'Exceso', 'Petrov', 'Waist', 'HBD']
    
    df['feature_type'] = df['feature'].apply(
        lambda x: 'Clinical (Original)' if any(kw in x for kw in clinical_keywords) 
        else 'Engineered (Derived)'
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    clinical = df[df['feature_type'] == 'Clinical (Original)']['importance']
    engineered = df[df['feature_type'] == 'Engineered (Derived)']['importance']
    
    data = [clinical, engineered]
    bp = ax.boxplot(data, labels=['Clinical (Original)', 'Engineered (Derived)'],
                     patch_artist=True)
    
    for patch, color in zip(bp['boxes'], ['#3498db', '#f39c12']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Permutation Importance', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance: Clinical vs. Engineered Features', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    print('Saved: feature_importance_comparison.png')
    plt.close()
    
    # Summary statistics
    print('\n' + '='*60)
    print('FEATURE IMPORTANCE SUMMARY')
    print('='*60)
    print(f'\nTop 5 Features:')
    for i, row in df.nlargest(5, 'importance').iterrows():
        print(f'  {i+1}. {row["feature"]}: {row["importance"]:.6f}')
    
    print(f'\nClinical vs. Engineered Features:')
    print(f'  Clinical features (mean): {clinical.mean():.6f}')
    print(f'  Engineered features (mean): {engineered.mean():.6f}')
    print(f'  Total features: {len(df)} ({len(df[df["feature_type"]=="Clinical (Original)"])} clinical, {len(df[df["feature_type"]=="Engineered (Derived)"])} engineered)')
    print('='*60)

if __name__ == '__main__':
    plot_feature_importance()
    print('\nDone. Open the PNG files to view visualizations.')
