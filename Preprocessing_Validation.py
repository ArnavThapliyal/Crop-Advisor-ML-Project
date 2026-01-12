"""
Comprehensive Preprocessing Validation Script
Checks what additional processing might be needed before training
"""

import pandas as pd
import numpy as np

# Load the preprocessed data
df = pd.read_csv("Smart_Farming_Crop_Yield_Finalized.csv")

print("="*70)
print(" PREPROCESSING VALIDATION REPORT")
print("="*70)

# 1. Basic Data Info
print("\n BASIC DATA INFORMATION")
print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 2. Missing Values Check
print("\n MISSING VALUES")
missing = df.isnull().sum()
if missing.sum() == 0:
    print(" No missing values found")
else:
    print(" Missing values detected:")
    print(missing[missing > 0])

# 3. Duplicate Rows Check
print("\n DUPLICATE ROWS")
duplicates = df.duplicated().sum()
if duplicates == 0:
    print(" No duplicate rows found")
else:
    print(f" {duplicates} duplicate rows found")
    print("   Consider removing duplicates before training")

# 4. Data Types Check
print("\n DATA TYPES")
print(f"   Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
print(f"   Boolean columns: {len(df.select_dtypes(include=['bool']).columns)}")
print(f"   Object columns: {len(df.select_dtypes(include=['object']).columns)}")

# Check if boolean columns need conversion
bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
if bool_cols:
    print(f" Boolean columns found: {bool_cols}")
    print("   Consider converting to int (0/1) for some ML algorithms")
    print("   Most algorithms accept bool, but int is more universal")

# 5. Target Variable Check
print("\n TARGET VARIABLE (yield_kg_per_hectare)")
if 'yield_kg_per_hectare' in df.columns:
    target = df['yield_kg_per_hectare']
    print(f"   Min: {target.min():.2f}")
    print(f"   Max: {target.max():.2f}")
    print(f"   Mean: {target.mean():.2f}")
    print(f"   Median: {target.median():.2f}")
    print(f"   Std: {target.std():.2f}")
    
    # Check for outliers (using IQR method)
    Q1 = target.quantile(0.25)
    Q3 = target.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((target < lower_bound) | (target > upper_bound)).sum()
    if outliers > 0:
        print(f" {outliers} potential outliers detected (IQR method)")
        print("   Consider: log transformation, capping, or robust scaling")
    else:
        print(" No significant outliers detected")
else:
    print(" Target variable not found!")

# 6. Feature Scaling Check
print("\n FEATURE SCALING")
numeric_cols = df.select_dtypes(include=[np.number]).columns
if 'yield_kg_per_hectare' in numeric_cols:
    numeric_cols = numeric_cols.drop('yield_kg_per_hectare')

scaled_check = df[numeric_cols].describe().loc[['min', 'max']]
all_scaled = ((scaled_check.loc['min'] >= 0) & (scaled_check.loc['max'] <= 1)).all()

if all_scaled:
    print(" All numeric features appear to be MinMaxScaled (0-1 range)")
else:
    print(" Some features may not be properly scaled")
    print("   Features outside 0-1 range:")
    unscaled = scaled_check.columns[
        (scaled_check.loc['min'] < 0) | (scaled_check.loc['max'] > 1)
    ]
    if len(unscaled) > 0:
        print(f"   {list(unscaled)}")

# 7. Categorical Encoding Check
print("\n CATEGORICAL ENCODING")
# Check for one-hot encoded columns
one_hot_patterns = [
    'fertilizer_type_', 'region_', 'crop_type_', 'irrigation_type_'
]
one_hot_cols = [col for pattern in one_hot_patterns 
                for col in df.columns if pattern in col]
if one_hot_cols:
    print(f" {len(one_hot_cols)} one-hot encoded columns found")
    # Check if all categories sum to 1 (proper encoding)
    fertilizer_cols = [col for col in df.columns if 'fertilizer_type_' in col]
    if fertilizer_cols:
        fertilizer_sum = df[fertilizer_cols].sum(axis=1)
        if (fertilizer_sum == 1).all():
            print(" Fertilizer type columns properly encoded (sum to 1)")
        else:
            print(" Some rows don't sum to 1 - check encoding")

# 8. Feature Correlation Check
print("\n  FEATURE CORRELATION")
if 'yield_kg_per_hectare' in df.columns:
    numeric_features = df.select_dtypes(include=[np.number]).columns
    if len(numeric_features) > 1:
        corr_matrix = df[numeric_features].corr()
        target_corr = corr_matrix['yield_kg_per_hectare'].drop('yield_kg_per_hectare')
        
        # Check for high correlations between features
        high_corr_features = []
        for col in numeric_features:
            if col != 'yield_kg_per_hectare':
                corr_vals = corr_matrix[col].drop([col, 'yield_kg_per_hectare'])
                high_corr = corr_vals[abs(corr_vals) > 0.95]
                if len(high_corr) > 0:
                    high_corr_features.extend([(col, corr_col, val) 
                                             for corr_col, val in high_corr.items()])
        
        if high_corr_features:
            print(f" {len(high_corr_features)} pairs of highly correlated features (>0.95)")
            print("   Consider: Removing one feature from each pair to reduce multicollinearity")
            # Show first few
            for feat1, feat2, corr_val in high_corr_features[:5]:
                print(f"      {feat1} <-> {feat2}: {corr_val:.3f}")
        else:
            print(" No highly correlated features detected")
        
        # Check correlation with target
        top_corr = target_corr.abs().sort_values(ascending=False).head(5)
        print("\n   Top 5 features correlated with target:")
        for feat, corr in top_corr.items():
            print(f"      {feat}: {corr:.3f}")

# 9. Feature Variance Check
print("\n LOW VARIANCE FEATURES")
if 'yield_kg_per_hectare' in df.columns:
    X = df.drop('yield_kg_per_hectare', axis=1)
    numeric_X = X.select_dtypes(include=[np.number])
    if len(numeric_X.columns) > 0:
        variances = numeric_X.var()
        low_variance = variances[variances < 0.01]
        if len(low_variance) > 0:
            print(f"   {len(low_variance)} features with very low variance (<0.01)")
            print("   Consider removing these features")
            print(f"   Features: {list(low_variance.index)}")
        else:
            print("   No low variance features detected")

# 10. Data Balance Check
print("\n DATA BALANCE (for categorical features)")
for pattern in ['crop_type_', 'region_', 'fertilizer_type_']:
    cols = [col for col in df.columns if pattern in col]
    if cols:
        class_counts = df[cols].sum()
        min_count = class_counts.min()
        max_count = class_counts.max()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        if imbalance_ratio > 3:
            print(f"   {pattern.replace('_', '')} shows imbalance (ratio: {imbalance_ratio:.2f})")
            print(f"      Min: {min_count}, Max: {max_count}")
        else:
            print(f"   {pattern.replace('_', '')} relatively balanced")

# 11. Final Recommendations
print("\n" + "="*70)
print(" RECOMMENDATIONS")
print("="*70)

recommendations = []

if duplicates > 0:
    recommendations.append("• Remove duplicate rows: df.drop_duplicates(inplace=True)")

if bool_cols:
    recommendations.append(f"• Convert boolean columns to int: df[{bool_cols}] = df[{bool_cols}].astype(int)")

if 'yield_kg_per_hectare' in df.columns:
    target = df['yield_kg_per_hectare']
    if target.skew() > 2 or target.skew() < -2:
        recommendations.append("• Target variable is skewed - consider log transformation or Box-Cox")

if recommendations:
    print("\n   Additional preprocessing steps recommended:")
    for rec in recommendations:
        print(rec)
else:
    print("\n  Your data appears to be well preprocessed!")
    print("   Ready for model training!")

print("\n" + "="*70)
print(" OPTIONAL ADVANCED PREPROCESSING")
print("="*70)
print("""
   Consider these optional steps:
   
   1. Feature Engineering:
      - Interaction features (e.g., temperature × humidity)
      - Polynomial features for important predictors
      - Domain-specific features (e.g., growing degree days)
   
   2. Feature Selection:
      - Remove highly correlated features
      - Use feature importance from models
      - Recursive feature elimination
   
   3. Dimensionality Reduction (if many features):
      - PCA (Principal Component Analysis)
      - Feature selection methods
   
   4. Target Transformation (if needed):
      - Log transformation for skewed targets
      - Box-Cox transformation
   
   5. Train-Test Split:
      - Already in your script (good!)
      - Consider stratified split if dealing with imbalanced data
   
   6. Cross-Validation:
      - K-Fold cross-validation for robust evaluation
      - Time-series cross-validation if temporal order matters
""")

print("\n" + "="*70)
print("   VALIDATION COMPLETE")
print("="*70)

