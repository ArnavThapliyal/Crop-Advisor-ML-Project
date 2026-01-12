"""
Final Preprocessing Steps
Apply any remaining preprocessing steps before model training
"""

import pandas as pd
import numpy as np

def finalize_preprocessing(df, convert_bool=True, remove_duplicates=True, 
                          check_correlations=True, remove_low_variance=False):
    """
    Apply final preprocessing steps to the dataset
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    convert_bool : bool
        Convert boolean columns to int (0/1)
    remove_duplicates : bool
        Remove duplicate rows
    check_correlations : bool
        Flag highly correlated features for removal
    remove_low_variance : bool
        Remove features with very low variance (<0.01)
    
    Returns:
    --------
    df : DataFrame
        Preprocessed dataframe
    """
    
    print("🔧 Starting final preprocessing...")
    original_shape = df.shape
    
    # 1. Remove duplicates
    if remove_duplicates:
        before_dupes = df.shape[0]
        df = df.drop_duplicates()
        removed = before_dupes - df.shape[0]
        if removed > 0:
            print(f"   ✅ Removed {removed} duplicate rows")
        else:
            print("   ✅ No duplicates found")
    
    # 2. Convert boolean columns to int
    if convert_bool:
        bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
        if bool_cols:
            df[bool_cols] = df[bool_cols].astype(int)
            print(f"   ✅ Converted {len(bool_cols)} boolean columns to int")
        else:
            print("   ✅ No boolean columns to convert")
    
    # 3. Check and handle highly correlated features
    if check_correlations and 'yield_kg_per_hectare' in df.columns:
        numeric_features = df.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col != 'yield_kg_per_hectare']
        
        if len(numeric_features) > 1:
            corr_matrix = df[numeric_features].corr().abs()
            
            # Find pairs with high correlation
            high_corr_pairs = []
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            for col in upper_triangle.columns:
                high_corr = upper_triangle[col][upper_triangle[col] > 0.95]
                for corr_col, corr_val in high_corr.items():
                    # Keep the feature with higher correlation to target
                    if 'yield_kg_per_hectare' in df.columns:
                        corr_target_1 = abs(df[col].corr(df['yield_kg_per_hectare']))
                        corr_target_2 = abs(df[corr_col].corr(df['yield_kg_per_hectare']))
                        to_remove = corr_col if corr_target_2 < corr_target_1 else col
                        high_corr_pairs.append((to_remove, corr_val))
                    else:
                        high_corr_pairs.append((corr_col, corr_val))
            
            # Remove duplicate feature names
            features_to_remove = list(set([pair[0] for pair in high_corr_pairs]))
            
            if features_to_remove:
                print(f"   ⚠️ Found {len(features_to_remove)} highly correlated features")
                print(f"   Keeping features with higher target correlation")
                print(f"   Features to consider removing: {features_to_remove[:5]}...")
                # Don't auto-remove, just inform
                # df = df.drop(columns=features_to_remove)
            else:
                print("   ✅ No highly correlated features to remove")
    
    # 4. Remove low variance features (optional)
    if remove_low_variance:
        if 'yield_kg_per_hectare' in df.columns:
            X = df.drop('yield_kg_per_hectare', axis=1)
            numeric_X = X.select_dtypes(include=[np.number])
            
            if len(numeric_X.columns) > 0:
                variances = numeric_X.var()
                low_variance = variances[variances < 0.01]
                
                if len(low_variance) > 0:
                    df = df.drop(columns=low_variance.index)
                    print(f"   ✅ Removed {len(low_variance)} low variance features")
                else:
                    print("   ✅ No low variance features to remove")
    
    final_shape = df.shape
    print(f"\n   📊 Shape: {original_shape} → {final_shape}")
    print("   ✅ Preprocessing complete!")
    
    return df


# Main execution
if __name__ == "__main__":
    print("="*70)
    print("🔧 FINAL PREPROCESSING SCRIPT")
    print("="*70)
    
    # Load data
    df = pd.read_csv("Smart_Farming_Crop_Yield_MinMaxScaler.csv")
    print(f"\n📁 Loaded data: {df.shape}")
    
    # Apply preprocessing
    df_processed = finalize_preprocessing(
        df,
        convert_bool=True,        # Convert True/False to 1/0
        remove_duplicates=True,  # Remove duplicate rows
        check_correlations=True,  # Check for multicollinearity
        remove_low_variance=False  # Set to True to auto-remove low variance features
    )
    
    # Save the finalized dataset
    output_file = "Smart_Farming_Crop_Yield_Finalized.csv"
    df_processed.to_csv(output_file, index=False)
    print(f"\n💾 Saved finalized dataset: {output_file}")
    
    print("\n" + "="*70)
    print("✅ Data is ready for model training!")
    print("="*70)

