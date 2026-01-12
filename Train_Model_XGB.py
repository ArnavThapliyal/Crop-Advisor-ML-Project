"""
Train an XGBoost regressor on the Smart Farming dataset and report metrics.
"""

import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split


def load_dataset(csv_filename: str) -> pd.DataFrame:
    """Load the preprocessed dataset relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, csv_filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find dataset at {csv_path}")
    return pd.read_csv(csv_path)


def main() -> None:
    df = load_dataset("Smart_Farming_Crop_Yield_MinMaxScaler.csv")

    target_col = "yield_kg_per_hectare"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=30,
    )

    model = xgb.XGBRegressor(
        n_estimators=16,
        learning_rate=0.0001,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=30,
        objective="reg:squarederror",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Hold-out RMSE: {rmse:.2f}")
    print(f"Hold-out R²: {r2:.4f}")

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, scoring="r2", cv=kfold, n_jobs=-1)
    print(f"5-fold CV R² mean: {cv_scores.mean():.4f} (std: {cv_scores.std():.4f})")

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "crop_yield_xgb_model.json",
    )
    model.save_model(model_path)
    print(f"Saved trained XGBoost model to {model_path}")

    feature_importances = sorted(
        zip(X.columns, model.feature_importances_), key=lambda x: x[1], reverse=True
    )
    print("\nTop feature importances:")
    for feature, importance in feature_importances:
        print(f"{feature:30s}: {importance:.4f}")


if __name__ == "__main__":
    main()

