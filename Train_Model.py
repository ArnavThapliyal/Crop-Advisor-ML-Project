from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "Smart_Farming_Crop_Yield_MinMaxScaler.csv")
df = pd.read_csv(csv_path)
X = df.drop('yield_kg_per_hectare', axis=1)
y = df['yield_kg_per_hectare']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

print(df.head())
# list(df.columns)
print(df.columns)

# print(df.isna().sum())
# print(df.describe())

model = RandomForestRegressor(n_estimators= 16, max_depth= 10, random_state=30)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

model_path = os.path.join(script_dir, 'crop_yield_model.pkl')
joblib.dump(model, model_path)

