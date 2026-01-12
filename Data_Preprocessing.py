import pandas as pd

"""
Run if PS venv is not working 
    .\venv\Scripts\Activate
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

"""

df = pd.read_csv("Smart_Farming_Crop_Yield_MinMaxScaler.csv")

## Normal Data Exploration
# print("="*60)
# print("🔹 FIRST 5 ROWS OF THE DATASET")
# print(df.head(), "\n")

# print("="*60)
# print("🔹 DATAFRAME INFO (DATA TYPES & NON-NULL COUNTS)")
# df.info()
# print("\n")

# print("="*60)
# print("🔹 STATISTICAL SUMMARY (NUMERIC COLUMNS)")
# print(df.describe(), "\n")

# print("="*60)
# print("🔹 MISSING VALUES PER COLUMN")
# print(df.isnull().sum(), "\n")

# print("="*60)
# print("🔹 NUMBER OF DUPLICATE ROWS")
# print(df.duplicated().sum(), "\n")

# print("="*60)
# print("🔹 COLUMN NAMES")
# print(df.columns.tolist(), "\n")

# print("="*60)
# print("🔹 DATAFRAME SHAPE (ROWS, COLUMNS)")
# print(df.shape)

# print(df.dtypes)

## train-test split

X = df.drop('yield_kg_per_hectare', axis=1)
y = df['yield_kg_per_hectare']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,       # 20% data for testing
    random_state=42,     # keeps results reproducible
)

print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)

# print(df.info())
# print(df.head())
# df.to_csv("Smart_Farming_Crop_Yield_MinMaxScaler.csv", index=False)
# print("✅")