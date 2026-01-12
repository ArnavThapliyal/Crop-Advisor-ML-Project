import json
import joblib
import xgboost as xgb
import pandas as pd

# -------------------------
# Load both ML models
# -------------------------
rf_model = joblib.load("crop_yield_model.pkl")          # Random Forest
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("crop_yield_xgb_model.json")       # XGB

# All required feature columns
FEATURE_COLUMNS = [
    'soil_moisture_%', 'soil_pH', 'temperature_C', 'rainfall_mm',
    'humidity_%', 'sunlight_hours', 'pesticide_usage_ml', 'total_days',
    'latitude', 'longitude', 'NDVI_index', 'crop_disease_status',
    'sowing_month', 'harvest_month', 'days_to_harvest',

    'fertilizer_type_Inorganic', 'fertilizer_type_Mixed',
    'fertilizer_type_Organic',

    'region_East Africa', 'region_North India',
    'region_South India', 'region_South USA',

    'crop_type_Maize', 'crop_type_Rice',
    'crop_type_Soybean', 'crop_type_Wheat',

    'irrigation_type_Manual', 'irrigation_type_Sprinkler',
    'irrigation_type_Unknown'
]

def preprocess_input(json_data):
    """Convert frontend JSON → ML model input row."""
    df = pd.DataFrame([{
        'soil_moisture_%': json_data['soil_moisture'],
        'soil_pH': json_data['soil_pH'],
        'temperature_C': json_data['temperature'],
        'rainfall_mm': json_data['rainfall'],
        'humidity_%': json_data['humidity'],
        'sunlight_hours': json_data['sunlight_hours'],
        'pesticide_usage_ml': json_data['pesticide_usage'],
        'total_days': json_data['total_days'],
        'latitude': json_data['latitude'],
        'longitude': json_data['longitude'],
        'NDVI_index': json_data['NDVI_index'],
        'crop_disease_status': json_data['crop_disease_status'],
        'sowing_month': json_data['sowing_month'],
        'harvest_month': json_data['harvest_month'],
        'days_to_harvest': json_data['days_to_harvest'],

        # one-hot encodings
        'fertilizer_type_Inorganic': 1 if json_data['fertilizer_type'] == "Inorganic" else 0,
        'fertilizer_type_Mixed':     1 if json_data['fertilizer_type'] == "Mixed" else 0,
        'fertilizer_type_Organic':   1 if json_data['fertilizer_type'] == "Organic" else 0,

        'region_East Africa':  1 if json_data['region'] == "East Africa" else 0,
        'region_North India':  1 if json_data['region'] == "North India" else 0,
        'region_South India':  1 if json_data['region'] == "South India" else 0,
        'region_South USA':    1 if json_data['region'] == "South USA" else 0,

        'crop_type_Maize':   1 if json_data['crop_type'] == "Maize" else 0,
        'crop_type_Rice':    1 if json_data['crop_type'] == "Rice" else 0,
        'crop_type_Soybean': 1 if json_data['crop_type'] == "Soybean" else 0,
        'crop_type_Wheat':   1 if json_data['crop_type'] == "Wheat" else 0,

        'irrigation_type_Manual':     1 if json_data['irrigation_type'] == "Manual" else 0,
        'irrigation_type_Sprinkler':  1 if json_data['irrigation_type'] == "Sprinkler" else 0,
        'irrigation_type_Unknown':    1 if json_data['irrigation_type'] == "Unknown" else 0,
    }])

    # Ensure all columns exist
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df


def predict_yield(json_data):
    input_df = preprocess_input(json_data)
    rf_pred = rf_model.predict(input_df)[0]
    xgb_pred = xgb_model.predict(input_df)[0]
    
    return {
        "RandomForest_yield": float(rf_pred),
        "XGBoost_yield": float(xgb_pred)
    }


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    with open("sample_input.json") as file:
        user_json = json.load(file)

    result = predict_yield(user_json)
    print(result)
