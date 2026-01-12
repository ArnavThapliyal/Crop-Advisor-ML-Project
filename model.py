from __future__ import annotations

import logging
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Iterable, List

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


TARGET_COLUMN = "yield_kg_per_hectare"
DEFAULT_DATASET_FILES = [
    Path(__file__).with_name("Smart_Farming_Crop_Yield_2024.csv"),
    Path(__file__).with_name("Smart_Farming_Crop_Yield_MinMaxScaler.csv"),
]
LOGGER = logging.getLogger(__name__)

REQUIRED_FIELDS: List[str] = [
    "soil_moisture",
    "soil_pH",
    "temperature",
    "rainfall",
    "humidity",
    "sunlight_hours",
    "pesticide_usage",
    "total_days",
    "latitude",
    "longitude",
    "NDVI_index",
    "crop_disease_status",
    "sowing_month",
    "harvest_month",
    "days_to_harvest",
    "fertilizer_type",
    "region",
    "crop_type",
    "irrigation_type",
]

COLUMN_RENAMES = {
    "soil_moisture_%": "soil_moisture",
    "soil_moisture_percent": "soil_moisture",
    "temperature_c": "temperature",
    "temperature_celsius": "temperature",
    "rainfall_mm": "rainfall",
    "rainfall_mm_per_season": "rainfall",
    "humidity_%": "humidity",
    "humidity_percent": "humidity",
    "sunlight_hrs": "sunlight_hours",
    "pesticide_usage_ml": "pesticide_usage",
}

ONE_HOT_PREFIXES = ["fertilizer_type", "region", "crop_type", "irrigation_type"]

DOMAIN_RANGES: Dict[str, tuple[float, float]] = {
    "soil_moisture": (0.0, 100.0),
    "soil_pH": (3.5, 8.5),
    "temperature": (0.0, 50.0),
    "rainfall": (0.0, 400.0),
    "humidity": (0.0, 100.0),
    "sunlight_hours": (0.0, 14.0),
    "pesticide_usage": (0.0, 200.0),
    "total_days": (0.0, 220.0),
    "latitude": (-90.0, 90.0),
    "longitude": (-180.0, 180.0),
    "NDVI_index": (0.0, 1.0),
    "sowing_month": (1.0, 12.0),
    "harvest_month": (1.0, 12.0),
    "days_to_harvest": (0.0, 250.0),
}

DISEASE_MAP = {
    "healthy": 0,
    "none": 0,
    "no": 0,
    "0": 0,
    0: 0,
    False: 0,
    "mild": 1,
    "severe": 1,
    "diseased": 1,
    "yes": 1,
    "1": 1,
    1: 1,
    True: 1,
}


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load CSV or JSON dataset."""
    path = Path(file_path)
    if not path.exists():
        LOGGER.error("Dataset file missing: %s", path)
        raise FileNotFoundError(f"Dataset {path} not found.")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        LOGGER.info("Loading CSV dataset: %s", path)
        return pd.read_csv(path)
    if suffix == ".json":
        LOGGER.info("Loading JSON dataset: %s", path)
        return pd.read_json(path)

    LOGGER.error("Unsupported dataset format for file: %s", path)
    raise ValueError("Unsupported dataset type. Please select CSV or JSON.")


def Train_Model(user_input: Dict[str, Any], dataset_df: pd.DataFrame) -> float:
    """Train a lightweight model on the provided dataset and predict yield."""
    if dataset_df is None or dataset_df.empty:
        LOGGER.warning("Dataset not provided or empty; loading default dataset.")
        dataset_df = load_default_dataset()

    processed = _prepare_dataset(dataset_df)
    missing_fields = [field for field in REQUIRED_FIELDS if field not in processed.columns]
    if missing_fields:
        LOGGER.error("Dataset missing required fields: %s", missing_fields)
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing_fields),
        )
    if TARGET_COLUMN not in processed.columns:
        LOGGER.error("Missing target column '%s' in dataset.", TARGET_COLUMN)
        raise ValueError(f"Dataset must include target column '{TARGET_COLUMN}'.")

    feature_df = processed[REQUIRED_FIELDS + [TARGET_COLUMN]].dropna()
    if feature_df.empty:
        LOGGER.error("No usable rows after preprocessing; cannot train model.")
        raise ValueError("Dataset has no usable rows after preprocessing.")

    user_row = _build_user_row(user_input)
    aligned = pd.concat([feature_df[REQUIRED_FIELDS], user_row], ignore_index=True)
    aligned = pd.get_dummies(aligned, columns=_categorical_fields(), dummy_na=False)

    train_features = aligned.iloc[:-1]
    input_features = aligned.iloc[-1:]
    target = feature_df[TARGET_COLUMN]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42,
    )
    model.fit(train_features, target)
    prediction = float(model.predict(input_features)[0])
    LOGGER.info("Prediction completed: %.2f", prediction)
    return prediction


def _prepare_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()
    df.columns = [col.strip() for col in df.columns]
    df.rename(columns=_rename_map(df.columns), inplace=True)

    _maybe_denormalize(df)
    _ensure_numeric_status(df)
    _collapse_one_hot_groups(df)
    _derive_temporal_fields(df)

    LOGGER.debug("Prepared dataset columns: %s", list(df.columns))
    return df


def _rename_map(columns: Iterable[str]) -> Dict[str, str]:
    mapping = {}
    lower_map = {k.lower(): v for k, v in COLUMN_RENAMES.items()}
    for col in columns:
        lower = col.lower()
        if lower in lower_map:
            mapping[col] = lower_map[lower]
    return mapping


def _maybe_denormalize(df: pd.DataFrame) -> None:
    for field, (min_val, max_val) in DOMAIN_RANGES.items():
        if field not in df.columns:
            continue
        if max_val <= min_val:
            continue
        series = pd.to_numeric(df[field], errors="coerce")
        if series.empty:
            continue
        col_min, col_max = series.min(), series.max()
        if pd.isna(col_min) or pd.isna(col_max):
            continue
        # Detect MinMax scaling (values roughly between 0 and 1)
        if col_min >= -0.05 and col_max <= 1.05 and max_val > 1.5:
            df[field] = series.clip(0, 1) * (max_val - min_val) + min_val
            LOGGER.info("Denormalized field '%s' using domain range [%s, %s].", field, min_val, max_val)


def _ensure_numeric_status(df: pd.DataFrame) -> None:
    col = "crop_disease_status"
    if col not in df.columns:
        return

    def _map(val: Any) -> int:
        if pd.isna(val):
            return 0
        if val in DISEASE_MAP:
            return int(DISEASE_MAP[val])
        key = str(val).strip().lower()
        return int(DISEASE_MAP.get(key, 1))

    df[col] = df[col].apply(_map).astype(int)


def _collapse_one_hot_groups(df: pd.DataFrame) -> None:
    for prefix in ONE_HOT_PREFIXES:
        columns = [c for c in df.columns if c.lower().startswith(f"{prefix.lower()}_")]
        if not columns:
            continue

        subset = df[columns].copy()
        subset = subset.apply(pd.to_numeric, errors="coerce").fillna(0)
        choice = subset.idxmax(axis=1).fillna("")
        df[prefix] = choice.str.replace(f"{prefix}_", "", regex=False).replace("", "Unknown")
        df.drop(columns=columns, inplace=True)


def _derive_temporal_fields(df: pd.DataFrame) -> None:
    if "sowing_month" not in df.columns and "sowing_date" in df.columns:
        df["sowing_month"] = pd.to_datetime(df["sowing_date"], errors="coerce").dt.month
    if "harvest_month" not in df.columns and "harvest_date" in df.columns:
        df["harvest_month"] = pd.to_datetime(df["harvest_date"], errors="coerce").dt.month
    if "days_to_harvest" not in df.columns:
        if {"harvest_date", "sowing_date"}.issubset(df.columns):
            sow = pd.to_datetime(df["sowing_date"], errors="coerce")
            har = pd.to_datetime(df["harvest_date"], errors="coerce")
            df["days_to_harvest"] = (har - sow).dt.days
        elif "total_days" in df.columns:
            df["days_to_harvest"] = df["total_days"]


def _build_user_row(user_input: Dict[str, Any]) -> pd.DataFrame:
    data = {field: user_input.get(field) for field in REQUIRED_FIELDS}
    return pd.DataFrame([data])


def _categorical_fields() -> List[str]:
    return ["fertilizer_type", "region", "crop_type", "irrigation_type"]


@lru_cache(maxsize=1)
def load_default_dataset() -> pd.DataFrame:
    last_error = None
    for path in DEFAULT_DATASET_FILES:
        if not path.exists():
            LOGGER.warning("Default dataset candidate missing: %s", path)
            continue
        try:
            LOGGER.info("Loading default dataset from %s", path)
            return load_dataset(path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to load dataset %s", path)
            last_error = exc
    raise FileNotFoundError(
        f"None of the default dataset files could be loaded: {', '.join(str(p) for p in DEFAULT_DATASET_FILES)}"
    ) from last_error


__all__ = ["load_dataset", "load_default_dataset", "Train_Model", "DOMAIN_RANGES", "COLUMN_RENAMES"]

