from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Dict, List

import pandas as pd

from model import COLUMN_RENAMES, DOMAIN_RANGES, Train_Model, load_default_dataset


LOG_FILE = Path(__file__).with_name("advisor.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)

DEFAULT_VALUES: Dict[str, Any] = {
    "soil_moisture": 22.5,
    "soil_pH": 6.5,
    "temperature": 29.0,
    "rainfall": 120,
    "humidity": 55,
    "sunlight_hours": 8,
    "pesticide_usage": 30,
    "total_days": 90,
    "latitude": 30.32,
    "longitude": 78.03,
    "NDVI_index": 0.67,
    "crop_disease_status": 0,
    "sowing_month": 6,
    "harvest_month": 10,
    "days_to_harvest": 125,
    "fertilizer_type": "Organic",
    "region": "North India",
    "crop_type": "Wheat",
    "irrigation_type": "Sprinkler",
}

FIELD_DEFS: List[Dict[str, Any]] = [
    {"name": "soil_moisture", "label": "Soil Moisture (%)", "type": "float"},
    {"name": "soil_pH", "label": "Soil pH", "type": "float"},
    {"name": "temperature", "label": "Temperature (°C)", "type": "float"},
    {"name": "rainfall", "label": "Rainfall (mm)", "type": "float"},
    {"name": "humidity", "label": "Humidity (%)", "type": "float"},
    {"name": "sunlight_hours", "label": "Sunlight Hours", "type": "float"},
    {"name": "pesticide_usage", "label": "Pesticide Usage (ml)", "type": "float"},
    {"name": "total_days", "label": "Total Days", "type": "int"},
    {"name": "latitude", "label": "Latitude", "type": "float"},
    {"name": "longitude", "label": "Longitude", "type": "float"},
    {"name": "NDVI_index", "label": "NDVI Index", "type": "float"},
    {
        "name": "crop_disease_status",
        "label": "Disease Status",
        "type": "choice_map",
        "options": [("Healthy (0)", 0), ("Diseased (1)", 1)],
    },
    {"name": "sowing_month", "label": "Sowing Month", "type": "int"},
    {"name": "harvest_month", "label": "Harvest Month", "type": "int"},
    {"name": "days_to_harvest", "label": "Days to Harvest", "type": "int"},
    {
        "name": "fertilizer_type",
        "label": "Fertilizer Type",
        "type": "choice",
        "options": ["Organic", "Inorganic", "Mixed", "Bio"],
    },
    {
        "name": "region",
        "label": "Region",
        "type": "choice",
        "options": ["North India", "South India", "East Africa", "South USA"],
    },
    {
        "name": "crop_type",
        "label": "Crop Type",
        "type": "choice",
        "options": ["Wheat", "Rice", "Maize", "Soybean"],
    },
    {
        "name": "irrigation_type",
        "label": "Irrigation Type",
        "type": "choice",
        "options": ["Sprinkler", "Drip", "Manual", "Surface"],
    },
]

RAW_DATA_FILE = Path(__file__).with_name("Smart_Farming_Crop_Yield_2024.csv")
NUMERIC_FIELDS = [field["name"] for field in FIELD_DEFS if field["type"] in {"float", "int"}]


class FarmerAdvisorUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Farmer's Adviser - Crop Yield Prediction")
        self.root.geometry("700x800")

        self.dataset = self._load_dataset_or_warn()

        self.prediction_var = tk.StringVar(value="Predicted Crop Yield: --")
        self.suggestions_var = tk.StringVar(value="Suggestions will appear here.")

        self.slider_bounds = _load_numeric_bounds(self.dataset)

        self.inputs: Dict[str, Any] = {}
        self.choice_maps: Dict[str, Dict[str, Any]] = {}
        self.progress_bar: ttk.Progressbar | None = None

        self._build_layout()

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill=tk.BOTH, expand=True)

        form_frame = ttk.LabelFrame(container, text="Input Parameters")
        form_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        for idx, field in enumerate(FIELD_DEFS):
            ttk.Label(form_frame, text=field["label"]).grid(row=idx, column=0, sticky=tk.W, padx=(0, 12), pady=4)
            widget = self._create_widget(form_frame, field)
            widget.grid(row=idx, column=1, sticky=tk.EW, pady=4)
            form_frame.grid_columnconfigure(1, weight=1)

        action_frame = ttk.Frame(container)
        action_frame.pack(fill=tk.X, pady=12)

        ttk.Button(action_frame, text="Predict Yield", command=self._handle_predict).pack(side=tk.LEFT, padx=(0, 12))
        self.progress_bar = ttk.Progressbar(action_frame, mode="indeterminate", length=180)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.progress_bar.stop()
        self.progress_bar["value"] = 0
        ttk.Label(container, textvariable=self.prediction_var, font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, pady=(0, 8))
        ttk.Label(
            container,
            textvariable=self.suggestions_var,
            wraplength=640,
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

    def _create_widget(self, parent: ttk.Frame, field: Dict[str, Any]):
        name = field["name"]
        default_value = DEFAULT_VALUES.get(name, "")

        if field["type"] == "choice":
            combo = ttk.Combobox(parent, state="readonly", values=field["options"])
            combo.set(default_value or field["options"][0])
            self.inputs[name] = combo
            return combo

        if field["type"] == "choice_map":
            options = [label for label, _ in field["options"]]
            mapping = {label: value for label, value in field["options"]}
            combo = ttk.Combobox(parent, state="readonly", values=options)
            default_label = next((label for label, value in field["options"] if value == default_value), options[0])
            combo.set(default_label)
            self.inputs[name] = combo
            self.choice_maps[name] = mapping
            return combo

        if field["type"] in {"float", "int"}:
            frame = ttk.Frame(parent)
            domain_min, domain_max = DOMAIN_RANGES.get(name, (0.0, float(DEFAULT_VALUES.get(name, 100.0))))
            max_value = self.slider_bounds.get(name, domain_max)
            min_value = domain_min
            if max_value <= min_value:
                max_value = max(min_value + 1.0, float(DEFAULT_VALUES.get(name, 1.0)))
            try:
                initial = float(default_value) if default_value != "" else min_value
            except ValueError:
                initial = min_value
            initial = min(max(initial, min_value), max_value)

            var = tk.DoubleVar(value=initial)
            scale = ttk.Scale(
                frame,
                from_=min_value,
                to=max_value,
                orient=tk.HORIZONTAL,
                variable=var,
                command=lambda val, lbl=name, ftype=field["type"]: self._update_slider_value(lbl, float(val), ftype),
            )
            value_label = ttk.Label(frame, text=self._format_value(var.get(), field["type"]))
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            value_label.pack(side=tk.LEFT, padx=8)
            self.inputs[name] = var
            self.inputs[f"{name}_label"] = value_label
            return frame

        entry = ttk.Entry(parent)
        entry.insert(0, str(default_value))
        self.inputs[name] = entry
        return entry

    def _update_slider_value(self, name: str, value: float, field_type: str) -> None:
        label_widget = self.inputs.get(f"{name}_label")
        if isinstance(label_widget, ttk.Label):
            label_widget.config(text=self._format_value(value, field_type))

    def _load_dataset_or_warn(self):
        try:
            dataset = load_default_dataset()
            LOGGER.info("Default dataset loaded successfully.")
            return dataset
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to load default dataset.")
            messagebox.showerror("Dataset Error", str(exc))
            return None

    def _handle_predict(self) -> None:
        if self.dataset is None:
            LOGGER.warning("Prediction requested but dataset is unavailable.")
            messagebox.showwarning("Dataset Missing", "Default dataset is unavailable.")
            return

        try:
            user_input = self._collect_user_input()
            LOGGER.debug("Collected user input: %s", user_input)
        except ValueError as exc:
            LOGGER.warning("Input validation error: %s", exc)
            messagebox.showerror("Invalid Input", str(exc))
            return

        self.prediction_var.set("Predicting crop yield...")
        self.suggestions_var.set("Generating insights...")
        LOGGER.info("Starting prediction thread.")
        if self.progress_bar is not None:
            self.progress_bar.start(10)
        threading.Thread(target=self._run_prediction, args=(user_input,), daemon=True).start()

    def _collect_user_input(self) -> Dict[str, Any]:
        collected: Dict[str, Any] = {}
        for field in FIELD_DEFS:
            name = field["name"]
            widget = self.inputs[name]
            if isinstance(widget, tk.Variable):
                raw_value = widget.get()
            else:
                raw_value = widget.get().strip()

            if not raw_value:
                default = DEFAULT_VALUES.get(name)
                if default is None:
                    raise ValueError(f"{field['label']} cannot be empty.")
                raw_value = str(default)

            if field["type"] in {"float", "int"}:
                try:
                    numeric_value = float(raw_value)
                    value = numeric_value if field["type"] == "float" else int(round(numeric_value))
                except ValueError as exc:
                    raise ValueError(f"{field['label']} must be a number.") from exc
                if name in DOMAIN_RANGES:
                    min_val, max_val = DOMAIN_RANGES[name]
                    if not (min_val <= numeric_value <= max_val):
                        raise ValueError(f"{field['label']} must be between {min_val} and {max_val}.")
                collected[name] = value
            elif field["type"] == "choice":
                collected[name] = raw_value
            elif field["type"] == "choice_map":
                mapping = self.choice_maps.get(name, {})
                if raw_value not in mapping:
                    raise ValueError(f"Invalid selection for {field['label']}.")
                collected[name] = mapping[raw_value]
            else:
                collected[name] = raw_value

        return collected

    def _run_prediction(self, user_input: Dict[str, Any]) -> None:
        try:
            prediction = Train_Model(user_input, self.dataset)
            suggestions = self._generate_suggestions(user_input)
            LOGGER.info("Prediction succeeded: %.2f", prediction)
            self.root.after(
                0,
                lambda: self._update_results(prediction, suggestions),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Prediction failed.")
            self.root.after(0, lambda: self._handle_prediction_error(exc))

    def _update_results(self, prediction: float, suggestions: List[str]) -> None:
        self.prediction_var.set(f"Predicted Crop Yield: {prediction:.2f} tons/hectare")
        if suggestions:
            self.suggestions_var.set("Suggestions:\n- " + "\n- ".join(suggestions))
        else:
            self.suggestions_var.set("Field conditions already look optimal.")
        if self.progress_bar is not None:
            self.progress_bar.stop()
            self.progress_bar["value"] = 0

    def _handle_prediction_error(self, exc: Exception) -> None:
        self.prediction_var.set("Prediction failed.")
        self.suggestions_var.set("Suggestions unavailable due to error.")
        LOGGER.error("Prediction error surfaced to user: %s", exc)
        if self.progress_bar is not None:
            self.progress_bar.stop()
            self.progress_bar["value"] = 0
        messagebox.showerror("Prediction Error", str(exc))

    def _generate_suggestions(self, user_input: Dict[str, Any]) -> List[str]:
        tips: List[str] = []
        if user_input["soil_moisture"] < 20:
            tips.append("Increase irrigation frequency to lift soil moisture above 25%.")
        if user_input["soil_pH"] < 5.5 or user_input["soil_pH"] > 7.5:
            tips.append("Adjust soil pH with lime or sulfur amendments to stay near neutral.")
        if user_input["NDVI_index"] < 0.5:
            tips.append("Boost NDVI via balanced fertilization and timely pest management.")
        if user_input["sunlight_hours"] < 6:
            tips.append("Consider pruning or spacing to improve sunlight exposure.")
        if user_input["pesticide_usage"] > 60:
            tips.append("Review pest program; high pesticide use can stress crops.")
        if user_input["crop_disease_status"] == 1:
            tips.append("Treat detected disease immediately to prevent further yield loss.")
        return tips[:4]

    @staticmethod
    def _format_value(value: float, field_type: str) -> str:
        if field_type == "int":
            return f"{int(round(value))}"
        return f"{value:.2f}"


def _load_numeric_bounds(source_df: pd.DataFrame | None) -> Dict[str, float]:
    df: pd.DataFrame | None = None
    if source_df is not None:
        df = source_df.copy()
    else:
        try:
            df = pd.read_csv(RAW_DATA_FILE)
            LOGGER.info("Loaded raw dataset for slider bounds: %s", RAW_DATA_FILE)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Cannot load raw dataset for slider bounds: %s", exc)

    if df is None or df.empty:
        LOGGER.warning("Falling back to default slider ranges.")
        bounds: Dict[str, float] = {}
        for field in NUMERIC_FIELDS:
            if field in DOMAIN_RANGES:
                bounds[field] = DOMAIN_RANGES[field][1]
            else:
                bounds[field] = float(DEFAULT_VALUES.get(field, 100))
        return bounds

    rename_map = {k.lower(): v for k, v in COLUMN_RENAMES.items()}
    df.columns = [col.strip() for col in df.columns]
    df.rename(columns={col: rename_map.get(col.lower(), col) for col in df.columns}, inplace=True)

    if "sowing_month" not in df.columns and "sowing_date" in df.columns:
        df["sowing_month"] = pd.to_datetime(df["sowing_date"], errors="coerce").dt.month
    if "harvest_month" not in df.columns and "harvest_date" in df.columns:
        df["harvest_month"] = pd.to_datetime(df["harvest_date"], errors="coerce").dt.month
    if "days_to_harvest" not in df.columns and {"sowing_date", "harvest_date"}.issubset(df.columns):
        sow = pd.to_datetime(df["sowing_date"], errors="coerce")
        har = pd.to_datetime(df["harvest_date"], errors="coerce")
        df["days_to_harvest"] = (har - sow).dt.days

    for field, (min_val, max_val) in DOMAIN_RANGES.items():
        if field not in df.columns or max_val <= min_val:
            continue
        series = pd.to_numeric(df[field], errors="coerce")
        if series.empty:
            continue
        col_min, col_max = series.min(), series.max()
        if pd.isna(col_min) or pd.isna(col_max):
            continue
        if col_min >= -0.05 and col_max <= 1.05 and max_val > 1.5:
            df[field] = series.clip(0, 1) * (max_val - min_val) + min_val

    bounds: Dict[str, float] = {}
    for field in NUMERIC_FIELDS:
        if field not in df.columns:
            continue
        series = pd.to_numeric(df[field], errors="coerce")
        max_val = series.max()
        if pd.isna(max_val):
            continue
        bounds[field] = float(max_val)

    LOGGER.debug("Slider bounds computed: %s", bounds)
    return bounds


def main() -> None:
    root = tk.Tk()
    FarmerAdvisorUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

