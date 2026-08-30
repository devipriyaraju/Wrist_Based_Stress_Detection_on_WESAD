"""Central configuration: paths, dataset facts, windowing, and feature policy."""
from pathlib import Path

# Set WESAD_DIR to your dataset root (each subject at DATA_DIR/SX/SX.pkl).
DATA_DIR = Path("data/WESAD")

SUBJECTS = [f"S{i}" for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]  # S1, S12 dropped

FS = {"label": 700, "chest": 700, "wrist_ACC": 32, "wrist_BVP": 64, "wrist_EDA": 4, "wrist_TEMP": 4}
LABEL_MAP = {0: "transient", 1: "baseline", 2: "stress", 3: "amusement",
             4: "meditation", 5: "ignore", 6: "ignore", 7: "ignore"}
MODEL_STATES = {1: "baseline", 2: "stress", 3: "amusement"}

WINDOW_SEC, SHIFT_SEC, PURITY = 60, 60, 0.90
ACC_TO_G = 64.0

# Feature policy (see reports/model_card.md). HRV + hr_std unreliable on E4 wrist PPG -> excluded from model.
TRUSTED_FEATURES = ["eda_mean","eda_std","eda_min","eda_max","eda_slope","eda_scr_count",
                    "temp_mean","temp_std","temp_min","temp_max","temp_slope",
                    "hr_mean","acc_std","acc_energy","acc_mad","acc_max"]
UNTRUSTED_FEATURES = ["hr_std", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50"]  # kept in CSV as evidence, not modelled

OUT = Path("outputs"); TABLES = OUT / "tables"; FIGURES = OUT / "figures"
