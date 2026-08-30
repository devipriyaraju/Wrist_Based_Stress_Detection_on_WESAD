"""WESAD .pkl loading and label alignment."""
import pickle
from pathlib import Path
import numpy as np
from .config import DATA_DIR, FS, ACC_TO_G


def load_subject(subj, data_dir=DATA_DIR):
    """Load one subject dict. WESAD pickles are Python-2 -> require latin1."""
    with open(Path(data_dir) / subj / f"{subj}.pkl", "rb") as f:
        return pickle.load(f, encoding="latin1")


def labels_for(data, fs_signal, n):
    """Resample the 700 Hz protocol label onto a length-n signal at fs_signal (nearest by time)."""
    lab = np.asarray(data["label"]).ravel()
    idx = np.clip(np.round(np.arange(n) / fs_signal * FS["label"]).astype(int), 0, len(lab) - 1)
    return lab[idx]


def get_channel(data, location, channel):
    """Return (signal_1d, fs). ACC collapsed to magnitude; wrist ACC converted to g."""
    sig = np.asarray(data["signal"][location][channel], dtype=float)
    if channel == "ACC":
        sig = np.sqrt((sig ** 2).sum(axis=1))
        fs = FS[f"{location}_ACC"] if location == "wrist" else FS["chest"]
        if location == "wrist":
            sig = sig / ACC_TO_G
    else:
        sig = sig.ravel()
        fs = FS.get(f"{location}_{channel}", FS["chest"])
    return sig, fs
