"""Sliding-window feature extraction from wrist signals (EDA, TEMP, BVP->HR/HRV, ACC)."""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from .config import FS, MODEL_STATES, WINDOW_SEC, SHIFT_SEC, PURITY, SUBJECTS, UNTRUSTED_FEATURES
from .data import load_subject, get_channel, labels_for


def _bandpass(x, fs, lo, hi, order=3):
    ny = 0.5 * fs
    b, a = butter(order, [lo / ny, hi / ny], btype="band")
    return filtfilt(b, a, x)


def eda_features(x, fs):
    x = np.asarray(x, float)
    if len(x) < 4:
        return dict.fromkeys(["eda_mean","eda_std","eda_min","eda_max","eda_slope","eda_scr_count"], np.nan)
    t = np.arange(len(x)) / fs
    coef = np.polyfit(t, x, 1)
    peaks, _ = find_peaks(x - np.polyval(coef, t), prominence=0.01)
    return dict(eda_mean=x.mean(), eda_std=x.std(ddof=1), eda_min=x.min(),
                eda_max=x.max(), eda_slope=coef[0], eda_scr_count=float(len(peaks)))


def temp_features(x, fs):
    x = np.asarray(x, float)
    if len(x) < 4:
        return dict.fromkeys(["temp_mean","temp_std","temp_min","temp_max","temp_slope"], np.nan)
    t = np.arange(len(x)) / fs
    return dict(temp_mean=x.mean(), temp_std=x.std(ddof=1), temp_min=x.min(),
                temp_max=x.max(), temp_slope=np.polyfit(t, x, 1)[0])


def bvp_features(x, fs):
    """Quality-gated cardiac features. HR trusted; HRV NaN unless the window's beats are clean.
    See model card: E4 wrist HRV is unreliable, so hrv_* are excluded from the model."""
    keys = ["hr_mean","hr_std","hrv_sdnn","hrv_rmssd","hrv_pnn50","ppg_quality"]
    x = np.asarray(x, float).ravel()
    if len(x) < fs * 5 or np.allclose(x.std(), 0):
        return dict.fromkeys(keys, np.nan)
    try:
        xf = _bandpass(x, fs, 0.5, 4.0)
    except Exception:
        return dict.fromkeys(keys, np.nan)
    xf = (xf - xf.mean()) / (xf.std() + 1e-9)
    peaks, _ = find_peaks(xf, distance=int(0.4 * fs), prominence=0.5)
    if len(peaks) < 5:
        return dict.fromkeys(keys, np.nan)
    ibi = np.diff(peaks) / fs * 1000.0
    ibi_ok = ibi[(ibi > 300) & (ibi < 1600)]
    if len(ibi_ok) < 4:
        return dict(hr_mean=np.nan, hr_std=np.nan, hrv_sdnn=np.nan,
                    hrv_rmssd=np.nan, hrv_pnn50=np.nan, ppg_quality=0.0)
    med = np.median(ibi_ok)
    clean = ibi_ok[np.abs(ibi_ok - med) < 0.30 * med]
    quality = len(clean) / len(ibi)
    hr = 60000.0 / ibi_ok
    out = dict(hr_mean=float(hr.mean()), hr_std=float(hr.std(ddof=1)), ppg_quality=float(quality))
    if quality >= 0.80 and len(clean) >= 4:
        out.update(hrv_sdnn=float(clean.std(ddof=1)),
                   hrv_rmssd=float(np.sqrt(np.mean(np.diff(clean) ** 2))),
                   hrv_pnn50=float(np.mean(np.abs(np.diff(clean)) > 50) * 100))
    else:
        out.update(hrv_sdnn=np.nan, hrv_rmssd=np.nan, hrv_pnn50=np.nan)
    return out


def acc_features(mag, fs):
    mag = np.asarray(mag, float)
    if len(mag) < 4:
        return dict.fromkeys(["acc_std","acc_energy","acc_mad","acc_max"], np.nan)
    return dict(acc_std=mag.std(ddof=1), acc_energy=float(np.mean((mag - mag.mean()) ** 2)),
                acc_mad=float(np.abs(np.diff(mag)).mean()), acc_max=mag.max())


def window_label(data, t0, t1, purity=PURITY):
    lab = np.asarray(data["label"]).ravel()
    seg = lab[int(t0 * FS["label"]):int(t1 * FS["label"])]
    if len(seg) == 0:
        return None
    affect = seg[np.isin(seg, list(MODEL_STATES))]
    if len(affect) == 0:
        return None
    vals, counts = np.unique(affect, return_counts=True)
    top = int(vals[counts.argmax()])
    return top if counts.max() / len(seg) >= purity else None


def extract_subject(subj, window_sec=WINDOW_SEC, shift_sec=SHIFT_SEC):
    d = load_subject(subj)
    dur = len(np.asarray(d["label"])) / FS["label"]
    chans = {ch: get_channel(d, "wrist", ch) for ch in ["EDA", "TEMP", "BVP", "ACC"]}
    rows, t0 = [], 0.0
    while t0 + window_sec <= dur:
        t1 = t0 + window_sec
        lab = window_label(d, t0, t1)
        if lab is not None:
            rec = dict(subject=subj, condition=lab, condition_name=MODEL_STATES[lab], t_start=round(t0, 3))
            for ch, (sig, fs) in chans.items():
                seg = sig[int(t0 * fs):int(t1 * fs)]
                rec.update({"EDA": eda_features, "TEMP": temp_features,
                            "BVP": bvp_features, "ACC": acc_features}[ch](seg, fs))
            rows.append(rec)
        t0 += shift_sec
    return rows


def build_feature_matrix(subjects=SUBJECTS, **kw):
    """Extract all subjects, drop windows missing CORE features (HRV NaNs preserved), return DataFrame."""
    feat = pd.DataFrame([r for s in subjects for r in extract_subject(s, **kw)])
    core = [c for c in feat.columns
            if c not in {"subject", "condition", "condition_name", "t_start"}
            and c not in UNTRUSTED_FEATURES[1:]]  # hrv_* may be NaN by design
    return feat.dropna(subset=core).reset_index(drop=True)
