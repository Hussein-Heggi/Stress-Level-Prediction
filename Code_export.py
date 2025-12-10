#!/usr/bin/env python
# coding: utf-8

# # Stress Level Prediction Pipeline
# 
# End-to-end preprocessing and modeling for stress vs non-stress, including aerobic/anaerobic as active negatives.
# 

# In[1]:


from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

np.random.seed(42)


# ## Config
# - Window length 60s, step 30s
# - Minimum label coverage 0.7
# - Protocol tag mapping per provided notebook
# - Data constraints applied during load

# In[2]:


# Paths
DEFAULT_DATASET_ROOT = Path("./Datasets")
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", DEFAULT_DATASET_ROOT))

# Windowing
WINDOW_SECONDS = 60
WINDOW_STEP_SECONDS = 30
MIN_LABEL_COVERAGE = 0.6

# Sampling / weighting
PHASE_BALANCE_NO_STRESS_RATIO = 1.0
PHASE_WEIGHT_MAP = {
    "Stroop": 1.6,
    "Opposite Opinion": 1.4,
    "Real Opinion": 1.3,
    "Subtract": 1.2,
    "TMCT": 1.4,
}
PHASE_LABEL_OVERSAMPLE = {
    ("Stroop", "high_stress"): 3.5,
    ("Stroop", "moderate_stress"): 2.2,
    ("TMCT", "high_stress"): 2.8,
    ("TMCT", "moderate_stress"): 1.8,
    ("Opposite Opinion", "low_stress"): 2.5,
    ("Opposite Opinion", "moderate_stress"): 1.8,
    ("Real Opinion", "moderate_stress"): 1.6,
}
EMA_SPAN = 5
PHASE_Z_MIN_STD = 1e-3
STRESS_PROB_THRESHOLD = 0.4
PHASE_SMOOTHING_WINDOW = 3

# Data constraints
DUPLICATE_CUTS = {"S02": {"ACC": 49545, "BVP": 99091, "EDA": 6195, "TEMP": 6195}}
MISSING_SENSORS = {"f07": {"BVP", "TEMP", "HR", "IBI"}}

STATES = ["STRESS", "AEROBIC", "ANAEROBIC"]

# Stress protocol metadata
STRESS_STAGE_ORDER_S = ["Stroop", "TMCT", "Real Opinion", "Opposite Opinion", "Subtract"]
STRESS_STAGE_ORDER_F = ["TMCT", "Real Opinion", "Opposite Opinion", "Subtract"]
STRESS_TAG_PAIRS_S = [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
STRESS_TAG_PAIRS_F = [(2, 3), (4, 5), (6, 7), (8, 9)]
STRESS_LEVEL_FILES = ["Stress_Level_v1.csv", "Stress_Level_v2.csv"]

def load_stress_levels() -> Dict[str, Dict[str, float]]:
    tables = []
    for fname in STRESS_LEVEL_FILES:
        path = Path(fname)
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        df.columns = [str(c).strip() for c in df.columns]
        tables.append(df)
    levels: Dict[str, Dict[str, float]] = {}
    for table in tables:
        for subject, row in table.iterrows():
            subj = str(subject).strip()
            levels[subj] = {col: (float(row[col]) if not pd.isna(row[col]) else np.nan) for col in table.columns}
    return levels


STRESS_LEVELS = load_stress_levels()


STRESS_PHASES = {"Stroop", "TMCT", "Real Opinion", "Opposite Opinion", "Subtract"}
STRESS_LEVEL_BOUNDS = {"low": 3.0, "moderate": 6.0}
STRESS_LEVEL_PHASE_BOUNDS = {
    "Stroop": {"low": 2.5, "moderate": 5.0},
    "Opposite Opinion": {"low": 2.5, "moderate": 5.5},
    "Real Opinion": {"low": 2.8, "moderate": 5.5},
    "TMCT": {"low": 2.8, "moderate": 5.8},
    "Subtract": {"low": 2.8, "moderate": 5.8},
}
TEMPORAL_FEATURES = [
    "eda_mean",
    "eda_std",
    "eda_range",
    "eda_slope",
    "acc_energy",
    "temp_mean",
    "temp_range",
    "hr_mean",
    "bvp_mean",
    "bvp_std",
    "bvp_hr_est",
]
FEATURE_EXCLUDE_COLS = {
    "subject",
    "state",
    "phase",
    "stress_stage",
    "stress_level",
    "label",
    "is_stress",
    "win_start",
    "win_end",
    "phase_start",
    "phase_end",
}

ACC_ACTIVITY_WINDOW_SEC = 2.0
ACC_ACTIVITY_STEP_SEC = 0.5


# ## Helpers for Empatica format and tags

# In[3]:


# Get base subject ID without session suffix.
def base_subject_id(subject: str) -> str:
    return subject.split("_")[0]


# Read signal CSV and return (fs, data, start_timestamp).
def read_signal(path: Path) -> Tuple[float, np.ndarray, pd.Timestamp]:
    with open(path, "r") as f:
        start_line = f.readline()
        if not start_line:
            raise ValueError(f"Missing start timestamp in {path}")
        start = pd.to_datetime(start_line.split(",")[0])
        fs_line = f.readline()
        if not fs_line:
            raise ValueError(f"Missing sample rate in {path}")
        fs = float(fs_line.split(",")[0])
        data = np.genfromtxt(f, delimiter=",")
    if np.isscalar(data):
        if math.isnan(float(data)):
            data = np.empty((0, 1))
        else:
            data = np.array([[float(data)]])
    elif data.size == 0:
        data = np.empty((0, 1))
    else:
        data = np.asarray(data, dtype=float)
        if np.isnan(data).all():
            data = np.empty((0, 1))
        elif data.ndim == 1:
            data = data[:, None]
    return fs, data, start


# Read IBI CSV and return timestamped inter-beat intervals.
def read_ibi(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 2), dtype=float)
    try:
        df = pd.read_csv(path, header=None, skiprows=1)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return np.empty((0, 2), dtype=float)
    if df.empty:
        return np.empty((0, 2), dtype=float)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        return np.empty((0, 2), dtype=float)
    if df.shape[1] >= 2:
        times = df.iloc[:, 0].to_numpy(dtype=float)
        ibi = df.iloc[:, 1].to_numpy(dtype=float)
    else:
        ibi = df.iloc[:, 0].to_numpy(dtype=float)
        times = np.cumsum(ibi)
    if times.size:
        times = np.clip(times, a_min=0.0, a_max=None)
    mask = (ibi > 0) & np.isfinite(times) & np.isfinite(ibi)
    if not np.any(mask):
        return np.empty((0, 2), dtype=float)
    return np.column_stack([times[mask], ibi[mask]])


# Read tags.csv and return list of tag timestamps (seconds since start).
def read_tags(path: Path, start_ts: pd.Timestamp) -> List[Tuple[float, float]]:
    try:
        df = pd.read_csv(path, header=None)
    except pd.errors.EmptyDataError:
        return []
    tags = []
    for ts_str in df[0].astype(str):
        ts = pd.to_datetime(ts_str)
        tags.append((ts - start_ts).total_seconds())
    return [(t, t) for t in tags]


# Extract stress intervals from tags based on subject version.
def stress_intervals_from_tags(tags: List[Tuple[float, float]], subject: str) -> List[dict]:
    if not tags:
        return []
    t = [x[0] for x in tags]
    if subject.startswith("S"):
        idx_pairs = STRESS_TAG_PAIRS_S
        stage_order = STRESS_STAGE_ORDER_S
    else:
        idx_pairs = STRESS_TAG_PAIRS_F
        stage_order = STRESS_STAGE_ORDER_F
    base_id = base_subject_id(subject)
    spans = []
    for stage, (i, j) in zip(stage_order, idx_pairs):
        if i < len(t) and j < len(t) and t[j] > t[i]:
            level = STRESS_LEVELS.get(base_id, {}).get(stage)
            spans.append({
                "start": t[i],
                "end": t[j],
                "stage": stage,
                "stress_level": level,
            })
    return spans


# Extract active intervals from tags.
def active_intervals_from_tags(tags: List[Tuple[float, float]], label: str) -> List[dict]:
    if len(tags) < 2:
        return []
    t = [x[0] for x in tags]
    spans = []
    for a, b in zip(t[:-1], t[1:]):
        if b > a:
            spans.append({
                "start": a,
                "end": b,
                "stage": label,
                "stress_level": 0.0,
            })
    return spans


# Map numeric stress score to four-level category.
def stress_bucket(level: float | None, phase: str | None) -> str:
    if phase in {"aerobic", "anaerobic", "rest"}:
        return "no_stress"
    if level is None or pd.isna(level) or level <= 0:
        return "no_stress"
    bounds = STRESS_LEVEL_PHASE_BOUNDS.get(phase, STRESS_LEVEL_BOUNDS)
    if level <= bounds["low"]:
        return "low_stress"
    if level <= bounds["moderate"]:
        return "moderate_stress"
    return "high_stress"


# ## Resampling and preprocessing

# In[4]:


# Resample signal to target frequency.
def resample_to_rate(signal: np.ndarray, src_fs: float, tgt_fs: float) -> np.ndarray:
    if signal.ndim == 1:
        signal = signal[:, None]
    src_len = signal.shape[0]
    duration = src_len / src_fs
    tgt_len = int(duration * tgt_fs)
    src_t = np.linspace(0, duration, src_len)
    tgt_t = np.linspace(0, duration, tgt_len)
    resampled = np.stack([np.interp(tgt_t, src_t, signal[:, i]) for i in range(signal.shape[1])], axis=1)
    if resampled.shape[1] == 1:
        return resampled[:, 0]
    return resampled

# Simple Hampel filter for outlier suppression.
def hampel_filter(x: np.ndarray, k: int = 5, t0: float = 3.0) -> np.ndarray:
    x = x.copy()
    n = len(x)
    for i in range(n):
        window = x[max(i - k, 0): min(i + k, n)]
        med = np.median(window)
        mad = np.median(np.abs(window - med)) or 1e-6
        if abs(x[i] - med) > t0 * mad:
            x[i] = med
    return x


# In[5]:


# Approximate ACC activity envelope using sliding RMS to capture movement intensity.
def acc_activity_signal(
    acc_raw: np.ndarray,
    fs: float,
    win_sec: float = ACC_ACTIVITY_WINDOW_SEC,
    step_sec: float = ACC_ACTIVITY_STEP_SEC,
) -> np.ndarray:
    if acc_raw.size == 0 or fs <= 0:
        return np.array([])
    if acc_raw.ndim == 1:
        acc_raw = acc_raw[:, None]
    magnitude = np.linalg.norm(acc_raw, axis=1)
    win = max(1, int(round(win_sec * fs)))
    step = max(1, int(round(step_sec * fs)))
    if len(magnitude) < win:
        rms = float(np.sqrt(np.mean(magnitude ** 2))) if len(magnitude) else 0.0
        return np.array([rms])
    activity = []
    start = 0
    while start + win <= len(magnitude):
        segment = magnitude[start:start + win]
        rms = float(np.sqrt(np.mean(segment ** 2)))
        activity.append(rms)
        start += step
    if start < len(magnitude):
        segment = magnitude[-win:]
        rms = float(np.sqrt(np.mean(segment ** 2)))
        if not activity or abs(activity[-1] - rms) > 1e-9:
            activity.append(rms)
    return np.asarray(activity)


# In[6]:


def spectral_centroid(signal: np.ndarray, fs: float) -> float:
    if signal.size == 0 or fs <= 0:
        return np.nan
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    total = psd.sum()
    if total <= 0:
        return np.nan
    return float(np.sum(freqs * psd) / total)


def bandpower(signal: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    if signal.size == 0 or fs <= 0 or fmax <= fmin:
        return np.nan
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan
    trap = getattr(np, "trapezoid", np.trapz)
    return float(trap(psd[mask], freqs[mask]))


def dominant_frequency(signal: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    if signal.size == 0 or fs <= 0 or fmax <= fmin:
        return np.nan
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan
    idx = np.argmax(psd[mask])
    return float(freqs[mask][idx])


def linear_slope(signal: np.ndarray, fs: float) -> float:
    if signal.size < 2 or fs <= 0:
        return 0.0
    t = np.arange(signal.size) / fs
    slope, _ = np.polyfit(t, signal, 1)
    return float(slope)


def peak_rate(signal: np.ndarray, fs: float) -> float:
    if signal.size < 3 or fs <= 0:
        return 0.0
    duration = signal.size / fs
    if duration <= 0:
        return 0.0
    peaks = np.sum((signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:]))
    return float(peaks / duration)


def eda_features(eda: np.ndarray, fs: float) -> Dict[str, float]:
    if eda.size == 0 or fs <= 0:
        return {
            "eda_mean": np.nan,
            "eda_std": np.nan,
            "eda_slope": np.nan,
            "eda_peak_rate": np.nan,
            "eda_range": np.nan,
            "eda_power_slow": np.nan,
            "eda_power_mid": np.nan,
            "eda_power_fast": np.nan,
            "eda_centroid": np.nan,
            "eda_min": np.nan,
            "eda_max": np.nan,
            "eda_skew": np.nan,
            "eda_kurt": np.nan,
            "eda_diff_mean": np.nan,
            "eda_diff_std": np.nan,
        }
    clean = hampel_filter(eda)
    diff = np.diff(clean) if clean.size > 1 else np.array([0.0])
    return {
        "eda_mean": float(np.mean(clean)),
        "eda_std": float(np.std(clean)),
        "eda_slope": linear_slope(clean, fs),
        "eda_peak_rate": peak_rate(clean, fs),
        "eda_range": float(np.max(clean) - np.min(clean)),
        "eda_power_slow": bandpower(clean, fs, 0.01, 0.05),
        "eda_power_mid": bandpower(clean, fs, 0.045, 0.25),
        "eda_power_fast": bandpower(clean, fs, 0.25, 1.5),
        "eda_centroid": spectral_centroid(clean, fs),
        "eda_min": float(np.min(clean)),
        "eda_max": float(np.max(clean)),
        "eda_skew": float(skew(clean)),
        "eda_kurt": float(kurtosis(clean)),
        "eda_diff_mean": float(np.mean(diff)) if diff.size else 0.0,
        "eda_diff_std": float(np.std(diff)) if diff.size else 0.0,
    }


def temp_features(temp: np.ndarray, fs: float) -> Dict[str, float]:
    if temp.size == 0 or fs <= 0:
        return {
            "temp_mean": np.nan,
            "temp_std": np.nan,
            "temp_slope": np.nan,
            "temp_min": np.nan,
            "temp_max": np.nan,
            "temp_range": np.nan,
        }
    return {
        "temp_mean": float(np.mean(temp)),
        "temp_std": float(np.std(temp)),
        "temp_slope": linear_slope(temp, fs),
        "temp_min": float(np.min(temp)),
        "temp_max": float(np.max(temp)),
        "temp_range": float(np.max(temp) - np.min(temp)),
    }


def hrv_features(ibi: np.ndarray) -> Dict[str, float]:
    if ibi.size < 2:
        return {
            "hr_mean": np.nan,
            "rmssd": np.nan,
            "sdnn": np.nan,
            "pnn50": np.nan,
            "lf_hf": np.nan,
            "sd1": np.nan,
            "sd2": np.nan,
        }
    rr = ibi.astype(float)
    diff = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diff ** 2))) if diff.size else np.nan
    sdnn = float(np.std(rr))
    pnn50 = float(np.mean(np.abs(diff) > 0.05)) if diff.size else np.nan
    hr_mean = float(60.0 / np.mean(rr)) if np.mean(rr) > 0 else np.nan

    if diff.size:
        sd1 = float(np.sqrt(0.5) * np.std(diff))
    else:
        sd1 = np.nan
    if not np.isnan(sdnn) and not np.isnan(sd1):
        sd2_sq = max(0.0, 2 * (sdnn ** 2) - 0.5 * (sd1 ** 2))
        sd2 = float(np.sqrt(sd2_sq))
    else:
        sd2 = np.nan

    lf_hf = np.nan
    try:
        t = np.cumsum(rr)
        t = t - t[0]
        if t[-1] > 0 and rr.size >= 4:
            fs_interp = 4.0
            grid = np.arange(0, t[-1], 1 / fs_interp)
            if grid.size >= 8:
                interp_rr = np.interp(grid, t[: len(grid)], rr[: len(grid)])
                lf = bandpower(interp_rr, fs_interp, 0.04, 0.15)
                hf = bandpower(interp_rr, fs_interp, 0.15, 0.4)
                if hf and hf > 0:
                    lf_hf = float(lf / hf)
    except Exception:
        lf_hf = np.nan

    return {
        "hr_mean": hr_mean,
        "rmssd": rmssd,
        "sdnn": sdnn,
        "pnn50": pnn50,
        "lf_hf": lf_hf,
        "sd1": sd1,
        "sd2": sd2,
    }


# ## Feature extraction

# In[7]:


def acc_features(acc_mag: np.ndarray, fs: float, acc_activity: np.ndarray | None = None) -> Dict[str, float]:
    if len(acc_mag) == 0:
        base = {
            "acc_mean": np.nan,
            "acc_std": np.nan,
            "acc_energy": np.nan,
            "acc_peak_freq": np.nan,
            "acc_bandpower_low": np.nan,
            "acc_bandpower_mid": np.nan,
            "acc_bandpower_high": np.nan,
            "acc_mad": np.nan,
        }
    else:
        energy = np.mean(acc_mag ** 2)
        base = {
            "acc_mean": float(np.mean(acc_mag)),
            "acc_std": float(np.std(acc_mag)),
            "acc_energy": float(energy),
            "acc_peak_freq": spectral_centroid(acc_mag, fs),
            "acc_bandpower_low": bandpower(acc_mag, fs, 0.1, 0.5),
            "acc_bandpower_mid": bandpower(acc_mag, fs, 0.5, 2.0),
            "acc_bandpower_high": bandpower(acc_mag, fs, 2.0, 8.0),
            "acc_mad": float(np.median(np.abs(acc_mag - np.median(acc_mag)))),
        }
    if acc_activity is not None and len(acc_activity):
        base.update(
            {
                "acc_activity_mean": float(np.mean(acc_activity)),
                "acc_activity_std": float(np.std(acc_activity)),
                "acc_activity_max": float(np.max(acc_activity)),
            }
        )
    else:
        base.update(
            {
                "acc_activity_mean": np.nan,
                "acc_activity_std": np.nan,
                "acc_activity_max": np.nan,
            }
        )
    return base


def bvp_features(bvp: np.ndarray | None, fs: float | None) -> Dict[str, float]:
    if bvp is None or len(bvp) == 0 or fs is None or fs <= 0:
        return {
            "bvp_mean": np.nan,
            "bvp_std": np.nan,
            "bvp_slope": np.nan,
            "bvp_range": np.nan,
            "bvp_peak_rate": np.nan,
            "bvp_hr_est": np.nan,
            "bvp_power_low": np.nan,
            "bvp_power_high": np.nan,
            "bvp_centroid": np.nan,
            "bvp_power_ratio": np.nan,
            "bvp_resp_freq": np.nan,
        }
    clean = hampel_filter(bvp)
    peak_per_sec = peak_rate(clean, fs)
    hr_est = peak_per_sec * 60.0 if peak_per_sec > 0 else np.nan
    low_power = bandpower(clean, fs, 0.04, 0.15)
    high_power = bandpower(clean, fs, 0.15, 0.4)
    ratio = float(high_power / (low_power + 1e-6)) if not np.isnan(high_power) else np.nan
    resp_freq = dominant_frequency(clean, fs, 0.1, 0.5)
    return {
        "bvp_mean": float(np.mean(clean)),
        "bvp_std": float(np.std(clean)),
        "bvp_slope": linear_slope(clean, fs),
        "bvp_range": float(np.max(clean) - np.min(clean)),
        "bvp_peak_rate": float(peak_per_sec),
        "bvp_hr_est": float(hr_est),
        "bvp_power_low": low_power,
        "bvp_power_high": high_power,
        "bvp_centroid": spectral_centroid(clean, fs),
        "bvp_power_ratio": ratio,
        "bvp_resp_freq": float(resp_freq),
    }


def combine_features(eda, eda_fs, acc_mag, acc_fs, temp, temp_fs, ibi, acc_activity=None, bvp=None, bvp_fs=None) -> Dict[str, float]:
    feats = {}
    feats.update(eda_features(eda, eda_fs))
    feats.update(acc_features(acc_mag, acc_fs, acc_activity))
    feats.update(temp_features(temp, temp_fs))
    feats.update(hrv_features(ibi))
    feats.update(bvp_features(bvp, bvp_fs))
    return feats


# ## Windowing and label assignment

# In[8]:


@dataclass
class Window:
    start: float
    end: float
    label: str
    subject: str
    state: str


def window_intervals(duration: float, win_s: int = WINDOW_SECONDS, step_s: int = WINDOW_STEP_SECONDS) -> List[Tuple[float, float]]:
    windows = []
    t = 0.0
    while t + win_s <= duration:
        windows.append((t, t + win_s))
        t += step_s
    return windows


def _span_bounds(span) -> Tuple[float, float]:
    if isinstance(span, dict):
        return span["start"], span["end"]
    return span


def assign_label(win: Tuple[float, float], intervals: Dict[str, List[dict]]) -> Tuple[str | None, dict | None]:
    start, end = win
    length = end - start
    best_label = None
    best_cov = 0.0
    best_span = None
    for lbl, spans in intervals.items():
        label_overlap = 0.0
        label_best_span = None
        label_best_overlap = 0.0
        for span in spans:
            a, b = _span_bounds(span)
            inter = max(0.0, min(end, b) - max(start, a))
            if inter > 0:
                label_overlap += inter
                if inter > label_best_overlap:
                    label_best_overlap = inter
                    label_best_span = span
        coverage = label_overlap / length
        if coverage > best_cov:
            best_cov = coverage
            best_label = lbl
            best_span = label_best_span
    if best_cov >= MIN_LABEL_COVERAGE and best_label is not None:
        return best_label, best_span
    return None, None


def make_label_intervals(state: str, subject: str, tags: List[Tuple[float, float]], duration: float) -> Dict[str, List[dict]]:
    rest_span = {"start": 0.0, "end": duration, "stage": "rest", "stress_level": 0.0}
    if state == "STRESS":
        stress_spans = stress_intervals_from_tags(tags, subject)
        if not stress_spans:
            return {"rest": [rest_span]}
        return {
            "stress": stress_spans,
            "rest": [rest_span],
        }
    else:
        lbl = "aerobic" if state == "AEROBIC" else "anaerobic"
        active = active_intervals_from_tags(tags, lbl)
        return {
            lbl: active,
            "rest": [rest_span],
        }


# ## Data ingestion: read signals per subject/state

# In[9]:


def load_subject_state(state: str, subject: str) -> dict:
    folder = DATASET_ROOT / state / subject
    base_id = base_subject_id(subject)
    sensors = {}
    fs_map = {}
    missing = MISSING_SENSORS.get(base_id, set())

    if not folder.exists():
        raise FileNotFoundError(folder)
    fs_eda, eda_raw, start_ts = read_signal(folder / "EDA.csv")
    sensors["EDA"] = np.squeeze(eda_raw)
    fs_map["EDA"] = fs_eda

    temp_path = folder / "TEMP.csv"
    if "TEMP" not in missing and temp_path.exists():
        fs_temp, temp_raw, _ = read_signal(temp_path)
        sensors["TEMP"] = np.squeeze(temp_raw)
        fs_map["TEMP"] = fs_temp

    fs_acc, acc_raw, _ = read_signal(folder / "ACC.csv")
    acc_mag = np.linalg.norm(acc_raw, axis=1)
    sensors["ACC_MAG"] = acc_mag
    fs_map["ACC_MAG"] = fs_acc
    acc_activity = acc_activity_signal(acc_raw, fs_acc)

    if acc_activity.size:
        sensors["ACC_ACTIVITY"] = acc_activity
        fs_map["ACC_ACTIVITY"] = 1.0 / ACC_ACTIVITY_STEP_SEC

    bvp_path = folder / "BVP.csv"
    if "BVP" not in missing and bvp_path.exists():
        fs_bvp, bvp_raw, _ = read_signal(bvp_path)
        sensors["BVP"] = np.squeeze(bvp_raw)
        fs_map["BVP"] = fs_bvp

    if "IBI" not in missing:
        sensors["IBI"] = read_ibi(folder / "IBI.csv")
    else:
        sensors["IBI"] = np.empty((0, 2))
    tags = read_tags(folder / "tags.csv", start_ts)

    if base_id in DUPLICATE_CUTS:
        cuts = DUPLICATE_CUTS[base_id]
        if "EDA" in cuts and "EDA" in sensors:
            sensors["EDA"] = sensors["EDA"][:cuts["EDA"]]
        if "TEMP" in cuts and "TEMP" in sensors:
            sensors["TEMP"] = sensors["TEMP"][:cuts["TEMP"]]
        if "ACC" in cuts and "ACC_MAG" in sensors:
            sensors["ACC_MAG"] = sensors["ACC_MAG"][:cuts["ACC"]]
        if "BVP" in cuts and "BVP" in sensors:
            sensors["BVP"] = sensors["BVP"][:cuts["BVP"]]

    duration = len(sensors["EDA"]) / fs_map["EDA"]
    return {
        "sensors": sensors,
        "fs": fs_map,
        "tags": tags,
        "duration": duration,
    }


# ## Build windowed dataset

# In[10]:


def build_windows_for_subject(state: str, subject: str, tgt_fs: float = 4.0) -> List[dict]:
    info = load_subject_state(state, subject)
    sensors = info["sensors"]
    fs_map = info["fs"]
    tags = info["tags"]
    duration = info["duration"]

    # Resample signals
    eda = resample_to_rate(sensors["EDA"], fs_map["EDA"], tgt_fs)
    temp = resample_to_rate(sensors["TEMP"], fs_map.get("TEMP", tgt_fs), tgt_fs) if "TEMP" in sensors else np.array([])
    acc = resample_to_rate(sensors["ACC_MAG"], fs_map["ACC_MAG"], tgt_fs)
    acc_activity = resample_to_rate(sensors["ACC_ACTIVITY"], fs_map.get("ACC_ACTIVITY", tgt_fs), tgt_fs) if "ACC_ACTIVITY" in sensors else np.array([])
    bvp_signal = sensors.get("BVP")
    bvp_fs = fs_map.get("BVP")

    intervals = make_label_intervals(state, subject, tags, duration)
    windows = window_intervals(duration, WINDOW_SECONDS, WINDOW_STEP_SECONDS)
    rows = []
    for w in windows:
        lbl, span_meta = assign_label(w, intervals)
        if lbl is None or span_meta is None:
            continue
        start_idx = int(w[0] * tgt_fs)
        end_idx = int(w[1] * tgt_fs)
        eda_win = eda[start_idx:end_idx]
        temp_win = temp[start_idx:end_idx] if len(temp) else np.array([])
        acc_win = acc[start_idx:end_idx]
        activity_win = acc_activity[start_idx:end_idx] if len(acc_activity) else np.array([])
        if sensors["IBI"].size:
            ibi_arr = sensors["IBI"]
            mask = (ibi_arr[:, 0] >= w[0]) & (ibi_arr[:, 0] <= w[1])
            ibi_win = ibi_arr[mask, 1]
        else:
            ibi_win = np.array([])
        if bvp_signal is not None and bvp_fs:
            bvp_start = int(max(0, math.floor(w[0] * bvp_fs)))
            bvp_end = int(min(len(bvp_signal), math.floor(w[1] * bvp_fs)))
            bvp_win = bvp_signal[bvp_start:bvp_end]
        else:
            bvp_win = None

        feats = combine_features(
            eda_win,
            tgt_fs,
            acc_win,
            tgt_fs,
            temp_win,
            tgt_fs,
            ibi_win,
            activity_win,
            bvp_win,
            bvp_fs,
        )
        stress_stage = span_meta.get("stage") if isinstance(span_meta, dict) else None
        stress_level = span_meta.get("stress_level") if isinstance(span_meta, dict) else None
        phase_start = span_meta.get("start") if isinstance(span_meta, dict) else None
        phase_end = span_meta.get("end") if isinstance(span_meta, dict) else None
        if lbl == "stress":
            if stress_level is None or np.isnan(stress_level):
                continue
        else:
            stress_level = 0.0 if stress_level is None or np.isnan(stress_level) else stress_level
        phase_label = stress_stage if stress_stage else lbl
        stress_class = stress_bucket(stress_level, phase_label)
        phase_duration = None
        phase_progress = None
        phase_elapsed = None
        if phase_start is not None and phase_end is not None and phase_end > phase_start:
            phase_duration = float(phase_end - phase_start)
            phase_elapsed = float(w[0] - phase_start)
            phase_progress = max(0.0, min(1.0, phase_elapsed / phase_duration))
        row = {
            "subject": base_subject_id(subject),
            "state": state.lower(),
            "phase": phase_label,
            "stress_stage": stress_stage if lbl == "stress" else None,
            "stress_level": float(stress_level),
            "label": stress_class,
            "is_stress": 1 if phase_label in STRESS_PHASES else 0,
            "win_start": w[0],
            "win_end": w[1],
            "phase_start": phase_start,
            "phase_end": phase_end,
            "phase_duration": phase_duration,
            "phase_elapsed": phase_elapsed,
            "phase_progress": phase_progress,
        }
        row.update(feats)
        rows.append(row)
    return rows


# In[11]:


def build_dataset(states: List[str] = STATES, max_subjects: int | None = None) -> pd.DataFrame:
    rows = []
    for state in states:
        state_dir = DATASET_ROOT / state
        if not state_dir.exists():
            continue
        subjects = sorted([p.name for p in state_dir.iterdir() if p.is_dir()])
        if max_subjects:
            subjects = subjects[:max_subjects]
        for subj in subjects:
            try:
                rows.extend(build_windows_for_subject(state, subj))
            except Exception as exc:
                print(f"Skip {state}/{subj}: {exc}")
                continue
    return pd.DataFrame(rows)


# ## Build Dataset

# In[12]:


df = build_dataset()
df = df.sort_values(["subject", "win_start"]).reset_index(drop=True)

phase_group = df.groupby(["subject", "phase"], dropna=False)
phase_min_start = phase_group["win_start"].transform("min")
phase_max_end = phase_group["win_end"].transform("max")
phase_duration_fallback = (phase_max_end - phase_min_start).replace(0, np.nan)
df["phase_elapsed"] = df["phase_elapsed"].fillna(df["win_start"] - phase_min_start)
df["phase_duration"] = df["phase_duration"].fillna(phase_duration_fallback)
df["phase_progress"] = df["phase_progress"].fillna(
    df["phase_elapsed"] / df["phase_duration"].replace(0, np.nan)
)
df["phase_progress"] = df["phase_progress"].clip(0.0, 1.0).fillna(0.0)
df["phase_position"] = phase_group.cumcount()
phase_counts = phase_group.size().rename("phase_size")
df = df.join(phase_counts, on=["subject", "phase"])
df["phase_position"] = np.where(
    df["phase_size"] > 1,
    df["phase_position"] / (df["phase_size"] - 1),
    0.0,
)
df.drop(columns=["phase_size"], inplace=True)
df["phase_remaining"] = (df["phase_duration"] - df["phase_elapsed"]).clip(lower=0).fillna(0)
df["phase_early"] = (df["phase_progress"] <= 0.33).astype(int)
df["phase_mid"] = ((df["phase_progress"] > 0.33) & (df["phase_progress"] <= 0.66)).astype(int)
df["phase_late"] = (df["phase_progress"] > 0.66).astype(int)

context_cols = []
for feat in TEMPORAL_FEATURES:
    prev_col = f"prev_{feat}"
    delta_col = f"delta_{feat}"
    roll_mean_col = f"roll_mean_{feat}"
    roll_std_col = f"roll_std_{feat}"
    ema_col = f"ema_{feat}"
    phase_delta_col = f"phase_delta_{feat}"
    phase_z_col = f"phase_z_{feat}"
    subj_group = df.groupby("subject")[feat]
    df[prev_col] = subj_group.shift(1)
    df[delta_col] = df[feat] - df[prev_col]
    roll_mean = (
        subj_group.rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    roll_std = (
        subj_group.rolling(window=3, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )
    df[roll_mean_col] = roll_mean.to_numpy()
    df[roll_std_col] = roll_std.to_numpy()
    df[ema_col] = df.groupby("subject")[feat].transform(lambda s: s.ewm(span=EMA_SPAN, adjust=False).mean())
    phase_first = (
        df.sort_values("win_start")
        .groupby(["subject", "phase"], dropna=False)[feat]
        .transform("first")
    )
    df[phase_delta_col] = df[feat] - phase_first
    phase_running_mean = phase_group[feat].transform(lambda s: s.expanding().mean())
    phase_running_std = phase_group[feat].transform(lambda s: s.expanding().std()).replace(0, np.nan)
    df[phase_z_col] = (df[feat] - phase_running_mean) / phase_running_std
    df[phase_z_col] = df[phase_z_col].replace([np.inf, -np.inf], 0)
    context_cols.extend([prev_col, delta_col, roll_mean_col, roll_std_col, ema_col, phase_delta_col, phase_z_col])

df[context_cols] = df[context_cols].fillna(0)

df.to_csv("stress_level_dataset.csv", index=False)
print("Rows", len(df))
print("Phase counts", df["phase"].value_counts())
print("Label distribution", df["label"].value_counts())
print("Stress level summary", df["stress_level"].describe())


# In[13]:


def balance_by_phase_label(data: pd.DataFrame) -> pd.DataFrame:
    balanced = []
    for phase_name, phase_df in data.groupby("phase"):
        if phase_df.empty:
            continue
        stress_df = phase_df[phase_df["label"] != "no_stress"]
        phase_parts = []
        base_target = stress_df["label"].value_counts().max() if not stress_df.empty else 0
        phase_factor = PHASE_WEIGHT_MAP.get(phase_name, 1.0)
        target = max(1, int(np.ceil(base_target * phase_factor))) if base_target else 0
        if not stress_df.empty:
            for label_name, label_df in stress_df.groupby("label"):
                label_factor = PHASE_LABEL_OVERSAMPLE.get((phase_name, label_name), 1.0)
                label_target = max(1, int(np.ceil(target * label_factor))) if target else len(label_df)
                if len(label_df) < label_target:
                    phase_parts.append(label_df.sample(label_target, replace=True, random_state=42))
                else:
                    phase_parts.append(label_df)
        no_stress_df = phase_df[phase_df["label"] == "no_stress"]
        if not no_stress_df.empty:
            cap = int(np.ceil(PHASE_BALANCE_NO_STRESS_RATIO * max(target, 1))) if target else len(no_stress_df)
            if len(no_stress_df) > cap:
                phase_parts.append(no_stress_df.sample(cap, random_state=42, replace=False))
            else:
                phase_parts.append(no_stress_df)
        if phase_parts:
            balanced.append(pd.concat(phase_parts, ignore_index=True))
        else:
            balanced.append(phase_df)
    return pd.concat(balanced, ignore_index=True)


# In[14]:


def apply_phase_smoothing(meta: pd.DataFrame, predictions: np.ndarray, window: int = PHASE_SMOOTHING_WINDOW) -> np.ndarray:
    if window <= 1 or len(predictions) == 0:
        return predictions
    smoothed = predictions.copy()
    half = window // 2
    meta = meta.reset_index(drop=True)
    for (_, phase_df) in meta.groupby(["subject", "phase"], dropna=False):
        if phase_df.empty:
            continue
        order = phase_df.sort_values("win_start").index.to_numpy()
        group_vals = smoothed[order].copy()
        for i in range(len(group_vals)):
            start = max(0, i - half)
            end = min(len(group_vals), i + half + 1)
            window_vals = group_vals[start:end]
            if window_vals.size == 0:
                continue
            uniq, counts = np.unique(window_vals, return_counts=True)
            group_vals[i] = uniq[np.argmax(counts)]
        smoothed[order] = group_vals
    return smoothed


# In[15]:


# Prepare features for modeling
numeric_cols = [
    c for c in df.columns if c not in FEATURE_EXCLUDE_COLS and np.issubdtype(df[c].dtype, np.number)
]
helper_cols = ["label", "subject", "phase", "win_start", "stress_level"]
feature_df = df.loc[:, numeric_cols + helper_cols].copy()
feature_df = feature_df.dropna(subset=["label"]).reset_index(drop=True)
feature_df["phase"] = feature_df["phase"].fillna("unknown")

phase_sorted = feature_df.sort_values(["subject", "phase", "win_start"])
phase_baselines = (
    phase_sorted.groupby(["subject", "phase"], dropna=False)[numeric_cols]
    .transform("first")
)
feature_df.loc[:, numeric_cols] = feature_df[numeric_cols] - phase_baselines.fillna(0)
feature_df.loc[:, numeric_cols] = feature_df[numeric_cols].fillna(0)

non_nan_cols = feature_df[numeric_cols].columns[~feature_df[numeric_cols].isna().all()]
numeric_cols = non_nan_cols.tolist()
feature_df.loc[:, numeric_cols] = feature_df[numeric_cols].fillna(0)
helper_df = feature_df.loc[:, numeric_cols + helper_cols].copy().reset_index(drop=True)
helper_df["label"] = helper_df["label"].astype(str)
helper_df["phase"] = helper_df["phase"].fillna("unknown")
helper_df["stress_level"] = helper_df["stress_level"].fillna(0.0)

le = LabelEncoder()
le.fit(helper_df["label"])

param_grid = [
    {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 400},
    {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 600},
    {"max_depth": 4, "learning_rate": 0.1, "subsample": 0.75, "colsample_bytree": 0.75, "n_estimators": 400},
]
default_params = param_grid[0].copy()
regression_params = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
}


def compute_sample_weights(phases: np.ndarray, labels: np.ndarray) -> np.ndarray:
    phase_weights = pd.Series(phases).map(lambda p: PHASE_WEIGHT_MAP.get(p, 1.0)).fillna(1.0).to_numpy()
    label_counts = pd.Series(labels).value_counts()
    class_weights = {
        label: len(labels) / (len(label_counts) * count)
        for label, count in label_counts.items()
    }
    class_weight_vec = np.array([class_weights[label] for label in labels])
    return phase_weights * class_weight_vec


def prepare_training_subset(data: pd.DataFrame) -> pd.DataFrame:
    balanced = balance_by_phase_label(data)
    balanced["phase"] = balanced["phase"].fillna("unknown")
    return balanced.reset_index(drop=True)


def tune_hyperparameters(X: np.ndarray, y: np.ndarray, groups: np.ndarray, phases: np.ndarray) -> dict:
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        return default_params.copy()
    inner_splits = min(3, unique_groups.size)
    inner_gkf = GroupKFold(n_splits=inner_splits)
    best_params = default_params.copy()
    best_score = -np.inf
    for params in param_grid:
        fold_scores = []
        for inner_train_idx, inner_val_idx in inner_gkf.split(X, y, groups):
            scaler = StandardScaler().fit(X[inner_train_idx])
            X_inner_train = scaler.transform(X[inner_train_idx])
            X_inner_val = scaler.transform(X[inner_val_idx])
            weights_inner = compute_sample_weights(phases[inner_train_idx], y[inner_train_idx])
            model = XGBClassifier(
                objective="binary:logistic",
                n_jobs=4,
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
            )
            model.fit(
                X_inner_train,
                y[inner_train_idx],
                sample_weight=weights_inner,
            )
            preds_inner = model.predict(X_inner_val)
            report = classification_report(
                y[inner_val_idx],
                preds_inner,
                labels=[0, 1],
                output_dict=True,
                zero_division=0,
            )
            fold_scores.append(report["macro avg"]["f1-score"])
        mean_score = np.mean(fold_scores) if fold_scores else -np.inf
        if mean_score > best_score:
            best_score = mean_score
            best_params = params.copy()
    return best_params


def predict_stress_label_from_level(level: float, phase: str) -> str:
    level = max(0.0, float(level))
    return stress_bucket(level, phase)

outer_groups = helper_df["subject"].to_numpy()
outer_labels = helper_df["label"].to_numpy()
unique_subjects = np.unique(outer_groups)
if unique_subjects.size < 2:
    raise RuntimeError("Need at least two subjects for cross-validation")
outer_gkf = GroupKFold(n_splits=min(5, unique_subjects.size))
results = []
confusion = np.zeros((len(le.classes_), len(le.classes_)), dtype=float)
phase_records = []

for fold, (train_idx, test_idx) in enumerate(outer_gkf.split(helper_df[numeric_cols], outer_labels, outer_groups)):
    train_df_fold = helper_df.iloc[train_idx].copy().reset_index(drop=True)
    test_df_fold = helper_df.iloc[test_idx].copy().reset_index(drop=True)

    train_prepared = prepare_training_subset(train_df_fold)
    X_train = train_prepared[numeric_cols].to_numpy(dtype=float)
    phase_train = train_prepared["phase"].to_numpy()
    binary_labels_train = (train_prepared["label"] != "no_stress").astype(int).to_numpy()

    best_params = tune_hyperparameters(X_train, binary_labels_train, train_prepared["subject"].to_numpy(), phase_train)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(test_df_fold[numeric_cols].to_numpy(dtype=float))

    weights_stage1 = compute_sample_weights(phase_train, binary_labels_train)
    model_stage1 = XGBClassifier(
        objective="binary:logistic",
        n_jobs=4,
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
    )
    model_stage1.fit(X_train_scaled, binary_labels_train, sample_weight=weights_stage1)

    stress_mask_train = binary_labels_train.astype(bool)
    if not np.any(stress_mask_train):
        continue
    stage2_features = X_train_scaled[stress_mask_train]
    stage2_targets = train_prepared.loc[stress_mask_train, "stress_level"].to_numpy(dtype=float)
    valid_stage2 = np.isfinite(stage2_targets)
    stage2_features = stage2_features[valid_stage2]
    stage2_targets = stage2_targets[valid_stage2]
    phase_stage2 = phase_train[stress_mask_train][valid_stage2]
    if stage2_features.size == 0:
        continue
    weights_stage2 = np.array([PHASE_WEIGHT_MAP.get(phase, 1.0) for phase in phase_stage2])
    model_stage2 = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=regression_params["n_estimators"],
        learning_rate=regression_params["learning_rate"],
        max_depth=regression_params["max_depth"],
        subsample=regression_params["subsample"],
        colsample_bytree=regression_params["colsample_bytree"],
        n_jobs=4,
    )
    model_stage2.fit(stage2_features, stage2_targets, sample_weight=weights_stage2)

    stage1_proba = model_stage1.predict_proba(X_test_scaled)[:, 1]
    stress_pred_mask = stage1_proba >= STRESS_PROB_THRESHOLD
    preds = np.array(["no_stress"] * len(test_idx), dtype=object)
    if stress_pred_mask.any():
        stress_features = X_test_scaled[stress_pred_mask]
        stress_levels_pred = model_stage2.predict(stress_features)
        phase_subset = test_df_fold.loc[stress_pred_mask, "phase"].to_numpy()
        mapped_labels = [
            predict_stress_label_from_level(level, phase_name)
            for level, phase_name in zip(stress_levels_pred, phase_subset)
        ]
        preds[stress_pred_mask] = mapped_labels

    preds = apply_phase_smoothing(
        test_df_fold[["subject", "phase", "win_start"]],
        preds.astype(object),
        window=PHASE_SMOOTHING_WINDOW,
    )
    true_labels = test_df_fold["label"].to_numpy()
    accuracy = np.mean(preds == true_labels)
    report = classification_report(
        true_labels,
        preds,
        labels=le.classes_,
        output_dict=True,
        zero_division=0,
    )
    confusion += confusion_matrix(true_labels, preds, labels=le.classes_)
    results.append(
        {
            "fold": fold,
            "accuracy": accuracy,
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
        }
    )
    phase_records.append(
        pd.DataFrame(
            {
                "phase": test_df_fold["phase"].to_numpy(),
                "true": true_labels,
                "pred": preds,
            }
        )
    )

results_df = pd.DataFrame(results)
print("Cross-validated metrics:")
print(results_df)
print("Mean metrics:")
print(results_df[["accuracy", "macro_f1", "weighted_f1"]].mean())

conf_df = pd.DataFrame(
    confusion / max(1, len(results)),
    index=[f"true_{c}" for c in le.classes_],
    columns=[f"pred_{c}" for c in le.classes_],
)
print("Average confusion matrix:")
print(conf_df.round(1))

if phase_records:
    phase_df = pd.concat(phase_records, ignore_index=True)
    phase_summary = []
    for phase_name, subset in phase_df.groupby("phase"):
        phase_report = classification_report(
            subset["true"],
            subset["pred"],
            labels=le.classes_,
            output_dict=True,
            zero_division=0,
        )
        phase_summary.append(
            {
                "phase": phase_name,
                "support": len(subset),
                "accuracy": np.mean(subset["true"] == subset["pred"]),
                "macro_f1": phase_report["macro avg"]["f1-score"],
                "weighted_f1": phase_report["weighted avg"]["f1-score"],
            }
        )
    phase_summary_df = pd.DataFrame(phase_summary).sort_values("phase")
    print("Per-phase diagnostics:")
    print(phase_summary_df)

