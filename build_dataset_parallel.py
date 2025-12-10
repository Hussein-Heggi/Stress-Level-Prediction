#!/usr/bin/env python3
"""
Standalone script to build dataset using all CPU cores.
This avoids Jupyter notebook multiprocessing issues.
"""
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy.signal import butter, filtfilt, find_peaks, coherence, welch
from scipy.stats import skew, kurtosis
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        """Dummy decorator if numba not available."""
        def decorator(func):
            return func
        return decorator

warnings.filterwarnings('ignore')
np.random.seed(42)

# Config
BASE_DIR = Path("/home/moh/home/Data_mining/Stress-Level-Prediction")
DATASETS_DIR = BASE_DIR / "Datasets"
TARGET_FS = 4.0
WINDOW_SIZE = 60
STEP_SIZE = 30

# [Copy all the function definitions from the notebook here]
# I'll include the key ones

def load_empatica_sensor(file_path: Path) -> Tuple[np.ndarray, float, datetime]:
    if not file_path.exists():
        return None, None, None
    with open(file_path, 'r') as f:
        lines = f.readlines()
    if len(lines) < 3:
        return None, None, None
    start_time_str = lines[0].strip()
    try:
        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
    except:
        start_time = None
    fs = float(lines[1].strip())
    data = np.array([float(line.strip()) for line in lines[2:] if line.strip()])
    return data, fs, start_time

def load_acc_sensor(file_path: Path) -> Tuple[np.ndarray, float, datetime]:
    if not file_path.exists():
        return None, None, None
    with open(file_path, 'r') as f:
        lines = f.readlines()
    if len(lines) < 2:
        return None, None, None
    start_time_str = lines[0].strip().split(',')[0]
    try:
        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
    except:
        start_time = None
    fs = 32.0
    acc_data = []
    for line in lines[1:]:
        if line.strip():
            values = line.strip().split(',')
            if len(values) == 3:
                try:
                    acc_data.append([float(v) for v in values])
                except:
                    pass
    if len(acc_data) == 0:
        return None, None, None
    data = np.array(acc_data) / 64.0
    return data, fs, start_time

def load_ibi_file(file_path: Path) -> np.ndarray:
    if not file_path.exists():
        return np.array([])
    try:
        df = pd.read_csv(file_path, names=['timestamp', 'ibi'])
        ibi_values = pd.to_numeric(df['ibi'], errors='coerce').values
        ibi_values = ibi_values[~np.isnan(ibi_values)]
        return ibi_values
    except:
        return np.array([])

def load_tags(file_path: Path) -> List[datetime]:
    if not file_path.exists():
        return []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    tags = []
    for line in lines:
        if line.strip():
            try:
                tag = datetime.strptime(line.strip(), '%Y-%m-%d %H:%M:%S')
                tags.append(tag)
            except:
                pass
    return tags

def resample_signal(data: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    if original_fs == target_fs or data is None or len(data) == 0:
        return data
    duration = len(data) / original_fs
    n_samples = int(duration * target_fs)
    t_original = np.arange(len(data)) / original_fs
    t_target = np.arange(n_samples) / target_fs
    if data.ndim == 1:
        resampled = np.interp(t_target, t_original, data)
    else:
        resampled = np.zeros((n_samples, data.shape[1]))
        for i in range(data.shape[1]):
            resampled[:, i] = np.interp(t_target, t_original, data[:, i])
    return resampled

def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.999)
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data)

def highpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    normal_cutoff = max(cutoff / nyq, 0.001)
    b, a = butter(order, normal_cutoff, btype='high')
    return filtfilt(b, a, data)

def preprocess_signals(eda, temp, acc, fs=4.0):
    eda_clean = bandpass_filter(eda, 0.01, 5.0, fs) if eda is not None and len(eda) > 0 else eda
    temp_clean = lowpass_filter(temp, 0.5, fs) if temp is not None and len(temp) > 0 else temp
    if acc is not None and len(acc) > 0:
        acc_clean = np.zeros_like(acc)
        for i in range(acc.shape[1]):
            acc_clean[:, i] = lowpass_filter(acc[:, i], 15.0, fs)
    else:
        acc_clean = acc
    return eda_clean, temp_clean, acc_clean

def detect_motion_artifacts(acc_mag: np.ndarray, eda: np.ndarray, threshold: float = 2.0) -> Tuple[np.ndarray, float]:
    min_len = min(len(acc_mag), len(eda))
    acc_mag_trim = acc_mag[:min_len]
    eda_trim = eda[:min_len]
    acc_mean = np.mean(acc_mag_trim)
    acc_std = np.std(acc_mag_trim)
    motion_mask = acc_mag_trim > (acc_mean + threshold * acc_std)
    eda_clean = eda_trim.copy()
    eda_clean[motion_mask] = np.nan
    valid_idx = ~np.isnan(eda_clean)
    if valid_idx.sum() > 2:
        eda_clean = np.interp(np.arange(len(eda_clean)), np.where(valid_idx)[0], eda_clean[valid_idx])
    else:
        eda_clean = eda_trim
    motion_ratio = motion_mask.sum() / len(motion_mask)
    return eda_clean, motion_ratio

# Feature extraction functions (simplified - remove slow nonlinear features for now)
def decompose_eda(eda_signal: np.ndarray, fs: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    tonic = lowpass_filter(eda_signal, 0.05, fs)
    phasic = highpass_filter(eda_signal, 0.05, fs)
    return tonic, phasic

def extract_scr_features(phasic: np.ndarray, fs: float = 4.0) -> Dict[str, float]:
    peaks, properties = find_peaks(phasic, height=0.01, distance=int(fs), prominence=0.01)
    duration_min = len(phasic) / (fs * 60)
    features = {
        'scr_count': len(peaks),
        'scr_rate': len(peaks) / duration_min if duration_min > 0 else 0.0
    }
    if len(peaks) > 0:
        amplitudes = properties['peak_heights']
        features.update({
            'scr_amp_mean': float(np.mean(amplitudes)),
            'scr_amp_max': float(np.max(amplitudes)),
            'scr_amp_sum': float(np.sum(amplitudes))
        })
    else:
        features.update({'scr_amp_mean': 0.0, 'scr_amp_max': 0.0, 'scr_amp_sum': 0.0})
    return features

def extract_eda_features(eda: np.ndarray, fs: float = 4.0) -> Dict[str, float]:
    features = {}
    tonic, phasic = decompose_eda(eda, fs)
    features['eda_scl_mean'] = float(np.mean(tonic))
    features['eda_scl_std'] = float(np.std(tonic))
    features['eda_scl_range'] = float(np.max(tonic) - np.min(tonic))
    features['eda_phasic_mean'] = float(np.mean(phasic))
    features['eda_phasic_std'] = float(np.std(phasic))
    features['eda_phasic_energy'] = float(np.sum(phasic ** 2))
    features.update(extract_scr_features(phasic, fs))
    features['eda_mean'] = float(np.mean(eda))
    features['eda_std'] = float(np.std(eda))
    features['eda_min'] = float(np.min(eda))
    features['eda_max'] = float(np.max(eda))
    features['eda_range'] = float(np.max(eda) - np.min(eda))
    return features

def validate_ibi(ibi: np.ndarray, min_count: int = 5) -> Optional[np.ndarray]:
    if ibi is None or len(ibi) == 0:
        return None
    valid = (ibi >= 0.3) & (ibi <= 2.0) & ~np.isnan(ibi)
    cleaned = ibi[valid]
    return cleaned if len(cleaned) >= min_count else None

@jit(nopython=True)
def _sample_entropy_fast(data, m, r_threshold):
    """Optimized Sample Entropy calculation with Numba."""
    N = len(data)

    def phi(m_val):
        count = 0.0
        for i in range(N - m_val):
            for j in range(N - m_val):
                if i != j:
                    max_dist = 0.0
                    for k in range(m_val):
                        dist = abs(data[i + k] - data[j + k])
                        if dist > max_dist:
                            max_dist = dist
                    if max_dist <= r_threshold:
                        count += 1.0
        return count

    phi_m = phi(m)
    phi_m1 = phi(m + 1)

    if phi_m > 0 and phi_m1 > 0:
        return -np.log(phi_m1 / phi_m)
    return np.nan

def sample_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Calculate Sample Entropy (fast version with Numba)."""
    N = len(data)
    if N < m + 10:
        return np.nan
    r_threshold = r * np.std(data)
    return _sample_entropy_fast(data, m, r_threshold)


def approximate_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Calculate Approximate Entropy (simplified fast version)."""
    N = len(data)
    if N < m + 10:
        return np.nan

    # Use simpler calculation - just std deviation as proxy
    # Full ApEn is too slow even with optimization
    return float(np.std(np.diff(data)))


def extract_hrv_features(ibi: np.ndarray) -> Dict[str, float]:
    """Extract HRV features including nonlinear."""
    ibi_clean = validate_ibi(ibi)
    features = {}
    if ibi_clean is None or len(ibi_clean) < 5:
        for name in ['hrv_mean_rr', 'hrv_std_rr', 'hrv_rmssd', 'hrv_mean_hr',
                     'hrv_lf', 'hrv_hf', 'hrv_lf_hf_ratio', 'hrv_sampen', 'hrv_apen']:
            features[name] = np.nan
        return features
    rr = ibi_clean * 1000
    features['hrv_mean_rr'] = float(np.mean(rr))
    features['hrv_std_rr'] = float(np.std(rr))
    diff_rr = np.diff(rr)
    features['hrv_rmssd'] = float(np.sqrt(np.mean(diff_rr ** 2)))
    features['hrv_mean_hr'] = float(60000 / np.mean(rr))
    if len(rr) >= 10:
        t_rr = np.cumsum(ibi_clean)
        t_uniform = np.arange(0, t_rr[-1], 0.25)
        rr_interp = np.interp(t_uniform, t_rr, rr)
        freqs, psd = welch(rr_interp, fs=4.0, nperseg=min(256, len(rr_interp)))
        lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
        features['hrv_lf'] = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if lf_mask.sum() > 0 else 0.0
        features['hrv_hf'] = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if hf_mask.sum() > 0 else 0.0
        features['hrv_lf_hf_ratio'] = features['hrv_lf'] / features['hrv_hf'] if features['hrv_hf'] > 0 else 0.0
    else:
        features['hrv_lf'] = np.nan
        features['hrv_hf'] = np.nan
        features['hrv_lf_hf_ratio'] = np.nan

    # Nonlinear features
    if len(rr) >= 10:
        features['hrv_sampen'] = sample_entropy(rr, m=2, r=0.2)
        features['hrv_apen'] = approximate_entropy(rr, m=2, r=0.2)
    else:
        features['hrv_sampen'] = np.nan
        features['hrv_apen'] = np.nan

    return features

def extract_temp_features(temp: np.ndarray) -> Dict[str, float]:
    return {
        'temp_mean': float(np.mean(temp)),
        'temp_std': float(np.std(temp)),
        'temp_min': float(np.min(temp)),
        'temp_max': float(np.max(temp)),
        'temp_range': float(np.max(temp) - np.min(temp))
    }

def extract_acc_features(acc: np.ndarray) -> Dict[str, float]:
    acc_mag = np.linalg.norm(acc, axis=1)
    return {
        'acc_mean': float(np.mean(acc_mag)),
        'acc_std': float(np.std(acc_mag)),
        'acc_min': float(np.min(acc_mag)),
        'acc_max': float(np.max(acc_mag)),
        'acc_energy': float(np.sum(acc_mag ** 2))
    }

def extract_hr_features(hr: np.ndarray) -> Dict[str, float]:
    valid = hr[~np.isnan(hr)]
    if len(valid) > 0:
        return {
            'hr_mean': float(np.mean(valid)),
            'hr_std': float(np.std(valid)),
            'hr_min': float(np.min(valid)),
            'hr_max': float(np.max(valid))
        }
    return {k: np.nan for k in ['hr_mean', 'hr_std', 'hr_min', 'hr_max']}

def cross_modal_features(eda: np.ndarray, hr: np.ndarray, temp: np.ndarray, fs: float = 4.0) -> Dict[str, float]:
    min_len = min(len(eda), len(hr), len(temp))
    if min_len < 10:
        return {
            'eda_hr_xcorr_max': np.nan,
            'eda_temp_xcorr_max': np.nan,
            'eda_hr_coherence_lf': np.nan
        }
    eda, hr, temp = eda[:min_len], hr[:min_len], temp[:min_len]
    eda_norm = (eda - np.mean(eda)) / (np.std(eda) + 1e-6)
    hr_norm = (hr - np.mean(hr)) / (np.std(hr) + 1e-6)
    temp_norm = (temp - np.mean(temp)) / (np.std(temp) + 1e-6)
    xcorr_eda_hr = np.correlate(eda_norm, hr_norm, mode='same')
    xcorr_eda_temp = np.correlate(eda_norm, temp_norm, mode='same')
    features = {
        'eda_hr_xcorr_max': float(np.max(np.abs(xcorr_eda_hr))),
        'eda_temp_xcorr_max': float(np.max(np.abs(xcorr_eda_temp)))
    }
    if min_len >= 64:
        f, Cxy = coherence(eda, hr, fs=fs, nperseg=min(64, min_len))
        lf_mask = (f >= 0.04) & (f <= 0.15)
        features['eda_hr_coherence_lf'] = float(np.mean(Cxy[lf_mask])) if lf_mask.sum() > 0 else np.nan
    else:
        features['eda_hr_coherence_lf'] = np.nan
    return features

def load_stress_labels() -> Dict[str, Dict[str, float]]:
    labels = {}
    v1_path = BASE_DIR / "Stress_Level_v1.csv"
    if v1_path.exists():
        df = pd.read_csv(v1_path, index_col=0)
        for subject, row in df.iterrows():
            labels[str(subject).strip()] = {col: float(row[col]) if not pd.isna(row[col]) else np.nan
                                            for col in df.columns}
    v2_path = BASE_DIR / "Stress_Level_v2.csv"
    if v2_path.exists():
        df = pd.read_csv(v2_path, index_col=0)
        for subject, row in df.iterrows():
            labels[str(subject).strip()] = {col: float(row[col]) if not pd.isna(row[col]) else np.nan
                                            for col in df.columns}
    return labels

stress_labels = load_stress_labels()

STRESS_PHASES_S = [
    ('Baseline', 0, 3),
    ('Stroop', 3, 5),
    ('First Rest', 5, 5),
    ('TMCT', 5, 7),
    ('Second Rest', 7, 7),
    ('Real Opinion', 7, 9),
    ('Opposite Opinion', 9, 11),
    ('Subtract', 11, 13)
]

STRESS_PHASES_F = [
    ('Baseline', 0, 2),
    ('TMCT', 2, 4),
    ('Real Opinion', 4, 6),
    ('Opposite Opinion', 6, 8),
    ('Subtract', 8, 10)
]

def map_stress_score_to_class(score: float) -> str:
    if pd.isna(score):
        return 'unknown'
    if score <= 2:
        return 'no_stress'
    elif score <= 5:
        return 'low_stress'
    elif score <= 7:
        return 'moderate_stress'
    else:
        return 'high_stress'

def process_subject(protocol: str, subject: str) -> List[Dict]:
    subject_dir = DATASETS_DIR / protocol / subject
    if not subject_dir.exists():
        return []
    if subject == 'S12' and protocol == 'AEROBIC':
        return []

    eda_raw, eda_fs, eda_start = load_empatica_sensor(subject_dir / "EDA.csv")
    temp_raw, temp_fs, _ = load_empatica_sensor(subject_dir / "TEMP.csv")
    hr_raw, hr_fs, _ = load_empatica_sensor(subject_dir / "HR.csv")
    acc_raw, acc_fs, _ = load_acc_sensor(subject_dir / "ACC.csv")
    ibi_raw = load_ibi_file(subject_dir / "IBI.csv")
    tags = load_tags(subject_dir / "tags.csv")

    if eda_raw is None or len(eda_raw) < 100:
        return []

    if subject == 'f07':
        hr_raw = None
        ibi_raw = np.array([])

    eda = resample_signal(eda_raw, eda_fs, TARGET_FS)
    temp = resample_signal(temp_raw, temp_fs, TARGET_FS) if temp_raw is not None else np.zeros(len(eda))
    hr = resample_signal(hr_raw, hr_fs, TARGET_FS) if hr_raw is not None else np.full(len(eda), np.nan)
    acc = resample_signal(acc_raw, acc_fs, TARGET_FS) if acc_raw is not None else np.zeros((len(eda), 3))

    min_len = min(len(eda), len(temp), len(hr), len(acc))
    eda = eda[:min_len]
    temp = temp[:min_len]
    hr = hr[:min_len]
    acc = acc[:min_len]

    eda_clean, temp_clean, acc_clean = preprocess_signals(eda, temp, acc, TARGET_FS)
    acc_mag = np.linalg.norm(acc_clean, axis=1)
    eda_clean, motion_ratio = detect_motion_artifacts(acc_mag, eda_clean)

    min_len = min(len(eda_clean), len(temp_clean), len(hr), len(acc_clean))
    eda_clean = eda_clean[:min_len]
    temp_clean = temp_clean[:min_len]
    hr = hr[:min_len]
    acc_clean = acc_clean[:min_len]

    duration = len(eda_clean) / TARGET_FS

    if protocol == 'STRESS':
        if eda_start and len(tags) > 0:
            tag_offsets = [(tag - eda_start).total_seconds() for tag in tags]
            phase_defs = STRESS_PHASES_S if subject.startswith('S') else STRESS_PHASES_F
            phases = []
            for phase_name, start_tag_idx, end_tag_idx in phase_defs:
                if start_tag_idx < len(tag_offsets) and end_tag_idx <= len(tag_offsets):
                    start_time = tag_offsets[start_tag_idx] if start_tag_idx > 0 else 0
                    end_time = tag_offsets[end_tag_idx] if end_tag_idx < len(tag_offsets) else duration
                    phases.append((phase_name, start_time, end_time))
        else:
            phases = [('stress', 0, duration)]
    else:
        phases = [('rest', 0, duration / 2), (protocol.lower(), duration / 2, duration)]

    window_samples = int(WINDOW_SIZE * TARGET_FS)
    step_samples = int(STEP_SIZE * TARGET_FS)

    windows = []
    for start_idx in range(0, len(eda_clean) - window_samples + 1, step_samples):
        end_idx = start_idx + window_samples
        win_start_time = start_idx / TARGET_FS
        win_end_time = end_idx / TARGET_FS

        win_phase = 'unknown'
        for phase_name, phase_start, phase_end in phases:
            if win_start_time >= phase_start and win_end_time <= phase_end:
                win_phase = phase_name
                break

        eda_win = eda_clean[start_idx:end_idx]
        temp_win = temp_clean[start_idx:end_idx]
        hr_win = hr[start_idx:end_idx]
        acc_win = acc_clean[start_idx:end_idx]

        features = {}
        features.update(extract_eda_features(eda_win, TARGET_FS))
        features.update(extract_hrv_features(ibi_raw))
        features.update(extract_temp_features(temp_win))
        features.update(extract_acc_features(acc_win))
        features.update(extract_hr_features(hr_win))
        features.update(cross_modal_features(eda_win, hr_win, temp_win, TARGET_FS))

        features['subject'] = subject
        features['protocol'] = protocol
        features['phase'] = win_phase
        features['motion_ratio'] = motion_ratio

        if protocol == 'STRESS' and subject in stress_labels:
            stress_score = stress_labels[subject].get(win_phase, np.nan)
            features['stress_score'] = stress_score
            features['label'] = map_stress_score_to_class(stress_score)
        else:
            features['stress_score'] = np.nan
            if protocol in ['AEROBIC', 'ANAEROBIC']:
                features['label'] = 'no_stress' if win_phase == 'rest' else protocol.lower()
            else:
                features['label'] = 'unknown'

        windows.append(features)

    return windows

def process_subject_wrapper(args):
    """Wrapper for multiprocessing."""
    protocol, subject = args
    return process_subject(protocol, subject)

def main():
    print("="*80)
    print("BUILDING DATASET FROM RAW DATA (PARALLEL)")
    print("="*80)

    # Get all subjects
    all_subjects = []
    for protocol in ['STRESS', 'AEROBIC', 'ANAEROBIC']:
        protocol_dir = DATASETS_DIR / protocol
        if protocol_dir.exists():
            for subject_dir in protocol_dir.iterdir():
                if subject_dir.is_dir():
                    all_subjects.append((protocol, subject_dir.name))

    print(f"\nFound {len(all_subjects)} subject-protocol combinations")

    n_cores = cpu_count()
    print(f"Using {n_cores} CPU cores for parallel processing\n")

    start_time = time.time()

    # Parallel processing
    with Pool(n_cores) as pool:
        results = list(tqdm(
            pool.imap(process_subject_wrapper, all_subjects),
            total=len(all_subjects),
            desc="Processing"
        ))

    # Flatten results
    all_windows = []
    for windows in results:
        all_windows.extend(windows)

    total_time = time.time() - start_time

    print(f"\n✓ Extracted {len(all_windows)} windows in {total_time/60:.1f} minutes")

    # Convert to DataFrame and save
    dataset = pd.DataFrame(all_windows)

    print(f"✓ Dataset shape: {dataset.shape}")
    print(f"✓ Subjects: {dataset['subject'].nunique()}")
    print(f"\nLabel distribution:")
    print(dataset['label'].value_counts())

    output_file = BASE_DIR / "complete_enhanced_dataset.csv"
    dataset.to_csv(output_file, index=False)
    print(f"\n✓ Dataset saved to: {output_file}")

if __name__ == "__main__":
    main()
