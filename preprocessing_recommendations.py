"""
Advanced Preprocessing Recommendations for Stress Level Prediction
==================================================================

These enhancements build on your existing pipeline to improve model performance.
"""

import numpy as np
from scipy import signal
from scipy.stats import entropy
import pywt


# ============================================================================
# 1. ENHANCED EDA PROCESSING
# ============================================================================

def decompose_eda_cvxeda(eda_signal, sampling_rate=4.0):
    """
    Decompose EDA into tonic (SCL) and phasic (SCR) components using cvxEDA.

    SCR (phasic) is more responsive to acute stress.
    Install: pip install cvxopt (for cvxEDA implementation)

    Returns:
        tonic: Slow-varying baseline (SCL)
        phasic: Fast-varying stress response (SCR)
    """
    try:
        from cvxEDA import cvxEDA
        phasic, _, tonic, _, _, _, _ = cvxEDA(eda_signal, 1.0/sampling_rate)
        return tonic, phasic
    except ImportError:
        # Fallback: Simple moving average decomposition
        window_size = int(10 * sampling_rate)  # 10-second window
        tonic = np.convolve(eda_signal, np.ones(window_size)/window_size, mode='same')
        phasic = eda_signal - tonic
        return tonic, phasic


def extract_scr_features(phasic_eda, sampling_rate=4.0):
    """
    Extract Skin Conductance Response (SCR) features from phasic EDA.

    Key stress indicators:
    - SCR frequency: Number of responses per minute
    - SCR amplitude: Peak heights
    - SCR rise time: Time to peak
    """
    features = {}

    # Detect SCR peaks (threshold: 0.01 µS above baseline)
    peaks, properties = signal.find_peaks(
        phasic_eda,
        height=0.01,
        distance=int(1.0 * sampling_rate),  # Min 1s between peaks
        prominence=0.01
    )

    # SCR frequency (peaks per minute)
    duration_min = len(phasic_eda) / (sampling_rate * 60)
    features['scr_freq'] = len(peaks) / duration_min if duration_min > 0 else 0

    # SCR amplitude statistics
    if len(peaks) > 0:
        features['scr_amp_mean'] = np.mean(properties['peak_heights'])
        features['scr_amp_max'] = np.max(properties['peak_heights'])
        features['scr_amp_sum'] = np.sum(properties['peak_heights'])
    else:
        features['scr_amp_mean'] = 0
        features['scr_amp_max'] = 0
        features['scr_amp_sum'] = 0

    # SCR rise time (time from onset to peak)
    # Approximate using width at half-prominence
    if 'widths' in properties:
        features['scr_rise_time'] = np.mean(properties['widths']) / sampling_rate
    else:
        features['scr_rise_time'] = 0

    return features


# ============================================================================
# 2. WAVELET-BASED FEATURES (Multi-Scale Analysis)
# ============================================================================

def wavelet_features(signal_data, wavelet='db4', level=4):
    """
    Extract wavelet decomposition features for multi-scale analysis.

    Different stress responses occur at different time scales:
    - Fast oscillations: Acute stress reactions
    - Slow oscillations: Sustained stress
    """
    features = {}

    # Discrete wavelet transform
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)

    for i, coeff in enumerate(coeffs):
        prefix = f'wav_d{i}' if i > 0 else 'wav_a'
        features[f'{prefix}_energy'] = np.sum(coeff ** 2)
        features[f'{prefix}_std'] = np.std(coeff)
        features[f'{prefix}_entropy'] = entropy(np.abs(coeff) + 1e-10)

    return features


# ============================================================================
# 3. ENHANCED HRV FEATURES (Nonlinear Dynamics)
# ============================================================================

def nonlinear_hrv_features(ibi_intervals):
    """
    Nonlinear HRV features that capture complexity changes during stress.

    Stress typically reduces HRV complexity.
    """
    features = {}

    if len(ibi_intervals) < 10:
        return {
            'sampen': np.nan,
            'dfa_alpha1': np.nan,
            'dfa_alpha2': np.nan,
            'approximate_entropy': np.nan
        }

    # Sample Entropy (complexity measure)
    features['sampen'] = sample_entropy(ibi_intervals, m=2, r=0.2)

    # Detrended Fluctuation Analysis (self-similarity)
    features['dfa_alpha1'], features['dfa_alpha2'] = detrended_fluctuation_analysis(ibi_intervals)

    # Approximate Entropy
    features['approximate_entropy'] = approximate_entropy(ibi_intervals, m=2, r=0.2)

    return features


def sample_entropy(data, m=2, r=0.2):
    """Calculate Sample Entropy (SampEn)."""
    N = len(data)
    if N < m + 1:
        return np.nan

    # Normalize r to std
    r = r * np.std(data)

    def _maxdist(xi, xj):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])

    def _phi(m):
        patterns = np.array([[data[j] for j in range(i, i + m)] for i in range(N - m)])
        count = 0
        for i in range(len(patterns)):
            for j in range(len(patterns)):
                if i != j and _maxdist(patterns[i], patterns[j]) <= r:
                    count += 1
        return count / (N - m) / (N - m - 1) if (N - m) > 1 else 0

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    return -np.log(phi_m1 / phi_m) if phi_m > 0 and phi_m1 > 0 else np.nan


def approximate_entropy(data, m=2, r=0.2):
    """Calculate Approximate Entropy (ApEn)."""
    N = len(data)
    if N < m + 1:
        return np.nan

    r = r * np.std(data)

    def _phi(m):
        patterns = np.array([[data[j] for j in range(i, i + m)] for i in range(N - m + 1)])
        C = np.zeros(N - m + 1)
        for i in range(N - m + 1):
            C[i] = np.sum(np.max(np.abs(patterns - patterns[i]), axis=1) <= r) / (N - m + 1)
        return np.sum(np.log(C)) / (N - m + 1)

    return _phi(m) - _phi(m + 1)


def detrended_fluctuation_analysis(data):
    """
    Calculate DFA alpha coefficients.

    Returns:
        alpha1: Short-term fluctuations (4-11 beats)
        alpha2: Long-term fluctuations (>11 beats)
    """
    N = len(data)
    if N < 16:
        return np.nan, np.nan

    # Integrate the signal
    y = np.cumsum(data - np.mean(data))

    # Define scales
    scales = np.unique(np.logspace(0.5, 2, 20).astype(int))
    scales = scales[scales < N // 4]

    F = []
    for scale in scales:
        # Divide into segments
        n_segments = N // scale
        F_scale = 0

        for i in range(n_segments):
            segment = y[i * scale:(i + 1) * scale]
            # Fit polynomial
            t = np.arange(len(segment))
            coeffs = np.polyfit(t, segment, 1)
            trend = np.polyval(coeffs, t)
            # Calculate fluctuation
            F_scale += np.mean((segment - trend) ** 2)

        F.append(np.sqrt(F_scale / n_segments))

    # Fit power law: F(n) ~ n^alpha
    log_scales = np.log(scales)
    log_F = np.log(F)

    # Short-term (4-11 beats)
    mask1 = (scales >= 4) & (scales <= 11)
    if np.sum(mask1) > 1:
        alpha1 = np.polyfit(log_scales[mask1], log_F[mask1], 1)[0]
    else:
        alpha1 = np.nan

    # Long-term (>11 beats)
    mask2 = scales > 11
    if np.sum(mask2) > 1:
        alpha2 = np.polyfit(log_scales[mask2], log_F[mask2], 1)[0]
    else:
        alpha2 = np.nan

    return alpha1, alpha2


# ============================================================================
# 4. CROSS-MODAL SYNCHRONY FEATURES
# ============================================================================

def cross_signal_features(eda, hr, temp, sampling_rate=4.0):
    """
    Extract cross-modal synchrony features.

    During stress, physiological signals show coordinated responses.
    """
    features = {}

    # Ensure same length
    min_len = min(len(eda), len(hr), len(temp))
    eda = eda[:min_len]
    hr = hr[:min_len]
    temp = temp[:min_len]

    # Cross-correlation between signals
    features['eda_hr_xcorr_max'] = np.max(np.correlate(
        (eda - np.mean(eda)) / (np.std(eda) + 1e-6),
        (hr - np.mean(hr)) / (np.std(hr) + 1e-6),
        mode='same'
    ))

    features['eda_temp_xcorr_max'] = np.max(np.correlate(
        (eda - np.mean(eda)) / (np.std(eda) + 1e-6),
        (temp - np.mean(temp)) / (np.std(temp) + 1e-6),
        mode='same'
    ))

    # Coherence in low-frequency band (0.04-0.15 Hz)
    if len(eda) >= 64:
        f, Cxy_eda_hr = signal.coherence(eda, hr, fs=sampling_rate, nperseg=64)
        low_freq_mask = (f >= 0.04) & (f <= 0.15)
        features['eda_hr_coherence_lf'] = np.mean(Cxy_eda_hr[low_freq_mask])
    else:
        features['eda_hr_coherence_lf'] = np.nan

    return features


# ============================================================================
# 5. CONTEXT-AWARE NORMALIZATION
# ============================================================================

def subject_specific_normalization(features_df, subject_col='subject'):
    """
    Normalize features per subject using their baseline statistics.

    This accounts for individual differences in physiological baselines.
    """
    normalized_df = features_df.copy()

    # Get numeric columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns

    # For each subject
    for subject in features_df[subject_col].unique():
        subject_mask = features_df[subject_col] == subject
        subject_data = features_df.loc[subject_mask, numeric_cols]

        # Use rest/baseline windows for normalization
        if 'phase' in features_df.columns:
            baseline_mask = (features_df['phase'] == 'rest') & subject_mask
            if baseline_mask.sum() > 0:
                baseline_mean = features_df.loc[baseline_mask, numeric_cols].mean()
                baseline_std = features_df.loc[baseline_mask, numeric_cols].std().replace(0, 1)
            else:
                baseline_mean = subject_data.mean()
                baseline_std = subject_data.std().replace(0, 1)
        else:
            baseline_mean = subject_data.mean()
            baseline_std = subject_data.std().replace(0, 1)

        # Z-score normalization per subject
        normalized_df.loc[subject_mask, numeric_cols] = (
            (subject_data - baseline_mean) / baseline_std
        )

    return normalized_df


# ============================================================================
# 6. DATA AUGMENTATION FOR IMBALANCED CLASSES
# ============================================================================

def smote_with_temporal_constraints(X, y, subjects, random_state=42):
    """
    SMOTE augmentation that respects temporal structure.

    Only interpolates between samples from the same subject to maintain
    physiological consistency.
    """
    from imblearn.over_sampling import SMOTE
    from sklearn.neighbors import NearestNeighbors

    # Group by subject
    augmented_X = []
    augmented_y = []
    augmented_subjects = []

    for subject in np.unique(subjects):
        subject_mask = subjects == subject
        X_subj = X[subject_mask]
        y_subj = y[subject_mask]

        # Apply SMOTE within subject
        if len(np.unique(y_subj)) > 1 and len(y_subj) > 5:
            smote = SMOTE(random_state=random_state, k_neighbors=min(5, len(y_subj)-1))
            X_resampled, y_resampled = smote.fit_resample(X_subj, y_subj)
        else:
            X_resampled, y_resampled = X_subj, y_subj

        augmented_X.append(X_resampled)
        augmented_y.append(y_resampled)
        augmented_subjects.extend([subject] * len(y_resampled))

    return (
        np.vstack(augmented_X),
        np.concatenate(augmented_y),
        np.array(augmented_subjects)
    )


# ============================================================================
# 7. SLIDING WINDOW WITH ADAPTIVE OVERLAP
# ============================================================================

def adaptive_windowing(signal_dict, labels, sampling_rate=4.0,
                      base_window=60, stress_window=30, overlap_ratio=0.5):
    """
    Use smaller windows during stress phases for finer temporal resolution.

    Args:
        signal_dict: Dict of signals (e.g., {'EDA': array, 'HR': array})
        labels: Phase labels or stress indicators
        base_window: Window size (seconds) for non-stress
        stress_window: Window size (seconds) for stress
        overlap_ratio: Overlap ratio (0.5 = 50% overlap)
    """
    # Implementation depends on your data structure
    # This is a conceptual example
    pass


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Integration example with your existing pipeline.
    """

    # Example: Enhance EDA features
    def enhanced_eda_features(eda_signal, fs=4.0):
        # Your existing features
        from your_module import eda_features as original_eda_features
        features = original_eda_features(eda_signal, fs)

        # Add decomposition features
        tonic, phasic = decompose_eda_cvxeda(eda_signal, fs)
        scr_feats = extract_scr_features(phasic, fs)
        features.update(scr_feats)

        # Add wavelet features
        wav_feats = wavelet_features(eda_signal, wavelet='db4', level=4)
        features.update({f'eda_{k}': v for k, v in wav_feats.items()})

        return features

    # Example: Enhance HRV features
    def enhanced_hrv_features(ibi_intervals):
        from your_module import hrv_features as original_hrv_features
        features = original_hrv_features(ibi_intervals)

        # Add nonlinear features
        nl_feats = nonlinear_hrv_features(ibi_intervals)
        features.update(nl_feats)

        return features

    print("Preprocessing recommendations loaded successfully!")
    print("\nKey improvements:")
    print("1. EDA decomposition into tonic/phasic components")
    print("2. SCR-specific features (frequency, amplitude, rise time)")
    print("3. Wavelet multi-scale analysis")
    print("4. Nonlinear HRV features (SampEn, DFA, ApEn)")
    print("5. Cross-modal synchrony features")
    print("6. Subject-specific normalization")
    print("7. Temporal-constrained SMOTE augmentation")
