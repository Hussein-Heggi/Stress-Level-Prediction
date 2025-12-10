# Enhancement Plan: LSTM_ResNet Stress Prediction Model
## Goal: Improve macro F1 from 36% to 60%+ through Data Mining & Preprocessing

---

## Executive Summary

**Current Performance:**
- LSTM_ResNet: 36% macro F1, 76% accuracy
- Major issues: Poor low_stress (0% recall), moderate high_stress (20.8%)
- Severe class imbalance despite SMOTE

**Target Performance:**
- 60-65% macro F1
- 87-90% accuracy
- Balanced class detection across all stress levels

**Strategy:**
Implement 5 progressive phases focusing on data enrichment and architectural improvements, each building on the previous.

---

## Phase 1: High-Impact Quick Wins (Expected: +8-12% macro F1)
**Timeline: Week 1 | Target: 45-48% macro F1**

### 1.1 Subject-Specific Baseline Normalization
**Impact: +5-7% macro F1 (HIGHEST PRIORITY)**

**Current Problem:**
- Global normalization: `sequences.mean(axis=(0, 2))` averages across all subjects
- Ignores 10x variation in baseline EDA, 40bpm variation in resting HR
- Model learns subject signatures instead of stress patterns

**Implementation:**
```python
def subject_baseline_normalization(sequences, labels, subjects):
    """Normalize each subject relative to their rest-phase baseline."""
    normalized = sequences.copy()

    for subject in np.unique(subjects):
        subject_mask = subjects == subject
        rest_mask = subject_mask & (labels == 'no_stress')

        if rest_mask.sum() > 0:
            baseline_mean = sequences[rest_mask].mean(axis=(0, 2), keepdims=True)
            baseline_std = sequences[rest_mask].std(axis=(0, 2), keepdims=True) + 1e-6
        else:
            baseline_mean = sequences[subject_mask].mean(axis=(0, 2), keepdims=True)
            baseline_std = sequences[subject_mask].std(axis=(0, 2), keepdims=True) + 1e-6

        normalized[subject_mask] = (sequences[subject_mask] - baseline_mean) / baseline_std

    return normalized
```

**Files to modify:**
- `LSTM_ResNet.ipynb` cell `a4fa9274` (post-processing step)

---

### 1.2 Enhanced Channel Features
**Impact: +2-3% macro F1**

**Current State:** 8 channels (4 raw + 4 first-order diffs)

**Add 8 new channels:**
1. **EDA_phasic** - SCR component (stress-responsive)
2. **EDA_tonic** - Baseline arousal level
3. **EDA_accel** - Second-order derivative (rate of change of SCR)
4. **EDA_ma_short** - 5-second moving average (short-term trend)
5. **EDA_ma_long** - 15-second moving average (long-term trend)
6. **BVP_envelope** - Hilbert transform envelope (HRV proxy)
7. **EDA_BVP_interaction** - Cross-channel product (autonomic coordination)
8. **ACC_smoothed** - 3-second moving average (remove high-freq noise)

**Total: 16 channels**

**Implementation:**
```python
def extract_enhanced_channels(eda, temp, acc, bvp, fs=4.0):
    channels = [eda, temp, acc, bvp]  # Original

    # First-order derivatives
    channels.extend([np.diff(eda, prepend=eda[0]),
                     np.diff(temp, prepend=temp[0]),
                     np.diff(acc, prepend=acc[0]),
                     np.diff(bvp, prepend=bvp[0])])

    # EDA decomposition (moving average filter)
    window_tonic = int(10 * fs)
    eda_tonic = np.convolve(eda, np.ones(window_tonic)/window_tonic, mode='same')
    eda_phasic = eda - eda_tonic
    channels.extend([eda_tonic, eda_phasic])

    # Second-order derivative
    eda_diff = np.diff(eda, prepend=eda[0])
    eda_accel = np.diff(eda_diff, prepend=eda_diff[0])
    channels.append(eda_accel)

    # Moving averages
    window_short = int(5 * fs)
    window_long = int(15 * fs)
    eda_ma_short = np.convolve(eda, np.ones(window_short)/window_short, mode='same')
    eda_ma_long = np.convolve(eda, np.ones(window_long)/window_long, mode='same')
    channels.extend([eda_ma_short, eda_ma_long])

    # BVP envelope
    from scipy.signal import hilbert
    bvp_envelope = np.abs(hilbert(bvp))
    channels.append(bvp_envelope)

    # Cross-channel interaction
    channels.append(eda * bvp)

    # Smoothed ACC
    acc_smoothed = np.convolve(acc, np.ones(int(3*fs))/(3*fs), mode='same')
    channels.append(acc_smoothed)

    return np.stack(channels, axis=0)
```

**Files to modify:**
- `LSTM_ResNet.ipynb` cell `83737a60` (`build_sequence_dataset` function)

---

### 1.3 Temporal-Aware Data Augmentation
**Impact: +1-2% macro F1**

**Current Problem:**
- SMOTE flattens sequences, destroys temporal structure
- `X_train.reshape(X_train.shape[0], -1)` treats 240 timesteps as independent

**New Strategy:**
- Replace SMOTE with physiologically-plausible augmentation
- Only augment minority classes (low_stress, high_stress, moderate_stress)
- Preserve temporal coherence

**Augmentation Techniques:**
1. **Time warping** (±5% temporal stretch/compress)
2. **Magnitude jittering** (Gaussian noise σ=0.05×signal_std)
3. **Window shifting** (±10% temporal offset)

**Implementation:**
```python
def temporal_augmentation(sequences, labels, augment_factor=2):
    """Augment minority classes with temporal transformations."""
    augmented = []

    label_counts = pd.Series(labels).value_counts()
    minority_threshold = label_counts.median()
    minority_classes = label_counts[label_counts < minority_threshold].index

    for seq, label in zip(sequences, labels):
        augmented.append((seq, label))

        if label in minority_classes:
            for _ in range(augment_factor):
                # Time warp
                warp_factor = np.random.uniform(0.95, 1.05)
                aug_seq = time_warp(seq, warp_factor)

                # Add noise
                noise_std = 0.05 * np.std(seq, axis=1, keepdims=True)
                aug_seq += np.random.normal(0, noise_std, aug_seq.shape)

                # Temporal shift
                shift = np.random.randint(-int(0.1*seq.shape[1]), int(0.1*seq.shape[1]))
                aug_seq = np.roll(aug_seq, shift, axis=1)

                augmented.append((aug_seq, label))

    return augmented

def time_warp(sequence, factor):
    """Temporal stretch/compress via interpolation."""
    from scipy.interpolate import interp1d
    channels, length = sequence.shape
    new_length = int(length * factor)

    warped = []
    for ch in range(channels):
        f = interp1d(np.arange(length), sequence[ch], kind='cubic')
        new_indices = np.linspace(0, length-1, new_length)
        warped_ch = f(new_indices)
        warped_ch_resampled = np.interp(np.arange(length),
                                         np.linspace(0, length-1, new_length),
                                         warped_ch)
        warped.append(warped_ch_resampled)

    return np.array(warped)
```

**Files to modify:**
- `LSTM_ResNet.ipynb` cell `8869da7c` (training loop)

---

## Phase 2: Architecture Enhancements (Expected: +6-10% macro F1)
**Timeline: Week 2 | Target: 52-58% macro F1**

### 2.1 Multi-Scale Temporal CNN
**Impact: +3-5% macro F1**

**Current Limitation:**
- Single kernel size (5) in ResNet blocks
- Misses multi-scale stress patterns:
  - Fast (0.75s): Acute SCR peaks
  - Medium (2-4s): HR acceleration
  - Slow (30-60s): EDA tonic shift

**New Architecture:**
```python
class MultiScaleLSTMResNet(nn.Module):
    def __init__(self, input_channels=16, num_classes=4):
        super().__init__()

        # Multi-scale CNN branches
        self.conv_short = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResNetBlock(64, kernel_size=3, dilation=1),
            ResNetBlock(64, kernel_size=3, dilation=1),
        )

        self.conv_medium = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResNetBlock(64, kernel_size=7, dilation=2),
            ResNetBlock(64, kernel_size=7, dilation=2),
        )

        self.conv_long = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResNetBlock(64, kernel_size=15, dilation=4),
            ResNetBlock(64, kernel_size=15, dilation=4),
        )

        # Merge branches
        self.merge = nn.Conv1d(192, 128, kernel_size=1)
        self.merge_bn = nn.BatchNorm1d(128)

        # Enhanced LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=3,
            dropout=0.3,
            batch_first=True,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Multi-scale feature extraction
        feat_short = self.conv_short(x)
        feat_medium = self.conv_medium(x)
        feat_long = self.conv_long(x)

        # Merge
        feat_concat = torch.cat([feat_short, feat_medium, feat_long], dim=1)
        feat_merged = F.relu(self.merge_bn(self.merge(feat_concat)))

        # LSTM temporal modeling
        feat_lstm = feat_merged.transpose(1, 2)
        lstm_out, _ = self.lstm(feat_lstm)

        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        temporal_features = torch.mean(attn_out, dim=1)

        # Global CNN features (skip connection)
        cnn_global = self.global_pool(feat_merged).squeeze(-1)

        # Classify
        combined = torch.cat([temporal_features, cnn_global], dim=1)
        return self.classifier(combined)
```

**Files to create:**
- New cell in `LSTM_ResNet.ipynb` after cell `cb4a1b2d`

---

### 2.2 Focal Loss for Imbalanced Classes
**Impact: +1-2% macro F1**

**Replace CrossEntropyLoss:**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

criterion = FocalLoss(alpha=1.0, gamma=2.0)
```

**Files to modify:**
- `LSTM_ResNet.ipynb` cell `8869da7c` (training loop)

---

### 2.3 Improved Training Configuration
**Impact: +2-3% macro F1**

**Changes:**
1. Increase epochs: 20 → 40
2. Reduce batch size: 64 → 32 (better generalization)
3. Increase LR: 5e-4 → 1e-3
4. Use AdamW optimizer (better weight decay)
5. Cosine annealing scheduler with warm restarts
6. Mixed precision training (faster)
7. Gradient clipping: max_norm=1.0

**Implementation:**
```python
EPOCHS = 40
BATCH_SIZE = 32
LR = 1e-3

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2,
    eta_min=1e-6
)

scaler = torch.cuda.amp.GradScaler()

# Training loop with mixed precision
for xb, yb in train_loader:
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        logits = model(xb)
        loss = criterion(logits, yb)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
```

**Files to modify:**
- `LSTM_ResNet.ipynb` cell `6d812a0e` (config)
- `LSTM_ResNet.ipynb` cell `8869da7c` (training loop)

---

## Phase 3: Advanced Signal Processing (Expected: +4-8% macro F1)
**Timeline: Week 3 | Target: 56-62% macro F1**

### 3.1 Wavelet Multi-Resolution Analysis
**Impact: +2-3% macro F1**

**Add wavelet decomposition channels:**
```python
import pywt

def add_wavelet_channels(sequences, wavelet='db4', level=3):
    """Extract wavelet coefficients at multiple scales."""
    batch, channels, length = sequences.shape
    wavelet_channels = []

    for i in range(batch):
        sample_wavelets = []
        for ch in range(4):  # Only for raw signals (EDA, TEMP, ACC, BVP)
            signal = sequences[i, ch, :]
            coeffs = pywt.wavedec(signal, wavelet, level=level)

            for coeff in coeffs:
                if len(coeff) < length:
                    coeff = np.pad(coeff, (0, length - len(coeff)), mode='edge')
                elif len(coeff) > length:
                    coeff = coeff[:length]
                sample_wavelets.append(coeff)

        wavelet_channels.append(np.array(sample_wavelets))

    wavelet_channels = np.array(wavelet_channels)
    enhanced = np.concatenate([sequences, wavelet_channels], axis=1)
    return enhanced

# Results in 16 + (4 signals × 4 levels) = 32 channels
```

**Files to modify:**
- `LSTM_ResNet.ipynb` cell `a4fa9274` (post-processing)

---

### 3.2 Respiratory Features from BVP
**Impact: +1-2% macro F1**

**Extract respiratory-induced BVP variations:**
```python
from scipy.signal import butter, filtfilt, welch

def extract_respiratory_channel(bvp, fs=4.0):
    """Extract respiratory sinus arrhythmia from BVP."""
    # Bandpass filter for respiratory band (0.15-0.4 Hz)
    nyquist = fs / 2
    low, high = 0.15 / nyquist, 0.4 / nyquist
    b, a = butter(4, [low, high], btype='band')
    bvp_respiratory = filtfilt(b, a, bvp)

    # Power spectral density
    freqs, psd = welch(bvp, fs=fs, nperseg=min(128, len(bvp)))
    resp_mask = (freqs >= 0.15) & (freqs <= 0.4)
    resp_power = np.trapz(psd[resp_mask], freqs[resp_mask])

    return bvp_respiratory, resp_power
```

**Add as channel 33**

**Files to modify:**
- `LSTM_ResNet.ipynb` cell `83737a60` (feature extraction)

---

### 3.3 HRV Nonlinear Features
**Impact: +1-2% macro F1**

**Extract complexity metrics from BVP:**
```python
def extract_hrv_nonlinear_channel(bvp, fs=4.0):
    """Extract Sample Entropy as temporal channel."""
    from scipy.signal import find_peaks

    # Detect BVP peaks (R-waves)
    peaks, _ = find_peaks(bvp, distance=int(0.5*fs), prominence=0.5*np.std(bvp))

    if len(peaks) < 10:
        return np.zeros_like(bvp)

    # Inter-beat intervals
    ibi = np.diff(peaks) / fs

    # Sample Entropy (rolling window)
    window_size = 10
    sampen_values = []
    for i in range(len(bvp)):
        # Find IBIs in current window
        window_start = max(0, i - int(window_size * fs))
        window_peaks = peaks[(peaks >= window_start) & (peaks < i)]

        if len(window_peaks) >= 5:
            window_ibi = np.diff(window_peaks) / fs
            sampen = sample_entropy(window_ibi, m=2, r=0.2)
        else:
            sampen = 0.0

        sampen_values.append(sampen)

    return np.array(sampen_values)

def sample_entropy(data, m=2, r=0.2):
    """Calculate Sample Entropy."""
    N = len(data)
    r = r * np.std(data)

    def _maxdist(xi, xj):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])

    def _phi(m):
        patterns = np.array([[data[j] for j in range(i, i + m)]
                           for i in range(N - m)])
        count = 0
        for i in range(len(patterns)):
            for j in range(len(patterns)):
                if i != j and _maxdist(patterns[i], patterns[j]) <= r:
                    count += 1
        return count / (N - m) / (N - m - 1) if (N - m) > 1 else 0

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    return -np.log(phi_m1 / phi_m) if phi_m > 0 and phi_m1 > 0 else 0.0
```

**Add as channel 34**

**Files to modify:**
- `LSTM_ResNet.ipynb` cell `83737a60` (feature extraction)

---

## Phase 4: Hybrid Model (Expected: +3-5% macro F1)
**Timeline: Week 4 | Target: 60-65% macro F1**

### 4.1 Feature-Sequence Hybrid Architecture
**Impact: +2-3% macro F1**

**Combine sequence learning with engineered features:**

**Extract statistical features per window:**
```python
def extract_window_features(eda, temp, acc, bvp):
    """Extract 50 statistical features per window."""
    features = []

    for signal, name in [(eda, 'eda'), (temp, 'temp'), (acc, 'acc'), (bvp, 'bvp')]:
        # Time-domain stats
        features.extend([
            np.mean(signal),
            np.std(signal),
            np.min(signal),
            np.max(signal),
            np.median(signal),
            np.percentile(signal, 25),
            np.percentile(signal, 75),
            stats.skew(signal),
            stats.kurtosis(signal),
        ])

        # Derivative stats
        diff = np.diff(signal)
        features.extend([
            np.mean(diff),
            np.std(diff),
            np.max(np.abs(diff)),
        ])

    # Total: 4 signals × 12 features = 48 features
    # Add 2 cross-correlation features
    features.append(np.corrcoef(eda, bvp)[0, 1])
    features.append(np.corrcoef(eda, acc)[0, 1])

    return np.array(features)
```

**Hybrid model:**
```python
class HybridStressNet(nn.Module):
    def __init__(self, input_channels=34, num_features=50, num_classes=4):
        super().__init__()

        # Sequence branch (Multi-scale LSTM ResNet)
        self.sequence_branch = MultiScaleLSTMResNet(input_channels, num_classes)
        self.sequence_branch.classifier = nn.Identity()  # Remove classifier

        # Feature branch (MLP)
        self.feature_branch = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(384 + 64, 256),  # 256+128 from sequences + 64 from features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, sequences, features):
        seq_feat = self.sequence_branch(sequences)
        stat_feat = self.feature_branch(features)
        combined = torch.cat([seq_feat, stat_feat], dim=1)
        return self.fusion(combined)
```

**Files to modify:**
- `LSTM_ResNet.ipynb` new cell after `cb4a1b2d`
- `LSTM_ResNet.ipynb` cell `83737a60` (add feature extraction)
- `LSTM_ResNet.ipynb` cell `8869da7c` (modify training loop)

---

### 4.2 Two-Stage Prediction Strategy
**Impact: +1-2% macro F1**

**Adopt XGBoost's successful two-stage approach:**

Stage 1: Binary stress detection (stress vs no_stress)
Stage 2: Multi-class classification on stress samples

```python
def two_stage_prediction(model, sequences, features, threshold=0.4):
    """Two-stage prediction for better minority class handling."""
    model.eval()
    with torch.no_grad():
        logits = model(sequences, features)
        probs = F.softmax(logits, dim=1)

        # Stage 1: Stress probability
        stress_prob = probs[:, :3].sum(dim=1)  # low + moderate + high
        is_stress = stress_prob >= threshold

    # Stage 2: Classify stress level
    predictions = torch.full((len(sequences),), 3, dtype=torch.long)  # Default: no_stress

    if is_stress.sum() > 0:
        stress_indices = torch.where(is_stress)[0]
        stress_probs = probs[stress_indices, :3]
        stress_classes = torch.argmax(stress_probs, dim=1)
        predictions[stress_indices] = stress_classes

    return predictions
```

**Files to modify:**
- `LSTM_ResNet.ipynb` cell `8869da7c` (evaluation section)

---

## Phase 5: Final Optimizations (Expected: +2-4% macro F1)
**Timeline: Week 4 | Target: 62-67% macro F1**

### 5.1 Phase-Aware Conditioning
**Impact: +1-2% macro F1**

**Add experimental phase as input:**

```python
# Phase encoding
PHASE_ENCODING = {
    'Baseline': 0,
    'Stroop': 1,
    'TMCT': 2,
    'Real Opinion': 3,
    'Opposite Opinion': 4,
    'Subtract': 5,
    'rest': 6,
    'active': 7,
    'aerobic': 7,
    'anaerobic': 7,
}

class PhaseAwareHybridNet(nn.Module):
    def __init__(self, input_channels=34, num_features=50, num_phases=8, num_classes=4):
        super().__init__()

        # Phase embedding
        self.phase_embedding = nn.Embedding(num_phases, 32)

        # Sequence + feature branches (same as HybridStressNet)
        self.sequence_branch = MultiScaleLSTMResNet(input_channels, num_classes)
        self.sequence_branch.classifier = nn.Identity()

        self.feature_branch = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Fusion with phase conditioning
        self.fusion = nn.Sequential(
            nn.Linear(384 + 64 + 32, 256),  # Add phase embedding
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, sequences, features, phase_ids):
        seq_feat = self.sequence_branch(sequences)
        stat_feat = self.feature_branch(features)
        phase_emb = self.phase_embedding(phase_ids)

        combined = torch.cat([seq_feat, stat_feat, phase_emb], dim=1)
        return self.fusion(combined)
```

**Files to modify:**
- `LSTM_ResNet.ipynb` cell `83737a60` (add phase tracking)
- `LSTM_ResNet.ipynb` new architecture cell

---

### 5.2 Hyperparameter Tuning
**Impact: +1-2% macro F1**

**Grid search on validation set:**
- Learning rate: [5e-4, 1e-3, 2e-3]
- Batch size: [16, 32, 64]
- Dropout: [0.2, 0.3, 0.4]
- LSTM hidden: [96, 128, 160]
- CNN channels: [48, 64, 96]
- Focal loss gamma: [1.5, 2.0, 2.5]

**Files to modify:**
- `LSTM_ResNet.ipynb` new tuning cell

---

## Implementation Timeline

| Week | Phase | Target Macro F1 | Key Milestones |
|------|-------|----------------|----------------|
| 1 | Phase 1 | 45-48% | Subject normalization, enhanced channels, temporal augmentation |
| 2 | Phase 2 | 52-58% | Multi-scale CNN, attention, focal loss, improved training |
| 3 | Phase 3 | 56-62% | Wavelets, respiratory features, HRV nonlinear |
| 4 | Phase 4-5 | 60-67% | Hybrid model, two-stage prediction, phase conditioning, tuning |

---

## Critical Files to Modify

### Primary file:
- **`/home/moh/home/Data_mining/Stress-Level-Prediction/LSTM_ResNet.ipynb`**
  - Cell `83737a60`: Feature extraction in `build_sequence_dataset()`
  - Cell `a4fa9274`: Post-processing (normalization, channel engineering)
  - Cell `cb4a1b2d`: Model architecture (replace with MultiScaleLSTMResNet)
  - Cell `6d812a0e`: Training configuration
  - Cell `8869da7c`: Training loop

### Reference files (read-only):
- `/home/moh/home/Data_mining/Stress-Level-Prediction/Complete_Standalone_Pipeline.ipynb` (feature extraction examples)
- `/home/moh/home/Data_mining/Stress-Level-Prediction/Code.ipynb` (two-stage strategy reference)

---

## Expected Performance Gains

| Metric | Current | Phase 1 | Phase 2 | Phase 3 | Phase 4-5 | Total Gain |
|--------|---------|---------|---------|---------|-----------|------------|
| **Macro F1** | 36% | 45-48% | 52-58% | 56-62% | **60-67%** | **+24-31pp** |
| **Accuracy** | 76% | 80-82% | 83-86% | 85-88% | **87-90%** | **+11-14pp** |
| **High-stress recall** | 21% | 35-40% | 45-55% | 55-65% | **65-75%** | **+44-54pp** |
| **Low-stress recall** | 0% | 15-25% | 30-40% | 40-50% | **50-60%** | **+50-60pp** |

---

## Success Criteria

### Minimum acceptable:
- Macro F1 ≥ 55%
- All classes have recall ≥ 30%

### Target:
- Macro F1 ≥ 60%
- High-stress recall ≥ 60%
- Low-stress recall ≥ 40%

### Stretch goal:
- Macro F1 ≥ 65%
- All classes balanced recall (±10pp)
- Match or exceed XGBoost accuracy (91%)

---

## Risk Mitigation

### Risk: Overfitting with more channels
**Mitigation:**
- Start with fewer channels (16) before adding wavelets (34)
- Monitor train/val gap
- Increase dropout if needed

### Risk: LSTM capacity insufficient
**Mitigation:**
- Increase hidden units: 64 → 128
- Add 3rd LSTM layer
- Use attention to focus on key timesteps

### Risk: Longer training time
**Mitigation:**
- Use mixed precision (2x speedup)
- Reduce batch size gradually
- Early stopping if no improvement

---

## Next Steps After Planning

1. **User approval** of phased approach
2. **Confirm priorities**: All phases or subset?
3. **Baseline run**: Record current exact metrics for comparison
4. **Phase 1 implementation**: Start with highest-impact changes
5. **Iterative validation**: Test after each phase before proceeding
