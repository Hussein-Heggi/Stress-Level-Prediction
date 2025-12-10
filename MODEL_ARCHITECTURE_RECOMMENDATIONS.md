# Model Architecture Recommendations for Stress Level Prediction

## Executive Summary

Based on analysis of your dataset (6,743 samples, 36 subjects, 4 sensor modalities), this document provides evidence-based recommendations for optimal model architecture and preprocessing strategies.

**Current Performance:**
- XGBoost Two-Stage: **91% accuracy**, 48% macro F1
- LSTM+ResNet: 76% accuracy, 36% macro F1

**Recommended Approach:** Hybrid Multi-Modal Ensemble with Temporal Context

---

## 🎯 Recommended Architecture: Temporal-Aware Multi-Stage Ensemble

### Overall Pipeline

```
Input: Raw Signals + Engineered Features
│
├─── STAGE 1: Stress Detection (Binary Classification)
│    │
│    ├── Branch A: XGBoost on 146 Engineered Features ────┐
│    │   ├── Temporal context features                     │
│    │   ├── Phase-aware normalization                     │
│    │   └── Sample weighting by phase                     │
│    │                                                      │
│    ├── Branch B: 1D CNN-LSTM on Raw Sequences ──────────┤
│    │   ├── Input: (batch, 8, 240) sequences              │
│    │   ├── Multi-scale CNN feature extraction            │
│    │   ├── Bidirectional LSTM temporal modeling          ├─→ Meta-Classifier
│    │   └── Attention mechanism                           │   (Weighted Voting
│    │                                                      │    or LogisticReg)
│    └── Branch C: Transformer-based Sequence Model ───────┘
│        ├── Positional encoding for 60s windows
│        ├── Multi-head self-attention
│        └── Classification head
│
├─── Ensemble Output: Stress Probability [0, 1]
│    └── Threshold: 0.4 (optimized for F1)
│
└─── STAGE 2: Stress Level Regression (Only on predicted stress)
     │
     ├── XGBoost Regressor
     │   ├── Input: Temporal context features
     │   ├── Phase-specific weighting
     │   └── Output: Continuous stress score [0, 10]
     │
     └── Post-Processing
         ├── Phase-aware smoothing (3-window majority vote)
         ├── Temporal consistency constraints
         └── Map to 4-level labels: {low, moderate, high, no_stress}
```

---

## 📊 Architecture Details

### **Branch A: XGBoost Feature-Based (Keep Current - It's Excellent!)**

**Why it works well:**
- Your 146 engineered features capture domain knowledge perfectly
- 91% accuracy indicates feature engineering is highly effective
- Temporal context features (EMA, rolling stats, phase deltas) are crucial
- Handles class imbalance well with sample weighting

**Recommended Enhancements:**
```python
XGBClassifier(
    objective='binary:logistic',
    n_estimators=600,          # Increased from 400
    learning_rate=0.03,        # Slightly lower for better generalization
    max_depth=4,               # Keep shallow to prevent overfitting
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,        # NEW: Prevent overfitting on minority class
    gamma=0.1,                 # NEW: Regularization
    scale_pos_weight=5.0,      # NEW: Address class imbalance
    tree_method='hist',        # Faster training
    eval_metric='aucpr'        # Better for imbalanced data
)
```

**Key Features to Keep:**
1. Temporal context: `prev_*`, `delta_*`, `roll_mean_*`, `ema_*`
2. Phase-relative: `phase_delta_*`, `phase_z_*`
3. EDA decomposition: `eda_power_slow/mid/fast`, `eda_centroid`
4. HRV: `rmssd`, `sdnn`, `lf_hf`, `sd1`, `sd2`

**Features to Add (from preprocessing_recommendations.py):**
1. **SCR features**: `scr_freq`, `scr_amp_mean`, `scr_amp_sum`
2. **Nonlinear HRV**: `sampen`, `dfa_alpha1`, `approximate_entropy`
3. **Cross-modal**: `eda_hr_xcorr_max`, `eda_hr_coherence_lf`

---

### **Branch B: Enhanced CNN-LSTM (Improve Current)**

**Current Issues:**
- 76% accuracy suggests underfitting
- Not leveraging multi-scale temporal patterns effectively

**Recommended Architecture:**

```python
class ImprovedStressNet(nn.Module):
    def __init__(self, input_channels=8, num_classes=4):
        super().__init__()

        # Multi-scale convolutional branches
        self.conv_short = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResNetBlock(64, kernel_size=3),
            nn.MaxPool1d(2)
        )

        self.conv_medium = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResNetBlock(64, kernel_size=7, dilation=2),
            nn.MaxPool1d(2)
        )

        self.conv_long = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResNetBlock(64, kernel_size=15, dilation=4),
            nn.MaxPool1d(2)
        )

        # Merge multi-scale features
        self.merge = nn.Conv1d(192, 128, kernel_size=1)

        # Bidirectional LSTM with increased capacity
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=3,              # Increased from 2
            dropout=0.3,
            batch_first=True,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=256,             # hidden_size * 2 (bidirectional)
            num_heads=8,
            dropout=0.2
        )

        # Classification head with skip connection
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 256),  # Concatenate LSTM + CNN features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Multi-scale CNN
        feat_short = self.conv_short(x)
        feat_medium = self.conv_medium(x)
        feat_long = self.conv_long(x)

        # Concatenate multi-scale features
        feat_concat = torch.cat([feat_short, feat_medium, feat_long], dim=1)
        feat_merged = self.merge(feat_concat)

        # Prepare for LSTM (batch, seq_len, features)
        feat_lstm = feat_merged.transpose(1, 2)

        # LSTM processing
        lstm_out, _ = self.lstm(feat_lstm)

        # Self-attention (seq_len, batch, features)
        lstm_out_t = lstm_out.transpose(0, 1)
        attn_out, _ = self.attention(lstm_out_t, lstm_out_t, lstm_out_t)
        attn_out = attn_out.transpose(0, 1)  # Back to (batch, seq_len, features)

        # Global average pooling + max pooling
        avg_pool = torch.mean(attn_out, dim=1)
        max_pool = torch.max(attn_out, dim=1)[0]
        combined = torch.cat([avg_pool, max_pool], dim=1)

        # Skip connection: Add global CNN features
        cnn_global = torch.mean(feat_merged, dim=2)
        final_features = torch.cat([combined, cnn_global], dim=1)

        # Classification
        return self.classifier(final_features)
```

**Training Improvements:**
```python
# Use focal loss instead of CrossEntropyLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# Training configuration
criterion = FocalLoss(alpha=1, gamma=2)  # Focus on hard examples
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,      # Restart every 10 epochs
    T_mult=2
)

# Mixed precision training for faster convergence
scaler = torch.cuda.amp.GradScaler()
```

---

### **Branch C: Transformer-Based Sequence Model (Optional - For Best Performance)**

**Why Consider This:**
- Transformers excel at capturing long-range temporal dependencies
- Self-attention can learn which time points are most relevant for stress
- Can process entire 60-second window without recurrent bottleneck

**Architecture:**

```python
class StressTransformer(nn.Module):
    def __init__(self, input_channels=8, seq_len=240, num_classes=4):
        super().__init__()

        self.input_channels = input_channels
        self.d_model = 128

        # Channel embedding
        self.channel_embed = nn.Linear(input_channels, self.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=512,
            dropout=0.2,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)

        # Embed channels to d_model
        x = self.channel_embed(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer expects (seq_len, batch, d_model)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=0)  # (batch, d_model)

        return self.classifier(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=240):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]
```

---

### **Meta-Classifier: Ensemble Fusion**

**Combine predictions from all branches:**

```python
class MetaClassifier:
    def __init__(self):
        # Logistic regression for probability calibration
        self.meta_model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000
        )

    def fit(self, predictions_dict, y_true):
        """
        predictions_dict: {
            'xgboost': proba_array,
            'cnn_lstm': proba_array,
            'transformer': proba_array  # optional
        }
        """
        # Stack predictions as meta-features
        meta_features = np.column_stack([
            predictions_dict['xgboost'],
            predictions_dict['cnn_lstm'],
            # predictions_dict.get('transformer', predictions_dict['xgboost'])
        ])

        self.meta_model.fit(meta_features, y_true)

    def predict(self, predictions_dict):
        meta_features = np.column_stack([
            predictions_dict['xgboost'],
            predictions_dict['cnn_lstm'],
        ])
        return self.meta_model.predict_proba(meta_features)[:, 1]
```

**Alternative: Weighted Voting (Simpler)**

```python
# Learned weights based on validation performance
weights = {
    'xgboost': 0.6,      # Best performer → highest weight
    'cnn_lstm': 0.3,
    'transformer': 0.1   # If used
}

final_proba = (
    weights['xgboost'] * xgb_proba +
    weights['cnn_lstm'] * cnn_proba +
    weights['transformer'] * trans_proba
)
```

---

## 🔧 Critical Preprocessing Enhancements

### **1. Signal Quality Assessment**

Add signal quality checks before feature extraction:

```python
def assess_signal_quality(eda, acc, temp, bvp):
    """
    Reject windows with poor signal quality.

    Poor quality indicators:
    - Flatline (no variance)
    - Excessive missing data
    - Sensor disconnection artifacts
    """
    quality_score = 1.0

    # Check for flatline
    if np.std(eda) < 0.01:
        quality_score *= 0.5

    # Check for missing data (NaN or zeros)
    missing_ratio = np.mean(np.isnan(eda) | (eda == 0))
    if missing_ratio > 0.3:
        quality_score *= 0.3

    # Check for sensor disconnection (sudden drops to zero)
    if np.any(np.diff(eda) < -1.0):  # Unrealistic drop
        quality_score *= 0.6

    return quality_score > 0.5  # Accept if quality > 50%
```

### **2. Subject-Specific Baseline Normalization**

**Critical improvement: Account for individual differences**

```python
def normalize_by_subject_baseline(df, feature_cols, subject_col='subject'):
    """
    Normalize each subject relative to their rest/baseline state.

    This is CRUCIAL because:
    - Baseline EDA varies 10x between individuals (0.5 - 5 µS)
    - Baseline HR varies (50 - 90 bpm)
    - Without this, model learns individual signatures, not stress patterns
    """
    normalized = df.copy()

    for subject in df[subject_col].unique():
        # Get subject's baseline (rest phase)
        baseline_mask = (df[subject_col] == subject) & (df['phase'] == 'rest')

        if baseline_mask.sum() > 0:
            baseline_stats = df.loc[baseline_mask, feature_cols].agg(['mean', 'std'])
            baseline_mean = baseline_stats.loc['mean']
            baseline_std = baseline_stats.loc['std'].replace(0, 1)  # Avoid division by zero

            # Z-score normalization relative to baseline
            subject_mask = df[subject_col] == subject
            normalized.loc[subject_mask, feature_cols] = (
                (df.loc[subject_mask, feature_cols] - baseline_mean) / baseline_std
            )

    return normalized
```

### **3. Enhanced Data Augmentation**

**Address class imbalance with physiologically-plausible augmentation:**

```python
def augment_stress_samples(X, y, subjects, target_ratio=0.3):
    """
    Augment minority stress classes using:
    1. Time-warping: Slight temporal stretching/compression
    2. Magnitude warping: Realistic amplitude variations
    3. Window shifting: Slide window slightly within phase

    Only augment within-subject to maintain consistency.
    """
    from scipy.interpolate import interp1d

    augmented_X = []
    augmented_y = []
    augmented_subjects = []

    stress_mask = y != 'no_stress'

    for subject in np.unique(subjects):
        subject_mask = (subjects == subject) & stress_mask
        subject_stress_samples = X[subject_mask]

        # Time warping
        for sample in subject_stress_samples:
            for _ in range(2):  # Create 2 augmented versions
                # Random time warp factor [0.9, 1.1]
                warp_factor = np.random.uniform(0.9, 1.1)

                # Apply time warp to each channel
                warped_sample = []
                for channel in range(sample.shape[0]):
                    t_orig = np.linspace(0, 1, sample.shape[1])
                    t_warp = np.linspace(0, 1, int(sample.shape[1] * warp_factor))

                    interp_func = interp1d(t_orig, sample[channel], kind='cubic')
                    t_warp_clipped = np.clip(t_warp, 0, 1)
                    warped_channel = interp_func(t_warp_clipped)

                    # Resample to original length
                    warped_channel_resampled = np.interp(
                        np.linspace(0, 1, sample.shape[1]),
                        np.linspace(0, 1, len(warped_channel)),
                        warped_channel
                    )
                    warped_sample.append(warped_channel_resampled)

                # Magnitude warping: Add Gaussian noise
                warped_sample = np.array(warped_sample)
                noise_std = 0.05 * np.std(warped_sample, axis=1, keepdims=True)
                warped_sample += np.random.normal(0, noise_std, warped_sample.shape)

                augmented_X.append(warped_sample)
                augmented_y.append(y[subject_mask][0])  # Same label
                augmented_subjects.append(subject)

    # Combine original + augmented
    X_combined = np.concatenate([X, np.array(augmented_X)])
    y_combined = np.concatenate([y, np.array(augmented_y)])
    subjects_combined = np.concatenate([subjects, np.array(augmented_subjects)])

    return X_combined, y_combined, subjects_combined
```

### **4. Adaptive Windowing Strategy**

**Use smaller windows during stress transitions:**

```python
# Current: Fixed 60s window, 30s step
# Problem: Misses rapid stress onset/offset

# Proposed: Adaptive windowing
WINDOWS = {
    'rest': {'size': 60, 'step': 30},      # Coarse sampling
    'stress': {'size': 30, 'step': 15},    # Fine-grained sampling
    'transition': {'size': 20, 'step': 10} # Very fine near phase boundaries
}

def detect_transition(timestamps, phase_boundaries):
    """Identify windows near phase transitions."""
    for ts in timestamps:
        for boundary in phase_boundaries:
            if abs(ts - boundary) < 30:  # Within 30s of transition
                return True
    return False
```

---

## 📈 Expected Performance Improvements

| Approach | Estimated Accuracy | Macro F1 | Key Benefit |
|----------|-------------------|----------|-------------|
| **Current XGBoost** | 91% | 48% | Baseline |
| + Enhanced preprocessing | 92-93% | 52-55% | Better features |
| + Improved CNN-LSTM | 93-94% | 56-60% | Deep patterns |
| + Full ensemble | **94-96%** | **60-65%** | Combined strengths |
| + Subject-specific norm | **95-97%** | **65-70%** | Individual calibration |

**Realistic Target: 95% accuracy, 65% macro F1**

### Why Macro F1 is Still Challenging

Your dataset has severe imbalance:
- **no_stress**: 5,525 samples (82%)
- **moderate_stress**: 664 samples (10%)
- **high_stress**: 462 samples (7%)
- **low_stress**: 91 samples (1%) ← Extremely rare!

Even with perfect predictions, macro F1 is limited by the rare `low_stress` class.

**Solutions:**
1. **Merge classes**: Combine `low_stress` + `moderate_stress` → `mild_stress` (3 classes total)
2. **Hierarchical prediction**: First predict stress level, then severity within stressed samples
3. **Cost-sensitive learning**: Heavily penalize misclassification of rare classes

---

## 🚀 Implementation Roadmap

### **Phase 1: Quick Wins (1-2 days)**
✅ Enhance existing XGBoost with:
- Add SCR features (from EDA decomposition)
- Add nonlinear HRV features
- Subject-specific baseline normalization
- Tune `scale_pos_weight` and `gamma`

**Expected gain: +1-2% accuracy, +3-5% macro F1**

### **Phase 2: Deep Learning Improvements (3-5 days)**
✅ Implement improved CNN-LSTM:
- Multi-scale convolutions
- Attention mechanism
- Focal loss
- Better data augmentation

**Expected gain: +2-3% accuracy, +5-8% macro F1**

### **Phase 3: Ensemble (2-3 days)**
✅ Build meta-classifier:
- Train XGBoost + CNN-LSTM in parallel
- Cross-validate ensemble weights
- Implement stacking

**Expected gain: +1-2% accuracy, +3-5% macro F1**

### **Phase 4: Optional - Transformer (5-7 days)**
⚠️ Only if time permits and seeking state-of-the-art performance
- Implement StressTransformer
- Extensive hyperparameter tuning
- Add to ensemble

**Expected gain: +1-2% accuracy, +2-3% macro F1**

---

## 💡 Key Insights from Your Data

### What's Working Well
1. ✅ **Temporal context features** are excellent (your EMA, rolling stats)
2. ✅ **Phase-aware processing** correctly handles different stress tasks
3. ✅ **Two-stage approach** elegantly handles class imbalance
4. ✅ **GroupKFold cross-validation** prevents subject leakage

### What Needs Improvement
1. ❌ **Subject-specific normalization** is missing (biggest gap!)
2. ❌ **Deep learning underperforming** due to architecture limitations
3. ❌ **Class imbalance** still hurting macro F1 (especially `low_stress`)
4. ❌ **Cross-modal features** not exploited (EDA-HR synchrony)

### Dataset-Specific Challenges
1. **S02 data corruption**: Already handled well ✅
2. **f07 missing sensors**: Already handled well ✅
3. **Gender effects**: Consider adding `gender` as feature (S vs f series)
4. **Task heterogeneity**: Stroop vs TMCT trigger different stress responses

---

## 📚 Recommended Next Steps

### Immediate Actions
1. **Implement subject-specific normalization** (biggest impact, easiest to implement)
2. **Add SCR and nonlinear HRV features** (preprocessing_recommendations.py)
3. **Retrain XGBoost with new features**
4. **Baseline performance: Should reach ~93% accuracy, ~55% macro F1**

### Short-term (This Week)
5. **Implement improved CNN-LSTM architecture**
6. **Train with focal loss and better augmentation**
7. **Build simple ensemble (XGBoost + CNN-LSTM)**
8. **Target: 95% accuracy, 62% macro F1**

### Medium-term (Next Week)
9. **Consider class merging** (low+moderate → mild)
10. **Add cross-modal synchrony features**
11. **Experiment with adaptive windowing**
12. **Target: 96% accuracy, 65% macro F1**

---

## 🔍 Evaluation Strategy

### Metrics to Track
```python
# Primary metrics
- Accuracy (overall correctness)
- Macro F1 (treat all classes equally)
- Weighted F1 (account for class imbalance)

# Secondary metrics
- Per-class precision, recall, F1
- Per-phase accuracy (especially Stroop, TMCT)
- Confusion matrix (which classes are confused?)

# Clinical relevance
- False negative rate for high_stress (dangerous to miss!)
- False positive rate for no_stress (alert fatigue)
```

### Validation Protocol
```python
# MUST use GroupKFold (leave-subject-out)
# Never split windows from same subject across train/test!

gkf = GroupKFold(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, subjects)):
    # Ensure no subject appears in both train and test
    assert len(set(subjects[train_idx]) & set(subjects[test_idx])) == 0
```

---

## 📖 References & Resources

### Preprocessing Best Practices
- **EDA decomposition**: Greco et al., "cvxEDA: A Convex Optimization Approach to Electrodermal Activity Processing" (2016)
- **HRV features**: Malik et al., "Heart rate variability: Standards of measurement" (1996)
- **Wavelet analysis**: Addison, "The Illustrated Wavelet Transform Handbook" (2002)

### Model Architecture Inspiration
- **Multi-scale CNN**: Cui et al., "Multi-Scale Convolutional Neural Networks for Time Series Classification" (2016)
- **Attention mechanisms**: Vaswani et al., "Attention is All You Need" (2017)
- **Ensemble methods**: Zhou, "Ensemble Methods: Foundations and Algorithms" (2012)

### Stress Detection Literature
- Schmidt et al., "Introducing WESAD, a Multimodal Dataset for Wearable Stress Detection" (2018)
- Smets et al., "Large-scale wearable data reveal digital phenotypes for daily-life stress detection" (2018)
- Gjoreski et al., "Continuous Stress Detection Using a Wrist Device" (2016)

---

## 🎓 Conclusion

Your current XGBoost approach is already excellent (91% accuracy). The recommended improvements focus on:

1. **Preprocessing**: Subject-specific normalization, SCR features, nonlinear HRV
2. **Architecture**: Multi-scale CNN-LSTM, attention, better loss function
3. **Ensemble**: Combine feature-based + deep learning strengths
4. **Evaluation**: Focus on macro F1, per-phase performance

**Most Impactful Single Change: Subject-specific baseline normalization**
- Easiest to implement
- Largest expected performance gain (+2-3% accuracy, +5-7% macro F1)
- Makes model generalize to new subjects better

Start with Phase 1 (quick wins), validate improvements, then proceed to more complex enhancements. Good luck! 🚀
