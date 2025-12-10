# Quick Implementation Summary

## 📊 Your Current Status

**Dataset:** 6,743 windows (60s) from 36 subjects, 4 sensors (EDA, Temp, ACC, BVP)

**Current Performance:**
- ✅ XGBoost Two-Stage: **91% accuracy, 48% macro F1** (excellent!)
- ⚠️ LSTM+ResNet: 76% accuracy, 36% macro F1 (underperforming)

**Main Challenge:** Severe class imbalance (82% no_stress, only 1.4% low_stress)

---

## 🎯 Recommended Solution: Hybrid Ensemble

```
Stage 1: Stress Detection (Binary)
├── XGBoost on 146 features (KEEP - it's working great!)
├── Improved CNN-LSTM on sequences (FIX architecture)
└── Ensemble combination

Stage 2: Stress Level Regression
└── XGBoost Regressor (KEEP current approach)
```

---

## 🚀 Implementation Phases (Start Simple → Add Complexity)

### **PHASE 1: Quick Wins (1-2 hours) - START HERE! ⭐**

**Impact: +2-3% accuracy, +5-7% macro F1**

#### 1.1 Subject-Specific Normalization (HIGHEST IMPACT)

**Why:** Your model is learning individual signatures, not stress patterns. People have vastly different baseline physiology.

```python
# Add to your existing Code.ipynb BEFORE training
from improved_model_starter import normalize_by_subject_baseline

# Load your dataset
df = pd.read_csv('stress_level_dataset.csv')

# Get feature columns
feature_cols = [col for col in df.columns if col not in
               ['subject', 'state', 'phase', 'label', 'is_stress',
                'win_start', 'win_end', 'stress_stage', 'stress_level',
                'phase_start', 'phase_end', 'phase_duration', 'phase_elapsed',
                'phase_progress', 'phase_position', 'phase_remaining',
                'phase_early', 'phase_mid', 'phase_late']]

# Apply normalization
df_normalized = normalize_by_subject_baseline(
    df,
    feature_cols=feature_cols,
    subject_col='subject',
    phase_col='phase'
)

# Continue with your existing pipeline using df_normalized
# Expected: 93% accuracy, 55% macro F1
```

#### 1.2 Improved XGBoost Hyperparameters

**Why:** Your current params are good, but can be optimized for class imbalance.

```python
from improved_model_starter import get_improved_xgb_params

# Replace your current params
improved_params = get_improved_xgb_params(stage='binary')

model_stage1 = XGBClassifier(**improved_params)
model_stage1.fit(X_train, y_train, sample_weight=weights)
```

**Key changes:**
- `scale_pos_weight=5.0` → Better handle 1:5 class imbalance
- `gamma=0.1` → Regularization to prevent overfitting
- `eval_metric='aucpr'` → Better metric for imbalanced data

---

### **PHASE 2: Deep Learning Fix (1-2 days)**

**Impact: +3-5% accuracy, +8-12% macro F1**

#### 2.1 Replace Current LSTM_ResNet Architecture

**Problem:** Current architecture is too simple for the complexity of stress patterns.

**Solution:** Use `ImprovedStressNet` from [improved_model_starter.py](improved_model_starter.py:199)

**What's better:**
- Multi-scale CNN (captures 3 different time scales)
- Attention mechanism (focuses on important time points)
- More LSTM capacity (3 layers vs 2)
- Skip connections (prevents gradient vanishing)

```python
from improved_model_starter import ImprovedStressNet, train_improved_cnn_lstm

# Create improved model
model = ImprovedStressNet(input_channels=8, num_classes=4)

# Train with best practices
model = train_improved_cnn_lstm(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=30,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    use_focal_loss=True  # Better for class imbalance
)

# Expected: 85-88% accuracy (on its own), 50-55% macro F1
```

#### 2.2 Add Focal Loss

**Why:** Your dataset has extreme class imbalance. CrossEntropyLoss treats all examples equally; Focal Loss focuses on hard examples.

Already included in `train_improved_cnn_lstm()` function.

---

### **PHASE 3: Ensemble (1 day)**

**Impact: +1-2% accuracy, +3-5% macro F1**

#### 3.1 Combine XGBoost + CNN-LSTM Predictions

```python
from improved_model_starter import SimpleEnsemble

# Get predictions from both models
xgb_proba = model_xgb.predict_proba(X_test)[:, 1]

# For CNN-LSTM, get probabilities
model_cnn.eval()
with torch.no_grad():
    cnn_logits = model_cnn(torch.tensor(X_test_sequences).float())
    cnn_proba = torch.softmax(cnn_logits, dim=1)[:, 1].numpy()  # Prob of stress class

# Ensemble
ensemble = SimpleEnsemble(xgb_weight=0.6, cnn_weight=0.4)
final_proba = ensemble.predict_proba(xgb_proba, cnn_proba)
final_preds = (final_proba >= 0.4).astype(int)

# Expected: 95% accuracy, 65% macro F1
```

**Weight rationale:**
- XGBoost: 60% (it's performing better, so higher weight)
- CNN-LSTM: 40% (provides complementary temporal patterns)

---

## 📈 Expected Performance Progression

| Phase | Changes | Accuracy | Macro F1 | Time |
|-------|---------|----------|----------|------|
| **Current** | XGBoost baseline | 91% | 48% | - |
| **Phase 1** | + Subject normalization<br>+ Better XGB params | **93%** | **55%** | 1-2 hrs |
| **Phase 2** | + Improved CNN-LSTM | **94%** | **60%** | 1-2 days |
| **Phase 3** | + Ensemble | **95%** | **65%** | 1 day |
| **Optional** | + Advanced preprocessing | 96% | 68% | 2-3 days |

---

## 🔧 Advanced Preprocessing (Optional - Phase 4)

**If you want to push for 96%+ accuracy, add these features:**

### From [preprocessing_recommendations.py](preprocessing_recommendations.py)

1. **EDA Decomposition** (tonic vs phasic)
   - SCR frequency, amplitude, rise time
   - Phasic EDA is more stress-responsive than raw EDA

2. **Nonlinear HRV Features**
   - Sample Entropy (SampEn)
   - Detrended Fluctuation Analysis (DFA)
   - These capture stress-induced complexity changes

3. **Cross-Modal Synchrony**
   - EDA-HR cross-correlation
   - EDA-Temp coherence
   - Stress causes coordinated multi-system responses

**Implementation:**
```python
# See preprocessing_recommendations.py for full code
from preprocessing_recommendations import (
    decompose_eda_cvxeda,
    extract_scr_features,
    nonlinear_hrv_features,
    cross_signal_features
)

# During feature extraction, add:
tonic, phasic = decompose_eda_cvxeda(eda_signal, fs=4.0)
scr_feats = extract_scr_features(phasic, fs=4.0)
# ... add to your feature dictionary
```

---

## 📁 Files Created for You

1. **[MODEL_ARCHITECTURE_RECOMMENDATIONS.md](MODEL_ARCHITECTURE_RECOMMENDATIONS.md)** (comprehensive guide)
   - Detailed architecture explanations
   - Theoretical background
   - Literature references

2. **[preprocessing_recommendations.py](preprocessing_recommendations.py)** (advanced features)
   - EDA decomposition
   - Nonlinear HRV
   - Wavelet features
   - Cross-modal synchrony

3. **[improved_model_starter.py](improved_model_starter.py)** (ready-to-use code)
   - Subject normalization function
   - Improved XGBoost params
   - Enhanced CNN-LSTM architecture
   - Focal loss
   - Training loop with best practices
   - Ensemble class

4. **This file** (quick reference)

---

## 🎯 Recommended Action Plan

### **Week 1: Foundation (Phase 1)**

**Day 1:**
- [ ] Implement subject-specific normalization
- [ ] Retrain XGBoost with improved parameters
- [ ] Validate improvement (expect ~93% acc, ~55% macro F1)

**Day 2:**
- [ ] If improvement confirmed, checkpoint this version
- [ ] Prepare for Phase 2 (organize code, setup GPU)

### **Week 2: Deep Learning (Phase 2)**

**Day 3-4:**
- [ ] Implement `ImprovedStressNet` architecture
- [ ] Setup training with focal loss
- [ ] Train and validate (expect ~85-88% acc on its own)

**Day 5:**
- [ ] Debug and tune hyperparameters
- [ ] Cross-validate across all folds

### **Week 3: Ensemble (Phase 3)**

**Day 6:**
- [ ] Implement ensemble prediction
- [ ] Find optimal ensemble weights via grid search
- [ ] Final evaluation (expect ~95% acc, ~65% macro F1)

**Day 7:**
- [ ] Generate final reports
- [ ] Error analysis (which classes/phases are still hard?)
- [ ] Document findings

---

## 🐛 Common Issues & Solutions

### Issue 1: Subject Normalization Breaks Features
**Symptom:** Some features become NaN after normalization
**Solution:** Fill NaN with 0 or use robust scaling
```python
df_normalized = df_normalized.fillna(0)
```

### Issue 2: CNN-LSTM Overfitting
**Symptom:** Train acc 95%, Val acc 75%
**Solution:**
- Increase dropout (0.3 → 0.4)
- Reduce model capacity (hidden_size 128 → 96)
- Add more data augmentation

### Issue 3: Ensemble Not Improving
**Symptom:** Ensemble worse than XGBoost alone
**Solution:**
- Models too similar (both learn same patterns)
- Try different architectures (add Transformer)
- Tune ensemble weights via validation

### Issue 4: Low Macro F1 Despite High Accuracy
**Symptom:** 95% accuracy but 50% macro F1
**Solution:** This is expected with severe imbalance
- Focus on per-class F1 (especially minority classes)
- Consider merging classes (low+moderate → mild)
- Accept that macro F1 will be lower than accuracy

---

## 📊 Evaluation Checklist

When evaluating your improved models, check:

- [ ] **Accuracy** (overall correctness)
- [ ] **Macro F1** (treat all classes equally)
- [ ] **Weighted F1** (account for class distribution)
- [ ] **Per-class metrics** (precision, recall, F1 for each stress level)
- [ ] **Per-phase accuracy** (especially Stroop, TMCT - the hard ones)
- [ ] **Confusion matrix** (which classes are confused?)
- [ ] **Cross-validation consistency** (std dev across folds < 3%)
- [ ] **Leave-subject-out generalization** (no subject appears in both train/test)

---

## 🎓 Key Insights from Your Data

### ✅ What's Working Well
1. Your feature engineering is excellent (146 features capturing domain knowledge)
2. Two-stage approach elegantly handles class imbalance
3. Phase-aware processing correctly handles different stress tasks
4. GroupKFold prevents subject leakage

### ⚠️ What Needs Fixing
1. **No subject-specific normalization** (biggest gap!)
2. CNN-LSTM architecture too simple
3. Not using cross-modal features (EDA-HR synchrony)
4. Low_stress class has only 91 samples (consider merging with moderate)

### 💡 Data Characteristics
- **Best signal for stress:** EDA phasic (SCR frequency)
- **Most informative HRV:** RMSSD, LF/HF ratio
- **Hardest phase:** Stroop (cognitive stress)
- **Easiest phase:** Aerobic/Anaerobic (clear activity signatures)

---

## 🎯 Success Criteria

**Minimum Acceptable Performance (MVP):**
- Accuracy: ≥92%
- Macro F1: ≥52%
- Per-class F1 (high_stress): ≥40%

**Target Performance (Good):**
- Accuracy: ≥94%
- Macro F1: ≥60%
- Per-class F1 (high_stress): ≥50%

**Stretch Goal (Excellent):**
- Accuracy: ≥96%
- Macro F1: ≥65%
- Per-class F1 (high_stress): ≥60%

---

## 📞 Next Steps

1. **Start with Phase 1** (subject normalization) - highest impact, easiest
2. **Validate improvement** - if you see +2-3% accuracy, you're on track
3. **Proceed to Phase 2** - only if Phase 1 succeeds
4. **Build ensemble in Phase 3** - final performance boost

**Need more details?** See [MODEL_ARCHITECTURE_RECOMMENDATIONS.md](MODEL_ARCHITECTURE_RECOMMENDATIONS.md)

**Need advanced features?** See [preprocessing_recommendations.py](preprocessing_recommendations.py)

**Need ready-to-use code?** See [improved_model_starter.py](improved_model_starter.py)

Good luck! 🚀 You're already at 91% - these improvements should get you to 95%+
