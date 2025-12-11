# Stress-Level-Prediction

## Introduction

### The Problem: The Hidden Cost of Undetected Stress

Chronic stress is a silent epidemic, contributing to cardiovascular disease, anxiety disorders, depression, and compromised immune function. Despite its pervasive impact, stress remains largely invisible until it manifests as serious health consequences. Current stress assessment methods—self-reported questionnaires and periodic clinical evaluations—suffer from critical limitations:

- **Retrospective and subjective**: Rely on memory and self-awareness, introducing recall bias
- **Infrequent sampling**: Capture stress only during clinic visits, missing daily patterns
- **Lack of ecological validity**: Cannot identify real-world triggers and contexts
- **Delayed intervention**: Problems are detected only after prolonged exposure causes damage

This gap between stress occurrence and detection prevents timely intervention, leaving individuals unable to recognize their stress patterns, identify triggers, or take preventive action.

### The Solution: Continuous Physiological Monitoring

Modern wearable biosensors enable a paradigm shift from periodic self-reporting to continuous, objective stress monitoring. The body's autonomic nervous system produces measurable physiological responses during stress that can be captured in real-time:

**Key Physiological Stress Markers:**

- **Electrodermal Activity (EDA/GSR)**: Sympathetic activation increases sweat gland activity, changing skin conductance
- **Heart Rate Variability (HRV)**: Stress alters the balance between sympathetic and parasympathetic tone, reflected in inter-beat intervals
- **Blood Volume Pulse (BVP)**: Changes in peripheral blood flow indicate autonomic arousal
- **Skin Temperature**: Vasoconstriction during stress reduces peripheral temperature
- **Accelerometry (ACC)**: Movement patterns capture behavioral stress responses (fidgeting, reduced activity)

### The Opportunity: Data Mining for Real-Time Stress Detection

By applying machine learning and data mining techniques to multimodal physiological signals from consumer wearables, we can:

1. **Detect stress automatically** without conscious self-reporting
2. **Classify stress intensity** (no stress, low, moderate, high) for nuanced understanding
3. **Enable real-time intervention** through immediate alerts and recommendations
4. **Identify personal stress patterns** across contexts, times, and activities
5. **Support clinical decision-making** with objective longitudinal data

This project demonstrates the feasibility of automated stress detection using the Empatica E4 wristband, exploring both traditional machine learning (logistic regression with engineered features) and deep learning approaches (LSTM-ResNet on raw time-series). Our work addresses the critical challenge of severe class imbalance in naturalistic stress data, where stress episodes comprise only 18% of daily life, making accurate detection substantially more difficult than controlled laboratory settings.

### Dataset Overview

**Dataset Characteristics:**

- **31 subjects** (18 male, 13 female) performing three protocols: stress induction, aerobic exercise, and anaerobic exercise
- **~6,500 labeled windows** (60-second windows with 30-second overlap) across all subjects, typically split 75-80% train / 20-25% test
- **7 sensor channels**: Accelerometry (ACC), Blood Volume Pulse (BVP), Electrodermal Activity (EDA), Heart Rate (HR), Inter-Beat Interval (IBI), Skin Temperature (TEMP), and event tags
- **Multiple sampling rates**: ACC (32Hz), BVP (64Hz), EDA/TEMP (4Hz) — resampled to unified 4Hz
- **Stress labels**: Self-reported stress scores (0-10) mapped to 4 classes (no stress, low, moderate, high) or binary (stress vs. no stress)

**Key Dataset Limitations:**

- **Severe class imbalance**: 82% no-stress samples, only 1.3% low-stress (53 windows in 4-class)
- **Small subject pool**: 28 training subjects, 8 test subjects
- **Data quality issues**: Signal duplications (S02), sensor coverage failures (f07), Bluetooth disconnections (f14, S11, S16), incomplete protocols (S03, S06, S07)
- **Subjective labels**: Self-reported stress varies dramatically across individuals (e.g., baseline stress ranges from 0 to 7)
- **Homogeneous demographics**: Primarily young adults (19-31 years), 89% physically active, limited diversity
- **Single-session data**: One recording per subject per protocol (~30-40 minutes), insufficient for capturing longitudinal stress patterns
- **Label ambiguity**: 60-second windows with 30-second overlap create ambiguous labels during stress transitions

### Real-World Impact

**Individual Health Management:**

- Personal stress awareness and self-management
- Guided relaxation prompts during detected stress episodes
- Lifestyle modification based on identified triggers

**Clinical Applications:**

- Objective assessment for anxiety/stress disorders
- Treatment efficacy monitoring
- Personalized therapy recommendations

**Workplace Wellness:**

- Early burnout detection
- Work environment optimization
- Targeted intervention programs

**Research Advancement:**

- Large-scale stress epidemiology studies
- Understanding stress-disease mechanisms
- Developing adaptive stress management systems

---

## Methodology

### Model Architecture (Models 5-8): Phase-Aware Hybrid LSTM-ResNet

All four deep learning models share the same core architecture, combining convolutional feature extraction, temporal modeling, and statistical feature fusion:

#### **1. Multi-Scale Sequence Encoder**

Processes raw time-series signals through three parallel pathways to capture features at different temporal scales:

- **Short-term features** (kernel size 3, dilation 1): Captures immediate physiological responses
- **Medium-term features** (kernel size 7, dilation 2): Captures evolving stress patterns
- **Long-term features** (kernel size 15, dilation 4): Captures sustained physiological changes

Each pathway consists of:

- 1D Convolutional layer → Batch Normalization → ReLU
- 2 ResNet blocks with skip connections (prevents gradient degradation)

The three scales are concatenated and merged through a 1×1 convolution to 128 channels.

#### **2. Temporal Modeling with Bidirectional LSTM**

- **3-layer bidirectional LSTM** (256 hidden units total: 128 forward + 128 backward)
- Captures both past→future and future→past temporal dependencies
- Dropout (0.3) between layers for regularization
- Processes merged multi-scale features to learn sequential stress patterns

#### **3. Multi-Head Self-Attention**

- 8 attention heads enable the model to focus on different aspects of the time series
- Learns which temporal segments are most important for stress detection
- Reduces noise from irrelevant physiological fluctuations

#### **4. Statistical Feature Branch**

Parallel pathway processing engineered features from raw signals:

- Input: 39 statistical features (mean, std, min, max, percentiles, correlations)
- 2-layer MLP: 128 → 64 units
- Batch normalization and dropout (0.3, 0.2)
- Provides interpretable physiological metrics alongside learned representations

#### **5. Phase Embedding**

- Embeds protocol phase information (Stroop, TMCT, rest, aerobic, etc.) into 32-dimensional vectors
- Allows model to learn phase-specific stress signatures
- Accounts for contextual differences (e.g., stress during cognitive tasks vs. physical exercise)

#### **6. Fusion and Classification**

- Concatenates: Sequence features (384D) + Statistical features (64D) + Phase embedding (32D) = **480D fusion vector**
- 3-layer classifier: 480 → 256 → 128 → num_classes
- Dropout (0.3, 0.2) for regularization
- Outputs class logits for stress level prediction

**Total Parameters:** ~1.5-2M depending on input channel configuration

---

### Model-Specific Preprocessing Techniques

#### **Model 5: 4-Class with Conservative Temporal Augmentation**

**Preprocessing Pipeline:**

1. **Signal Resampling**: All sensors to 4 Hz
2. **Enhanced Channel Extraction** (30 channels total):

   - Raw signals: EDA, Temperature, ACC, BVP
   - First derivatives: ∂EDA/∂t, ∂Temp/∂t, ∂ACC/∂t, ∂BVP/∂t
   - EDA decomposition: Tonic (10s MA), Phasic (EDA - Tonic)
   - EDA acceleration: ∂²EDA/∂t²
   - EDA moving averages: 5s window, 15s window
   - BVP envelope: Hilbert transform for amplitude modulation
   - Cross-signal products: EDA × BVP (arousal interaction)
   - Smoothed ACC: 3s moving average
   - Respiratory signal: Extracted from BVP fluctuations
   - Sample entropy signal: Local complexity measure
   - Wavelet channels: Additional time-frequency features

3. **Subject-Specific Baseline Normalization**:

   - For each subject: Compute mean/std from "no_stress" windows
   - Normalize all windows: (x - baseline_mean) / baseline_std
   - Handles individual physiological differences

4. **Data Augmentation** (Conservative):

   - **Low-stress: 2× augmentation** (most underrepresented class)
   - Moderate-stress: 1× (no augmentation)
   - High-stress: 1× (no augmentation)
   - Techniques per augmentation:
     - Time warping: 0.95-1.05× speed variation
     - Gaussian noise: 3% of signal std
     - Temporal shift: ±10% of window length

5. **Train/Test Split**: StratifiedGroupKFold (5 folds, fold 0 used)

   - Ensures subjects appear in only train OR test set
   - Maintains class distribution across folds

6. **Loss Function**: Focal Loss (γ=2.0, class weights from inverse frequency)
   - Down-weights easy no-stress examples
   - Up-weights hard minority class examples

**Training Config:**

- 50 epochs, batch size 32
- AdamW optimizer (LR=5e-4, weight decay=1e-4)
- Cosine annealing warm restarts
- Label smoothing: 0.05
- EMA: Exponential moving average (decay=0.995) for stable predictions

---

#### **Model 6: 4-Class with Asymmetric Focal Loss**

**Preprocessing Pipeline:**
1-3. **Identical to Model 5** (Enhanced channels, baseline normalization)

4. **Data Augmentation: DISABLED**

   - No temporal augmentation applied
   - Relies on loss function to handle imbalance instead
   - (BorderlineSMOTE imported but not integrated)

5. **Train/Test Split**: Same as Model 5

6. **Loss Function: Asymmetric Focal Loss**
   - **Per-class gamma values**:
     - High-stress: γ=2.5 (hardest to detect)
     - **Low-stress: γ=3.0** (most challenging class - only 1.3% of data)
     - Moderate-stress: γ=2.0
     - No-stress: γ=1.0 (easiest, majority class)
   - Higher gamma = more focus on misclassified examples
   - Class-specific weights from inverse frequency

**Training Config:**

- **60 epochs** (10 more than Model 5 to compensate for no augmentation)
- Other hyperparameters identical to Model 5

**Key Difference:** Trades data augmentation for sophisticated loss function that explicitly prioritizes learning from rare classes.

---

#### **Model 7: Binary with Conservative Temporal Augmentation**

**Preprocessing Pipeline:**
1-3. **Identical to Models 5-6** (Enhanced channels, baseline normalization)

4. **Label Mapping: 4-class → Binary**

   - `stress_bucket()` function modified:
     - Low/Moderate/High stress → **"stress"** (class 1)
     - No stress → **"no_stress"** (class 0)
   - Simplifies task: ~82% no-stress, ~18% stress

5. **Data Augmentation** (Conservative):

   - **Stress: 2× augmentation** (minority class)
   - No-stress: 1× (no augmentation)
   - Same augmentation techniques as Model 5

6. **Train/Test Split**: StratifiedGroupKFold (same as Models 5-6)

7. **Loss Function**: Focal Loss (γ=2.0, binary class weights)

**Training Config:**

- 50 epochs, batch size 32
- Identical hyperparameters to Model 5

**Key Difference:** Binary classification is easier than 4-class, expected higher performance but less granular stress level information.

---

#### **Model 8: Binary with Asymmetric Focal Loss**

**Preprocessing Pipeline:**
1-3. **Identical to Models 5-7** (Enhanced channels, baseline normalization)

4. **Label Mapping: Binary** (same as Model 7)

5. **Data Augmentation: DISABLED** (same as Model 6)

6. **Train/Test Split**: StratifiedGroupKFold

7. **Loss Function: Asymmetric Focal Loss (Binary)**
   - **Per-class gamma values**:
     - No-stress: γ=1.0 (easy majority class)
     - **Stress: γ=2.5** (harder minority class)
   - Binary class weights from inverse frequency

**Training Config:**

- **60 epochs**
- Other hyperparameters identical to Model 6

**Key Difference:** Combines binary simplification with asymmetric loss for best stress detection performance, expected highest recall for stress class.

---

### Preprocessing Comparison Summary

| Technique                  | Model 5                     | Model 6                        | Model 7               | Model 8              |
| -------------------------- | --------------------------- | ------------------------------ | --------------------- | -------------------- |
| **Task**                   | 4-class                     | 4-class                        | Binary                | Binary               |
| **Enhanced Channels**      | ✓ 30 channels               | ✓ 30 channels                  | ✓ 30 channels         | ✓ 30 channels        |
| **Baseline Normalization** | ✓ Subject-specific          | ✓ Subject-specific             | ✓ Subject-specific    | ✓ Subject-specific   |
| **Temporal Augmentation**  | ✓ 2× low-stress             | ✗ Disabled                     | ✓ 2× stress           | ✗ Disabled           |
| **Loss Function**          | Focal (γ=2.0)               | **Asymmetric Focal**           | Focal (γ=2.0)         | **Asymmetric Focal** |
| **Class Gammas**           | Uniform 2.0                 | [2.5, 3.0, 2.0, 1.0]           | Uniform 2.0           | [1.0, 2.5]           |
| **Training Epochs**        | 50                          | **60**                         | 50                    | **60**               |
| **Philosophy**             | Feature-rich + Augmentation | Loss-driven imbalance handling | Binary + Augmentation | Binary + Loss-driven |

**Design Rationale:**

- **Models 5 & 7**: Trust rich features and conservative augmentation to maintain signal quality
- **Models 6 & 8**: Minimize data manipulation, let asymmetric loss guide learning toward minority classes
- **Models 7 & 8**: Binary simplification trades granularity for higher overall performance

---

## Results and Performance Analysis

### Overall Performance Summary

Our experiments across four model variants demonstrate the challenges and successes of stress detection with severe class imbalance. The best performing model achieved **66.4% macro F1** (Model 8 - Binary with Asymmetric Focal Loss), representing a **66% improvement** over the majority-class baseline (40% macro F1).

| Model       | Task    | Accuracy | Macro F1  |
| ----------- | ------- | -------- | --------- |
| **Model 5** | 4-class | 85.9%    | 54.3%     |
| **Model 6** | 4-class | 80.3%    | 41.8%     |
| **Model 7** | Binary  | 82.8%    | 59.0%     |
| **Model 8** | Binary  | 80.7%    | **66.4%** |

### Key Findings

#### **1. Binary Models Outperform 4-Class Models**

Binary models (7, 8) achieved higher macro F1 than 4-class models (5, 6), with Model 8 showing the best overall performance:

- **Model 8 vs. Model 6**: +24.6% macro F1 (66.4% vs. 41.8%)
- **Model 7 vs. Model 5**: +4.7% macro F1 (59.0% vs. 54.3%)

**Interpretation:** Collapsing stress levels into a single "stress" class (18% of data) provides:

- More training examples per class, reducing data scarcity
- Clearer decision boundary (stress vs. no-stress)
- Less label ambiguity between stress intensity levels

**Trade-off:** Binary models lose granularity and cannot distinguish between mild stress requiring self-management and severe stress requiring clinical intervention.

#### **2. Asymmetric Focal Loss Shows Mixed Results**

The asymmetric focal loss strategy produced different outcomes across tasks:

- **4-class task (Model 6)**: Underperformed augmentation-based Model 5 (-12.5% macro F1)
- **Binary task (Model 8)**: Outperformed augmentation-based Model 7 (+7.4% macro F1)

**Interpretation:** Asymmetric focal loss is more effective for binary classification where the decision boundary is simpler. With 4 classes and extreme imbalance (1.3% low-stress), the per-class gamma weighting may have over-emphasized rare classes, leading to overfitting or unstable training.

#### **3. Model 8 (Binary with Asymmetric Focal Loss) Achieves Best Performance**

Model 8's **66.4% macro F1** makes it the top-performing model overall:

**Strengths:**

- Best balanced performance across stress/no-stress classes
- Asymmetric focal loss (γ=2.5 for stress) effectively handles binary imbalance
- 80.7% accuracy demonstrates reasonable overall performance
- Suitable for binary stress detection applications

**Practical Applications:**

- Real-time wearable stress monitoring
- Binary alert systems (stress detected / no stress)
- Applications prioritizing balanced class performance

**Limitations:**

- Cannot distinguish stress intensity levels (low/moderate/high)
- Accuracy (80.7%) indicates room for improvement
- May confuse physical activity with stress

#### **4. Dataset Limitations Impact Performance**

All models were constrained by:

- **Small subject pool**: 28 training subjects, 8 test subjects
- **Severe class imbalance**: 82% no-stress samples
- **Individual variability**: Wide range of physiological stress responses
- **Data quality issues**: Sensor disconnections, missing data, protocol incompletions

These limitations explain why models achieved 54-66% macro F1 rather than higher performance levels seen in larger, more balanced datasets.

### Common Challenges

**Primary sources of classification errors:**

1. **Physical Activity Confusion**: Exercise-induced EDA/HR elevation mimics stress responses
2. **Transition Windows**: Windows spanning stress onset/offset periods have ambiguous labels
3. **Individual Variability**: Some subjects show minimal physiological response to self-reported stress
4. **Data Quality**: Sensor disconnections and missing data reduce model performance
5. **Small Sample Size**: Limited training examples (especially for minority classes) restrict model learning

---
