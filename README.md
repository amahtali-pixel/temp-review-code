# Radial Encoding Framework (REF)

**Beyond the Black Box: A Non-Neural, Evidence-Based Framework with Complete Decision Transparency**

**Apache 2.0 License** | Implementation Code | Patent-Protected Methodology

---

## 📢 Important Notice

This repository has been renamed and expanded from "Rotary Pattern Extraction Network (RPENet)" to reflect the broader Radial Encoding Framework methodology presented in our 2026 publication.

---

## 🔬 Research Overview

This repository contains the official implementation of the **Radial Encoding Framework**, a non-neural, geometry-based approach that achieves **99.89% accuracy** on the full MNIST test set while providing **complete decision traceability**. Our work challenges the entrenched belief that high accuracy necessitates opaque, black-box models.

**Key Result:** Only **11 errors** on the 10,000-image held-out MNIST test set, with every decision traceable to specific stored geometric patterns. An in-sample development check on the 50,000-image training partition gives 99.90% (49,950/50,000); this is a development-consistency check, not a second generalization result.

---

## 🏆 Key Contributions

### Geometric Foundation
- **Radial Encoding Methodology**: Deterministic geometric operations bridging continuous spatial reality with discrete computation
- **Three Complementary Strategies**: Local (LRE), Hierarchical (HRE), and Extended (ERE) radial encoding
- **Contrapositive Geometric Validation**: Objects must prove identity through both presence of class-specific patterns and absence of disqualifying patterns

### Systematic Optimization
- **Optimal Tolerances**:
  - `τ_c* = 0.10` (clustering tolerance) — established through analysis spanning 6.3× to 327× compression
  - `τ_v* = 0.19` (validation tolerance) — peak accuracy at this threshold
- **Digit Complexity Metric**: Quantifies geometric complexity (e.g., digit 7 requires 76.5% more prototypes than digit 1)
- **Margin-Based Separation**: Errors show **27× lower** margin than correct predictions (mean 0.0005 vs 0.0133) on the held-out test set, and 28× on the in-sample training partition. The score is a normalized margin, not a calibrated posterior probability.

### Competitive Performance with Full Transparency
- **99.89% accuracy** on the held-out MNIST test set (9,989/10,000 correct)
- Competitive with state-of-the-art neural networks on MNIST
- **Complete traceability**: Every decision references specific geometric patterns, spatial locations, and training examples
- **97.3%** of cases resolved in Stage 1; only **2.7%** require arbitration

---

## 📊 Three Architecture Variants

### 1. Three-Stage Validation Model (Primary — 99.89% accuracy)

| Stage | Method | Purpose |
|-------|--------|---------|
| **Stage 1** | Local Radial Encoding (LRE) | Primary classification with τ_v = 0.19 |
| **Stage 2** | Extended Radial Encoding (ERE) | Arbitration for ambiguous cases |
| **Stage 3** | Pixel density analysis | Resolution for rare persistent ambiguities (~0% of cases) |

**Note on tie-breaking:** An earlier version defaulted ties involving digit 1 to digit 1. A controlled ablation at `τ_v = 0.19` showed that this rule changed no test-set errors (six with and without it, 11 total in both configurations) and one training-set error (19→18 digit-1 errors; 50→49 total), so it has been removed from the reported architecture. The full before/after record is documented in the accompanying technical report.

**Implementation:** `Mnist_Radial_3_Stages_Classifier_0.1_0.19.py`

### 2. One-Stage Model with Logarithmic Weighting (97.52% accuracy)
- Addresses systematic voting biases through `log(frequency + 1)` weighting
- Reduces cluster distribution imbalance (3.16× ratio reduced to ~1.10×)
- Optimal τ_v = 0.10 for this variant
- **Implementation:** `Weighted_classifier_0.10.py`

### 3. Single-Stage Hierarchical Encoding (99.2% accuracy on digits 0,1,7)
- Demonstrates HRE's standalone descriptive power
- Minimalist 5-dimensional geometric representation
- Evaluated on the digits 0, 1, and 7 subset only (3,143 test images); no full 10-class HRE result is claimed
- Maintains complete interpretability

---

## ⚖️ Legal & Licensing Information

### 📄 Code License
The implementation code in this repository is released under the **Apache License 2.0** for **non-commercial purposes only**.

**⚠️ IMPORTANT COMMERCIAL USE RESTRICTION:**

While the code is open source under Apache 2.0, the underlying **patented methodology** (Algerian Patent DZ/P/2025/1546) is NOT licensed for commercial use.

| Use Type | Allowed? |
|----------|----------|
| Research, education, personal projects | ✅ Yes (free) |
| Open-source contributions | ✅ Yes (free) |
| **Commercial products or services** | ❌ **No** (requires separate license) |
| Monetization of any kind | ❌ **No** (requires separate license) |

**For commercial licensing inquiries**, please contact the author.

**Permitted uses under Apache 2.0 (non-commercial):**
- ✅ Non-commercial use
- ✅ Modification and distribution for non-commercial purposes
- ✅ Attribution and license preservation

### 🔒 Patent Protection
The underlying Radial Encoding Framework methodology represents novel research and is protected under applicable patent laws.

| | |
|---|---|
| **Protected Methodology** | Radial Encoding Framework |
| **Patent Application** | Algerian Patent Application DZ/P/2025/1546 |
| **Filing Date** | November 5, 2025 (Pending Examination) |
| **Rights** | All commercial rights reserved. © 2026 |

**For commercial licensing inquiries**, please contact the author.

---

## 🏗️ Architectural Highlights

### Knowledge Base Construction
- **10.48 million** geometric patterns library (at τ_c = 0.10)
- Complete metadata integration: spatial coordinates, zone assignments, training references
- Interpretable clusters: each represents a distinct geometric prototype with quantitative descriptors (S₁–S₈ directional sums)

### Systematic Tolerance Optimization
Exploration across three orders of magnitude:

| Regime | τ_c Range | Characteristics |
|--------|-----------|-----------------|
| **Precision regime** | < 0.10 | Excessive differentiation, limited by hardware |
| **Optimal regime** | 0.10–0.15 | Balanced trade-off (6.3–10.5× compression, optimal discrimination) |
| **Generalization regime** | > 0.20 | Rapid information loss with marginal gains |

### Cluster Size Comparison (at τ_c = 0.10)

| Digit | Clusters | Complexity vs. Digit 1 |
|-------|----------|------------------------|
| 1 | 86,074 | 1.00× (baseline) |
| 7 | 151,894 | 1.765× |
| 8 | 272,495 | 3.16× |

### Computational Efficiency
- **80% data reduction** through sudden change detection (edge-based processing)
- Linear scaling with contour pixels (not quadratic with image size)
- **97.3%** of cases resolved in Stage 1; only **2.7%** require arbitration

---

## 📄 Supplementary Technical Report

A separate technical report documents the portability of the framework to a second, structurally different dataset: the **BreakHis** breast-cancer histopathology benchmark. It reports the full adaptation trajectory (grayscale baseline through opponent-channel LRE), six independent ablations, a structural analysis of the prototype reservoir, and the boundary conditions of the transfer. The single-resolution pipeline reaches 85.58% binary accuracy, below the 90–99% range reported by CNN-based methods on the same dataset; the report diagnoses the reason rather than leaving it unexplained, and specifies a falsifiable next test. The report also contains the full record of the Digit 1 rule ablation.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
