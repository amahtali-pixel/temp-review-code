# Radial Encoding Framework (REF)

**Beyond the Black Box: A Non-Neural, Evidence-Based Framework with Complete Decision Transparency**

**Apache 2.0 License** | Implementation Code | Patent-Protected Methodology

---

## 📢 Important Notice

This repository has been renamed and expanded from "Rotary Pattern Extraction Network (RPENet)" to reflect the broader Radial Encoding Framework methodology presented in our 2026 publication.

---

## 🔬 Research Overview

This repository contains the official implementation of the **Radial Encoding Framework**, a novel non-neural, geometry-based approach that achieves **99.89% accuracy** on the full MNIST test set while providing **complete decision transparency**. Our work fundamentally challenges the entrenched belief that high accuracy necessitates opaque, black-box models.

**Key Result:** Only **11 errors** on 10,000 MNIST test images — matching state-of-the-art neural networks (99.79%) while offering complete interpretability.

---

## 🏆 Key Contributions

### Geometric Foundation
- **Radial Encoding Methodology**: Deterministic geometric operations bridging continuous spatial reality with discrete computation
- **Three Complementary Strategies**: Local (LRE), Hierarchical (HRE), and Extended (ERE) radial encoding
- **Contrapositive Geometric Validation**: Objects must prove identity through both presence of class-specific patterns and absence of disqualifying patterns
- **Digit 1 Preference Rule**: Geometrically-justified tie-breaking addressing pattern library imbalance

### Systematic Optimization
- **Optimal Tolerances**: 
  - `τ_c* = 0.10` (clustering tolerance) — established through comprehensive analysis spanning 6.3× to 327× compression
  - `τ_v* = 0.19` (validation tolerance) — peak accuracy at this threshold
- **Digit Complexity Metric**: Quantifies geometric complexity (e.g., digit 7 requires 76.5% more prototypes than digit 1)
- **Proper Confidence Calibration**: Errors show **27× lower** confidence than correct predictions (mean 0.0005 vs 0.0133)

### Competitive Performance with Full Transparency
- **99.89% accuracy** on full MNIST test set (9,989/10,000 correct)
- Within **0.10%** of state-of-the-art neural networks (99.79%)
- **Complete traceability**: Every decision references specific geometric patterns, spatial locations, and training examples
- **94.3%** of cases resolved in Stage 1; only **5.7%** require arbitration

---

## 📊 Three Architecture Variants

### 1. Three-Stage Validation Model (Primary — 99.89% accuracy)

| Stage | Method | Purpose |
|-------|--------|---------|
| **Stage 1** | Local Radial Encoding (LRE) | Primary classification with τ_v = 0.19 |
| **Stage 2** | Extended Radial Encoding (ERE) | Arbitration for ambiguous cases (excluding digit 1 ties) |
| **Stage 3** | Pixel density analysis | Resolution for rare persistent ambiguities (<0.1% of cases) |

**Digit 1 Preference Rule:** Whenever a tie occurs between digit 1 and any other digit class, the algorithm defaults to digit 1 — geometrically justified by digit 1's minimal pattern library and structural simplicity.

**Implementation:** `Mnist_Radial_Classifier_0.1.py`

### 2. One-Stage Model with Logarithmic Weighting (97.52% accuracy)
- Addresses systematic voting biases through `log(frequency + 1)` weighting
- Eliminates cluster distribution imbalances (3.16× ratio reduced to ~1.10×)
- Optimal τ_v = 0.10 for this variant
- **Implementation:** `Weighted_classifier_0.10.py`

### 3. Single-Stage Hierarchical Encoding (99.2% accuracy on digits 0,1,7)
- Demonstrates HRE's standalone descriptive power
- Minimalist 5-dimensional geometric representation
- Maintains complete interpretability

---

## ⚖️ Legal & Licensing Information

### 📄 Code License
The implementation code in this repository is released under the **Apache License 2.0**, permitting:

- ✅ Commercial and non-commercial use
- ✅ Modification and distribution
- ✅ Express patent grant for users of this implementation
- ✅ Requirement: Attribution and license preservation

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
- Complete metadata integration: Spatial coordinates, zone assignments, training references
- Interpretable clusters: Each represents a distinct geometric prototype with quantitative descriptors (S₁–S₈ directional sums)

### Systematic Tolerance Optimization
Comprehensive exploration across three orders of magnitude:

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
- **94.3%** of cases resolved in Stage 1; only **5.7%** require arbitration

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
