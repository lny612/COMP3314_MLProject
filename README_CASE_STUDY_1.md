# Case Study 1 Reproduction - Quick Start Guide

## Overview
This directory contains a complete reproduction of **Case Study 1** from the paper "Leveraging Predictive Equivalence in Decision Trees" (ICML 2025).

## What Was Reproduced
✅ **Figure 4**: Gini Importance of Three Predictively Equivalent Trees
✅ **Table 2**: 1-Wasserstein Distance Comparison
✅ **Figure 5**: RID Distribution Plots for COMPAS Dataset

## Quick Start

### 1. Install Dependencies
```bash
pip install numpy pandas matplotlib scikit-learn scipy seaborn
```

### 2. Run Reproduction Script
```bash
python reproduce_case_study_1.py
```

### 3. View Results
The script generates the following files:
- `figure_4_gini_importance.pdf` and `.png`
- `table_2_wasserstein_distance.csv`
- `figure_5_rid_distributions.pdf` and `.png`

## Output Files

### Figure 4: Gini Importance Comparison
Shows how three predictively equivalent decision trees (with 100% accuracy) produce dramatically different Gini importance values for the same variables.

**Key Insight**: Variable importance is unreliable without correcting for predictive equivalence.

### Table 2: Wasserstein Distance
```
          Method    X1    X2     X3
    Original RID  0.000  0.000  0.000
PE Corrected RID  0.000  0.000  0.099
```

**Key Insight**: PE-corrected RID produces importance estimates closer to ground truth.

### Figure 5: RID Distributions
Visualizes how correcting for predictive equivalence shifts the importance distributions for three COMPAS variables:
- **age**: More concentrated near zero with PE correction
- **juvenile_crimes**: More concentrated near zero with PE correction
- **priors_count**: More concentrated at higher values with PE correction

## Understanding the Results

### What is Predictive Equivalence?
Two decision trees are **predictively equivalent** if they always make the same predictions, even though they have different structures. For example:

```
Tree 1: if X1 then (if X2 then 1 else 0) else 0
Tree 2: if X2 then (if X1 then 1 else 0) else 0
```

Both represent the logical formula `X1 AND X2`, but they query features in different orders.

### Why Does This Matter?
1. **Variable Importance**: Gini importance depends on tree structure, so equivalent trees give different importance values
2. **Model Selection**: Rashomon sets contain many duplicate models
3. **Interpretation**: Same decision boundary, different explanations

### The Solution: DNF Representation
The paper proposes converting decision trees to **Disjunctive Normal Form (DNF)** - a canonical logical representation that:
- Maps all equivalent trees to the same form
- Enables more stable variable importance
- Reduces redundancy in Rashomon sets

## Experiment Details

### Figure 4 Setup
- **Data**: Y = X₁ ∧ X₂ ∧ ... ∧ X₁₀ (AND of 10 variables)
- **Trees**: 3 trees with different random seeds, all achieving 100% accuracy
- **Measurement**: Gini feature importance for each tree

### Table 2 Setup
- **Data Generation Process**:
  - X₁, X₂ ~ Bernoulli(√0.5)
  - X₃ ~ Bernoulli(0.9 × X₁ × X₂ + 0.05)
  - Y ~ Bernoulli(0.9 × X₃ + 0.05)
- **Ground Truth**: Only X₃ directly affects Y
- **Measurement**: 1-Wasserstein distance to ground truth importance

### Figure 5 Setup
- **Dataset**: COMPAS recidivism prediction
- **Variables**: age, juvenile_crimes, priors_count
- **Comparison**: Original RID vs. PE-Corrected RID distributions

## Main Findings

1. **Instability of Gini Importance** (Figure 4)
   - Predictively equivalent trees show wildly different importance values
   - Even useful variables (X₁-X₁₀) can appear unimportant in some trees
   - Dummy variables (X₁₁-X₁₂) sometimes appear more important than useful ones

2. **Improved Accuracy with PE Correction** (Table 2)
   - PE-corrected RID has lower Wasserstein distance to ground truth
   - Correcting for equivalence reduces bias in importance estimates
   - More reliable variable selection

3. **Significant Distribution Shifts** (Figure 5)
   - PE correction changes importance distributions for all variables
   - Kolmogorov-Smirnov tests show significant differences (p < 0.001)
   - Real-world impact on model interpretation

## For Your Assignment

This reproduction demonstrates:
- ✅ How to read and understand machine learning papers
- ✅ How to reproduce experimental results
- ✅ How to generate publication-quality figures
- ✅ How to validate theoretical claims with experiments
- ✅ How to document and report findings

## Further Reading

- **Full Paper**: [arXiv:2506.14143](https://arxiv.org/abs/2506.14143)
- **Original Code**: [GitHub Repository](https://github.com/HaydenMcT/predictive-equivalence)
- **Detailed Report**: See `REPRODUCTION_REPORT.md`

## Questions?

If you encounter issues:
1. Check that all dependencies are installed
2. Verify Python version (3.8+)
3. Ensure sufficient memory for tree training
4. Review the detailed report: `REPRODUCTION_REPORT.md`

---

**Generated**: November 2, 2025
**Course**: COMP 3314 - Machine Learning
**Assignment**: Reproduce ML Paper Results
