# Reproducing Case Study 2 Results

This document provides instructions for reproducing Tables 3-5 and Figures 9-10 from the paper (pages 17-21), which analyze RID (Rashomon Importance Distribution) across multiple datasets.

## Overview

The paper's Case Study 2 analyzes how predictive equivalence affects variable importance across 12 different datasets:
1. Tic-Tac-Toe
2. Wisconsin (breast cancer)
3. Coupon
4. COMPAS
5. Wine Quality
6. Broward General 2Y
7. FICO Complete
8. Iris Setosa
9. Iris Versicolor
10. Iris Virginica
11. Spiral
12. Netherlands (optional - not included in repository)

## What Gets Generated

The script generates:

### **Table 3** (Page 17)
- Rashomon set sizes for each dataset
- Shows: Total Trees, Trees w/o Trivial, Unique Trees (PE-corrected)
- Location: Embedded in the output of each dataset run

### **Tables 4 & 5** (Pages 18-19)
- KS test statistics and p-values showing distribution shift
- For each dataset and variable combination
- Columns: dataset, variable, Sup. Distance (KS stat), p-Value
- Location: `case_study_2_summary_statistics.csv` (after aggregation)

### **Figures 9 & 10** (Pages 20-21)
- Histogram plots showing RID distributions
- Original RID vs PE-Corrected RID for each significant variable
- Location: `rid_plots_1_22/{dataset_name}/rid_comparison_{variable}_alt.pdf`

## Method 1: Run All Datasets at Once (Recommended)

### Using the Automated Script

The repository includes `reproduce_case_study_2.py` which automates the entire process:

```bash
cd C:\Github\COMP 3314\COMP3314_MLProject

# Activate the virtual environment
venv\Scripts\activate

# Run the script
python reproduce_case_study_2.py
```

When prompted:
- Enter `y` to run all datasets (takes 1-2 hours)
- Enter `n` to just collect existing results
- Enter specific indices (e.g., `0,3,4`) to run only those datasets

## Method 2: Run Individual Datasets

To run RID comparison for a specific dataset:

```bash
cd C:\Github\COMP 3314\COMP3314_MLProject
venv\Scripts\activate

# Run for a specific dataset (0-11)
python -m experiments.run_rid_comparison 0  # Tic-Tac-Toe
python -m experiments.run_rid_comparison 3  # COMPAS
python -m experiments.run_rid_comparison 4  # Wine Quality
# etc...
```

### Dataset Index Reference:
```
0  = tic-tac-toe.csv
1  = wisconsin.csv
2  = coupon_full.csv
3  = compas_complete.csv
4  = wine_quality.csv
5  = broward_general_2y.csv
6  = fico_complete.csv
7  = iris_setosa.csv
8  = iris_versicolor.csv
9  = iris_virginica.csv
10 = spiral.csv
11 = netherlands.csv (may not be available)
```

## Method 3: Run in Parallel (Fastest)

For faster execution, run multiple datasets in parallel:

### Windows (PowerShell):
```powershell
cd C:\Github\COMP 3314\COMP3314_MLProject
venv\Scripts\activate

# Run multiple datasets in parallel
0..10 | ForEach-Object -Parallel {
    python -m experiments.run_rid_comparison $_
} -ThrottleLimit 4
```

### Linux/Mac (Bash):
```bash
cd /path/to/COMP3314_MLProject
source venv/bin/activate

# Run all datasets in parallel
for i in {0..10}; do
    python -m experiments.run_rid_comparison $i &
done
wait
```

## Output Structure

After running, you'll find:

```
rid_plots_1_22/
├── tic-tac-toe_100_resamples_.../
│   ├── rid_comparison_Feat0_x_alt.pdf
│   ├── rid_comparison_Feat1_x_alt.pdf
│   ├── ...
│   └── summary.csv                    # Per-dataset results
├── compas_complete_100_resamples_.../
│   ├── rid_comparison_age_alt.pdf
│   ├── rid_comparison_priors_count_alt.pdf
│   ├── ...
│   └── summary.csv
├── ...
└── [other datasets]

case_study_2_combined_results.csv      # All results combined
case_study_2_summary_statistics.csv    # Tables 4 & 5 data
```

## Understanding the Results

### Individual Dataset Summary Files

Each `summary.csv` contains:
- `var`: Variable name
- `distance`: Wasserstein distance between Original and PE-Corrected RID
- `original_rid_max`: Maximum importance in original RID
- `corrected_rid_max`: Maximum importance in PE-corrected RID
- `ks_test_stat`: Kolmogorov-Smirnov test statistic
- `ks_test_p`: P-value for KS test
- `pe_rates`: Predictive equivalence duplication rates

### Interpreting Results

**Significant variables** (p < 0.05):
- Show meaningful difference between Original and PE-Corrected RID
- These are the variables plotted in Figures 9 & 10
- Indicates predictive equivalence affects importance estimation for these variables

**Wasserstein distance**:
- Measures magnitude of distribution shift
- Larger values = bigger impact of PE correction

**PE rates (duplication rates)**:
- Shows what proportion of trees in Rashomon set are duplicates
- Higher rates = more predictive equivalence present

## Configuration

The script uses the following TreeFARMS configuration (matching the paper):

```python
tf_config = {
    'depth_budget': 4,                              # Max depth = 3
    'regularization': 0.02,                         # Sparsity penalty
    'rashomon_bound_adder': 0.02,                   # Rashomon set epsilon
    'rashomon_ignore_trivial_extensions': True      # Remove trivial duplicates
}
n_resamples = 100                                   # Bootstrap iterations
```

## Troubleshooting

### Issue: "Dataset not found"
- **Solution**: Check that the CSV file exists in the `data/` directory
- Netherlands dataset may not be included; skip index 11

### Issue: "Out of memory"
- **Solution**: Run datasets sequentially instead of in parallel
- **Solution**: Process smaller datasets first (Iris, Spiral)
- **Solution**: Reduce `n_resamples` in the script (edit line 136 of `experiments/run_rid_comparison.py`)

### Issue: Script takes too long
- **Expected time**:
  - Small datasets (Iris, Spiral): 5-10 minutes each
  - Medium datasets (COMPAS, Wisconsin): 15-30 minutes each
  - Large datasets (Coupon): 30-60 minutes each
- **Solution**: Run in parallel or only process specific datasets of interest

### Issue: Missing dependencies
```bash
pip install -r requirements.txt
```

## Verification

To verify your results match the paper:

1. **Check Table 3 values**:
   - Look at the console output showing Rashomon set sizes
   - Compare "Total Trees", "w/o Trivial", and "Ours" columns

2. **Check Tables 4 & 5**:
   - Open `case_study_2_summary_statistics.csv`
   - Find variables with p-value < 0.05
   - Compare KS statistics with paper's Tables 4 & 5

3. **Check Figures 9 & 10**:
   - Open PDF files in `rid_plots_1_22/{dataset}/`
   - Verify histograms show two distributions (Original vs PE-Corrected)
   - Compare visual patterns with paper's figures

## Expected Results

Based on the paper (pages 17-21):

### Table 3 - Example Values:
- **COMPAS**: ~12,785 total → ~3,913 w/o trivial → ~2,135 unique
- **Wisconsin**: ~24,052 total → ~11,990 w/o trivial → ~4,657 unique
- **Wine Quality**: ~6,936 total → ~2,341 w/o trivial → ~1,409 unique

### Tables 4 & 5 - Significant Variables:
- **COMPAS**: age, priors_count, juvenile_crimes (all p < 0.001)
- **Wisconsin**: texture3, concave_points3, area3 (p < 0.05)
- **FICO**: ExternalRiskEstimate, MSinceMostRecentInqexcl7days

### Figures 9 & 10:
- Should show overlapping histograms
- Orange (PE-Corrected) often shifts distribution
- Some variables show dramatic changes, others minimal

## Additional Analysis

To extend the analysis:

### Compute Table 3 statistics from results:
```python
import pandas as pd
from pathlib import Path

# After running, collect Rashomon set sizes
results_dir = Path('rid_plots_1_22')
for subdir in results_dir.iterdir():
    if subdir.is_dir():
        # Rashomon set info is in the dataset name and output logs
        print(f"Dataset: {subdir.name}")
```

### Filter for significant variables only:
```python
df = pd.read_csv('case_study_2_summary_statistics.csv')
significant = df[df['ks_test_p'] < 0.05]
print(f"Variables with significant shift: {len(significant)}")
print(significant[['dataset', 'var', 'distance', 'ks_test_p']])
```

### Compare with paper's reported values:
```python
# Example for COMPAS age variable
compas_age = df[(df['dataset'].str.contains('compas')) &
                 (df['var'] == 'age')]
print(f"KS statistic: {compas_age['ks_test_stat'].values[0]:.6f}")
print(f"P-value: {compas_age['ks_test_p'].values[0]:.6f}")
```

## Questions?

If you encounter issues:
1. Check that all datasets are in `data/` directory
2. Verify virtual environment is activated
3. Ensure all dependencies are installed
4. Review error messages for specific issues
5. Try running a single small dataset first (e.g., index 7 for Iris Setosa)

## Citation

If using these results, cite the paper:
```
McTavish, H., Boner, Z., Donnelly, J., Seltzer, M., & Rudin, C. (2025).
Leveraging Predictive Equivalence in Decision Trees.
Proceedings of the 42nd International Conference on Machine Learning (ICML).
```
