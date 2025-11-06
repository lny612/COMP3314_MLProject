# Quick Start: Reproducing Case Study 2

## TL;DR

**Goal**: Reproduce Table 3, Table 4, Table 5, Figure 9, and Figure 10 from pages 17-21 of the paper.

**What they show**: How predictive equivalence affects variable importance across 12 datasets.

## Fastest Way to Run

### Windows:
```cmd
cd C:\Github\COMP 3314\COMP3314_MLProject
run_case_study_2.bat
```

Then press `y` when prompted to run all datasets.

### Alternative (Python directly):
```bash
cd C:\Github\COMP 3314\COMP3314_MLProject
venv\Scripts\activate
python reproduce_case_study_2.py
```

## What You'll Get

After running (takes 1-2 hours total):

1. **PDF plots** in `rid_plots_1_22/{dataset_name}/`
   - These are Figures 9 & 10
   - One plot per variable showing Original vs PE-Corrected RID

2. **CSV file**: `case_study_2_combined_results.csv`
   - Contains all the data for Tables 4 & 5
   - KS test statistics and p-values

3. **Individual summaries** in `rid_plots_1_22/{dataset_name}/summary.csv`
   - Per-dataset results
   - Includes Rashomon set sizes (Table 3 data)

## Quick Test

To test if everything works, run just one small dataset:

```bash
cd C:\Github\COMP 3314\COMP3314_MLProject
venv\Scripts\activate
python -m experiments.run_rid_comparison 7  # Iris Setosa (fastest)
```

This should complete in 5-10 minutes. Check `rid_plots_1_22/` for output.

## Paper References

- **Table 3** (page 17): Rashomon set sizes
  - Look for console output showing "Total Trees", "w/o Trivial", "Ours"

- **Table 4 & 5** (pages 18-19): KS test statistics and p-values
  - In `case_study_2_combined_results.csv`
  - Look at columns: `ks_test_stat`, `ks_test_p`

- **Figure 9 & 10** (pages 20-21): RID distribution histograms
  - PDFs in `rid_plots_1_22/{dataset_name}/`
  - Blue = Original RID, Orange = PE-Corrected RID

## Need Help?

See `CASE_STUDY_2_INSTRUCTIONS.md` for detailed documentation.

## Key Datasets to Test

If you want to run just a few representative datasets:

```bash
python -m experiments.run_rid_comparison 3   # COMPAS (from main paper)
python -m experiments.run_rid_comparison 1   # Wisconsin
python -m experiments.run_rid_comparison 4   # Wine Quality
python -m experiments.run_rid_comparison 10  # Spiral
```

These 4 datasets give good coverage and take ~1 hour total.
