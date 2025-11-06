#!/usr/bin/env python
"""
Script to run only the missing datasets (tic-tac-toe and wisconsin)
with detailed error logging.
"""

import sys
import traceback
from experiments.run_rid_comparison import run_rid_comparison
import pandas as pd
import numpy as np
from models.treefarms_wrapper import DEFAULT_TF_CONFIG

# Set random seed for reproducibility
np.random.seed(0)

# Configuration matching the main script
tf_config = DEFAULT_TF_CONFIG
tf_config['depth_budget'] = 4
tf_config['rashomon_ignore_trivial_extensions'] = True
tf_config['regularization'] = 0.02
tf_config['rashomon_bound_adder'] = 0.02
n_resamples = 100

datasets_to_run = [
    ('tic-tac-toe.csv', 0),
    ('wisconsin.csv', 1)
]

print("="*80)
print("Running Missing Datasets with Error Logging")
print("="*80)

for df_file, idx in datasets_to_run:
    print(f"\n{'='*80}")
    print(f"Processing: {df_file} (index {idx})")
    print(f"{'='*80}\n")

    try:
        # Load the dataset
        print(f"Loading data from data/{df_file}...")
        original_df = pd.read_csv(f"data/{df_file}")
        print(f"✓ Loaded {len(original_df)} rows, {len(original_df.columns)} columns")

        # Drop NaN values
        original_df = original_df.dropna(axis=0).reset_index(drop=True)
        print(f"✓ After dropping NaN: {len(original_df)} rows")

        # Check if data is already binarized
        internal_bin = not ('tic-tac-toe' in df_file or 'monk' in df_file)
        print(f"Internal binarization: {internal_bin}")

        # Create dataset name
        df_name = f"{df_file.split('.')[0]}_{n_resamples}_resamples_triv_ext_{tf_config['rashomon_ignore_trivial_extensions']}_db{tf_config['depth_budget']}_reg_{tf_config['regularization']}_rashomon_bound_adder_{tf_config['rashomon_bound_adder']}"
        print(f"Output name: {df_name}")

        # Run the comparison
        print(f"\nRunning RID comparison...")
        run_rid_comparison(
            original_df,
            dataset_name=df_name,
            n_resamples=n_resamples,
            internal_bin=internal_bin,
            tf_config=tf_config
        )

        print(f"\n✓ Successfully completed {df_file}")

    except Exception as e:
        print(f"\n✗ ERROR processing {df_file}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("\nContinuing to next dataset...\n")

print("\n" + "="*80)
print("Processing Complete")
print("="*80)
print("\nCheck rid_plots_1_22/ for output files")
