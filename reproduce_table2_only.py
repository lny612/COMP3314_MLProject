"""
REAL Reproduction of Table 2 ONLY
Using the actual TreeFARMS and DNF algorithms from the paper's repository
"""

from rid.rashomon_importance_distribution import RashomonImportanceDistribution as RID
from models.treefarms_wrapper import DEFAULT_TF_CONFIG
import pandas as pd
import numpy as np

print("=" * 80)
print("REAL REPRODUCTION OF TABLE 2")
print("Using TreeFARMS + DNF Algorithm")
print("=" * 80)

# Configuration matching the paper
np.random.seed(42)
tf_config = DEFAULT_TF_CONFIG.copy()
tf_config['depth_budget'] = 4
tf_config['rashomon_ignore_trivial_extensions'] = True
tf_config['regularization'] = 0.02
tf_config['rashomon_bound_adder'] = 0.02

n_resamples = 100  # Paper uses 100 bootstrap samples

print("\nReproducing Table 2 - Synthetic Dataset...")
print("This uses the EXACT data generating process from Section 5.2 of the paper")

# Synthetic DGP from Section 5.2: X1, X2 ~ Bernoulli(sqrt(0.5)), X3 ~ Bernoulli(0.9*X1*X2 + 0.05)
# Y ~ Bernoulli(0.9*X3 + 0.05)
num_samples = 5000
x1 = np.random.binomial(1, p=0.5**0.5, size=(num_samples))
x2 = np.random.binomial(1, p=0.5**0.5, size=(num_samples))
x3 = np.random.binomial(1, p=0.9 * x1 * x2 + 0.05, size=(num_samples))
y = np.random.binomial(1, p=0.9 * x3 + 0.05, size=(num_samples))

synthetic_data = pd.DataFrame({
    "X1": x1,
    "X2": x2,
    "X3": x3,
    "y": y
})

# Ground truth: only X3 matters
ground_truth_importance = {'X1': 0.0, 'X2': 0.0, 'X3': 1.0}

print(f"\nComputing Rashomon sets with TreeFARMS (this may take several minutes)...")
print(f"  - Using {n_resamples} bootstrap samples")
print(f"  - Max depth: {tf_config['depth_budget']}")
print(f"  - Regularization: {tf_config['regularization']}")

# Compute RID WITHOUT predictive equivalence correction
print("\n  Computing Original RID (with duplicate trees)...")
rid_original = RID(
    synthetic_data,
    n_resamples=n_resamples,
    tf_config=tf_config,
    binning_map={0: 'X1', 1: 'X2', 2: 'X3'},
    binning_fn=lambda x: x,
    use_test_df=False,
    tree_type="default",  # Regular trees (includes duplicates)
    dataset_name="synthetic_table2",
    verbose=True
)

# Compute RID WITH predictive equivalence correction (using DNF)
print("\n  Computing PE-Corrected RID (removing duplicate trees via DNF)...")
rid_corrected = RID(
    synthetic_data,
    n_resamples=n_resamples,
    tf_config=tf_config,
    binning_map={0: 'X1', 1: 'X2', 2: 'X3'},
    binning_fn=lambda x: x,
    use_test_df=False,
    unique_trees_only=True,  # This uses DNF to remove duplicates!
    tree_type="DNF",
    dataset_name="synthetic_table2_corrected",
    verbose=True
)

# Compute Wasserstein distances to ground truth
print("\n  Computing Wasserstein distances...")
distances_original = {}
distances_corrected = {}

for i, var_name in enumerate(['X1', 'X2', 'X3']):
    gt_value = ground_truth_importance[var_name]
    # Wasserstein distance to point mass at ground truth
    distances_original[var_name] = np.mean(np.abs(rid_original.rid_with_counts[i] - gt_value))
    distances_corrected[var_name] = np.mean(np.abs(rid_corrected.rid_with_counts[i] - gt_value))

# Create Table 2
table_2 = pd.DataFrame({
    'Method': ['Original RID', 'PE Corrected RID'],
    'X1': [distances_original['X1'], distances_corrected['X1']],
    'X2': [distances_original['X2'], distances_corrected['X2']],
    'X3': [distances_original['X3'], distances_corrected['X3']],
})

print("\n" + "=" * 80)
print("TABLE 2: 1-Wasserstein Distance to Ground Truth (REAL REPRODUCTION)")
print("(Lower is better - closer to ground truth importance)")
print("=" * 80)
print(table_2.to_string(index=False))
print("=" * 80)

table_2.to_csv("table_2_REAL_reproduction.csv", index=False)
print("\n✓ Table 2 saved as 'table_2_REAL_reproduction.csv'")

print("\n" + "=" * 80)
print("TABLE 2 REPRODUCTION COMPLETE!")
print("=" * 80)
