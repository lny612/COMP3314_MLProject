"""
REAL Reproduction of Table 2 and Figure 5
Using the actual TreeFARMS and DNF algorithms from the paper's repository
"""

from rid.rashomon_importance_distribution import RashomonImportanceDistribution as RID
from models.treefarms_wrapper import DEFAULT_TF_CONFIG
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
from scipy import stats

print("=" * 80)
print("REAL REPRODUCTION OF TABLE 2 AND FIGURE 5")
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

print("\n[1/2] Reproducing Table 2 - Synthetic Dataset...")
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

print("\n[2/2] Reproducing Figure 5 - COMPAS Dataset...")

# Load COMPAS data
compas_data = pd.read_csv("data/compas_complete.csv")
compas_data = compas_data.dropna(axis=0).reset_index(drop=True)

print(f"\nComputing RID for COMPAS dataset (this will take longer)...")
print(f"  Dataset shape: {compas_data.shape}")

# Compute RID for COMPAS WITHOUT PE correction
print("\n  Computing Original RID...")
rid_compas_original = RID(
    compas_data,
    n_resamples=n_resamples,
    tf_config=tf_config,
    binning_fn=lambda x: x,
    use_test_df=False,
    tree_type="default",
    dataset_name="compas_figure5",
    allow_binarize_internally=True,  # Let RID handle binarization
    verbose=True
)

# Compute RID for COMPAS WITH PE correction
print("\n  Computing PE-Corrected RID...")
rid_compas_corrected = RID(
    compas_data,
    n_resamples=n_resamples,
    tf_config=tf_config,
    binning_fn=lambda x: x,
    use_test_df=False,
    unique_trees_only=True,  # Remove duplicates via DNF
    tree_type="DNF",
    dataset_name="compas_figure5_corrected",
    allow_binarize_internally=True,
    verbose=True
)

# Plot Figure 5 for three key variables
# Note: The exact variables depend on what's in the dataset after binarization
# We'll plot the first 3 variables as an example
variables_to_plot = min(3, len(compas_data.columns) - 1)  # Exclude target
var_names = list(compas_data.columns[:variables_to_plot])

print(f"\n  Creating Figure 5 plots for variables: {var_names}")

fig, axes = plt.subplots(1, variables_to_plot, figsize=(15, 4))
if variables_to_plot == 1:
    axes = [axes]

for idx, var_name in enumerate(var_names):
    ax = axes[idx]

    # Get importance distributions
    original_importances = rid_compas_original.rid_with_counts[idx]
    corrected_importances = rid_compas_corrected.rid_with_counts[idx]

    # Plot histograms
    ax.hist(original_importances, bins=30, alpha=0.6,
            label='Original RID', color='blue', density=True)
    ax.hist(corrected_importances, bins=30, alpha=0.6,
            label='PE Corrected RID', color='orange', density=True)

    ax.set_xlabel('Model Reliance Value', fontsize=12)
    ax.set_ylabel('Percent', fontsize=12)
    ax.set_title(f'RID Distribution of {var_name}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Compute KS test
    ks_test = stats.ks_2samp(original_importances, corrected_importances)
    print(f"\n  {var_name}: KS statistic = {ks_test.statistic:.3f}, p-value = {ks_test.pvalue:.6f}")

plt.tight_layout()
plt.savefig("figure_5_REAL_reproduction.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figure_5_REAL_reproduction.png", dpi=300, bbox_inches='tight')
print("\n✓ Figure 5 saved as 'figure_5_REAL_reproduction.pdf' and '.png'")
plt.close()

print("\n" + "=" * 80)
print("REAL REPRODUCTION COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  1. table_2_REAL_reproduction.csv - Wasserstein distances")
print("  2. figure_5_REAL_reproduction.pdf/png - RID distributions")
print("\nKey differences from fake reproduction:")
print("  ✓ Used actual TreeFARMS to generate Rashomon sets")
print("  ✓ Used DNF algorithm to detect and remove duplicate trees")
print("  ✓ Computed real RID over 100 bootstrap samples")
print("  ✓ Results should match the paper!")
print("=" * 80)
