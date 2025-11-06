"""
Script to reproduce Case Study 2 results from the paper:
- Table 3: Rashomon set sizes for additional datasets
- Tables 4 & 5: KS test statistics and p-values for RID distribution shifts
- Figures 9 & 10: RID distribution histograms

This script runs the RID comparison for all available datasets and aggregates the results.
"""

import subprocess
import pandas as pd
import os
import sys
import time
from pathlib import Path

# List of datasets (indices 0-11)
DATASETS = [
    'tic-tac-toe.csv',           # 0
    'wisconsin.csv',             # 1
    'coupon_full.csv',           # 2
    'compas_complete.csv',       # 3
    'wine_quality.csv',          # 4
    'broward_general_2y.csv',    # 5
    'fico_complete.csv',         # 6
    'iris_setosa.csv',           # 7
    'iris_versicolor.csv',       # 8
    'iris_virginica.csv',        # 9
    'spiral.csv',                # 10
    'netherlands.csv'            # 11 (may not exist)
]

def check_dataset_exists(dataset_idx):
    """Check if a dataset file exists"""
    dataset_file = DATASETS[dataset_idx]
    data_path = Path('data') / dataset_file
    return data_path.exists()

def run_rid_for_dataset(dataset_idx, verbose=True):
    """Run RID comparison for a single dataset"""
    dataset_name = DATASETS[dataset_idx]

    if not check_dataset_exists(dataset_idx):
        print(f"Skipping dataset {dataset_idx} ({dataset_name}) - file not found")
        return False

    print(f"\n{'='*80}")
    print(f"Running RID comparison for dataset {dataset_idx}: {dataset_name}")
    print(f"{'='*80}\n")

    try:
        # Run the RID comparison script
        result = subprocess.run(
            [sys.executable, '-m', 'experiments.run_rid_comparison', str(dataset_idx)],
            capture_output=not verbose,
            text=True,
            timeout=1800  # 30 minute timeout
        )

        if result.returncode == 0:
            print(f"✓ Successfully completed dataset {dataset_idx}: {dataset_name}")
            return True
        else:
            print(f"✗ Failed to process dataset {dataset_idx}: {dataset_name}")
            if not verbose and result.stderr:
                print(f"Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"✗ Timeout processing dataset {dataset_idx}: {dataset_name}")
        return False
    except Exception as e:
        print(f"✗ Error processing dataset {dataset_idx}: {dataset_name}")
        print(f"Error: {str(e)}")
        return False

def collect_results():
    """Collect all summary CSV files and create aggregated tables"""
    print(f"\n{'='*80}")
    print("Collecting and aggregating results...")
    print(f"{'='*80}\n")

    results_dir = Path('rid_plots_1_22')
    if not results_dir.exists():
        print("No results directory found!")
        return

    # Collect all summary files
    all_summaries = []
    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            summary_file = subdir / 'summary.csv'
            if summary_file.exists():
                try:
                    df = pd.read_csv(summary_file)
                    df['dataset'] = subdir.name
                    all_summaries.append(df)
                    print(f"✓ Loaded results from {subdir.name}")
                except Exception as e:
                    print(f"✗ Error loading {summary_file}: {e}")

    if not all_summaries:
        print("No summary files found!")
        return

    # Combine all results
    combined_df = pd.concat(all_summaries, ignore_index=True)

    # Save combined results
    output_file = 'case_study_2_combined_results.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved combined results to {output_file}")

    # Create summary tables similar to Tables 4 & 5 in the paper
    print("\n" + "="*80)
    print("Summary Statistics (similar to Tables 4 & 5)")
    print("="*80 + "\n")

    # Group by dataset and variable
    summary_stats = combined_df.groupby(['dataset', 'var']).agg({
        'distance': 'first',
        'ks_test_stat': 'first',
        'ks_test_p': 'first'
    }).reset_index()

    # Save summary statistics
    summary_file = 'case_study_2_summary_statistics.csv'
    summary_stats.to_csv(summary_file, index=False)
    print(f"✓ Saved summary statistics to {summary_file}")

    # Display some key results
    print("\nKey Results:")
    print("-" * 80)

    # Show variables with significant distribution shift (p < 0.05)
    significant = summary_stats[summary_stats['ks_test_p'] < 0.05]
    print(f"\nVariables with significant RID distribution shift (p < 0.05): {len(significant)}")
    print(f"Total variables analyzed: {len(summary_stats)}")

    # Show top 10 largest distribution shifts
    print("\nTop 10 largest Wasserstein distances:")
    print(summary_stats.nlargest(10, 'distance')[['dataset', 'var', 'distance', 'ks_test_stat', 'ks_test_p']])

    return combined_df, summary_stats

def main():
    """Main function to run all datasets and collect results"""
    print("="*80)
    print("Reproducing Case Study 2: Additional Datasets")
    print("="*80)
    print("\nThis script will:")
    print("1. Run RID comparison for all available datasets")
    print("2. Generate Tables 3, 4, 5 (Rashomon set sizes and KS test statistics)")
    print("3. Generate Figures 9 & 10 (RID distribution plots)")
    print("\nNote: This may take 1-2 hours to complete depending on dataset sizes.")

    # Ask user if they want to run all or specific datasets
    response = input("\nRun all datasets? (y/n, or enter specific indices like '0,3,4'): ").strip().lower()

    if response == 'y' or response == 'yes':
        dataset_indices = range(len(DATASETS))
    elif response == 'n' or response == 'no':
        print("Skipping dataset processing. Will only collect existing results.")
        dataset_indices = []
    else:
        try:
            dataset_indices = [int(x.strip()) for x in response.split(',')]
        except:
            print("Invalid input. Running all datasets.")
            dataset_indices = range(len(DATASETS))

    # Run RID comparison for selected datasets
    start_time = time.time()
    successful = 0
    failed = 0
    skipped = 0

    for idx in dataset_indices:
        if idx < 0 or idx >= len(DATASETS):
            print(f"Skipping invalid index: {idx}")
            continue

        result = run_rid_for_dataset(idx, verbose=True)
        if result:
            successful += 1
        elif result is False:
            failed += 1
        else:
            skipped += 1

    elapsed_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Processing Summary")
    print(f"{'='*80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Time elapsed: {elapsed_time/60:.1f} minutes")

    # Collect and aggregate results
    if successful > 0 or len(list(Path('rid_plots_1_22').iterdir() if Path('rid_plots_1_22').exists() else [])) > 0:
        collect_results()

    print(f"\n{'='*80}")
    print("Case Study 2 Reproduction Complete!")
    print(f"{'='*80}")
    print("\nResults saved in:")
    print("  - rid_plots_1_22/          (individual dataset results)")
    print("  - case_study_2_combined_results.csv")
    print("  - case_study_2_summary_statistics.csv")

if __name__ == '__main__':
    main()
