"""
Statistical significance testing utilities.
Computes paired t-test and Wilcoxon signed-rank test for optimizer comparisons.
"""

import numpy as np
from scipy import stats
import json


def compute_pairwise_tests(final_accs_1, final_accs_2, optimizer_1_name, optimizer_2_name):
    """
    Run paired t-test and Wilcoxon signed-rank test.
    
    Args:
        final_accs_1, final_accs_2: arrays of final test accuracies across runs
        optimizer_1_name, optimizer_2_name: names for reporting
    
    Returns:
        dict with t-test and Wilcoxon results
    """
    final_accs_1 = np.array(final_accs_1)
    final_accs_2 = np.array(final_accs_2)
    
    if len(final_accs_1) != len(final_accs_2):
        raise ValueError("Arrays must have same length (same number of runs)")
    
    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(final_accs_1, final_accs_2)
    
    # Wilcoxon signed-rank test
    wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(final_accs_1 - final_accs_2)
    
    return {
        'comparison': f"{optimizer_1_name} vs {optimizer_2_name}",
        'optimizer_1': optimizer_1_name,
        'optimizer_2': optimizer_2_name,
        'n_runs': len(final_accs_1),
        'mean_1': float(np.mean(final_accs_1)),
        'std_1': float(np.std(final_accs_1)),
        'mean_2': float(np.mean(final_accs_2)),
        'std_2': float(np.std(final_accs_2)),
        'mean_diff': float(np.mean(final_accs_1 - final_accs_2)),
        't_test_statistic': float(t_stat),
        't_test_pvalue': float(t_pvalue),
        'wilcoxon_statistic': float(wilcoxon_stat),
        'wilcoxon_pvalue': float(wilcoxon_pvalue),
    }


def run_statistical_tests(runs_dict, available_optimizers):
    """
    Run all pairwise comparisons: Adam vs SR-Adam, Adam vs SR-Adam-All-Weights (if present).
    
    Args:
        runs_dict: {optimizer_name: [list of metrics dicts for each run]}
        available_optimizers: list of optimizer names present in runs_dict
    
    Returns:
        list of test result dicts
    """
    results = []
    
    # Only proceed if Adam is present
    if "Adam" not in available_optimizers:
        return results
    
    adam_final = [m['test_acc'][-1] for m in runs_dict["Adam"]]
    
    # Test Adam vs SR-Adam (conv-only)
    if "SR-Adam" in available_optimizers:
        sradam_final = [m['test_acc'][-1] for m in runs_dict["SR-Adam"]]
        test_result = compute_pairwise_tests(adam_final, sradam_final, "Adam", "SR-Adam")
        results.append(test_result)
    
    # Test Adam vs SR-Adam-All-Weights (if present)
    if "SR-Adam-All-Weights" in available_optimizers:
        sradam_all_final = [m['test_acc'][-1] for m in runs_dict["SR-Adam-All-Weights"]]
        test_result = compute_pairwise_tests(adam_final, sradam_all_final, "Adam", "SR-Adam-All-Weights")
        results.append(test_result)
    
    return results


def format_pvalue(pvalue, decimal_places=4):
    """Format p-value for display (e.g., '0.0234' or '<0.001')."""
    if pvalue < 0.001:
        return "<0.001"
    return f"{pvalue:.{decimal_places}f}"
