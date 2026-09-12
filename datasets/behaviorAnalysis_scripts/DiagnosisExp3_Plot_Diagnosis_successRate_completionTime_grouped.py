#!/usr/bin/env python3
"""
Script to plot SmallExp7 success rate comparing case1 vs case2 for each experiment group.

This script:
1. Reads CSV files from SmallExp7 experiments (7-1, 7-2, 7-3)
2. Treats *_case1.csv and *_case2.csv as different conditions
3. Creates separate bar charts for each experiment group showing case1 vs case2 success rates

Usage:
    python SmallExp7_Plot_Diagnosis1_successRate_completionTime_grouped.py --input ./smallExp7_v1/anaylzed_csv/*.csv
    python SmallExp7_Plot_Diagnosis1_successRate_completionTime_grouped.py --input ./smallExp7_v1/anaylzed_csv/*.csv --output-dir ./smallExp7_v1/plots
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Mapping from experiment names to display labels
EXPERIMENT_DISPLAY_LABELS = {
    "7-1": "Explicit",
    "7-2": "Semi-implicit",
    "7-3": "Implicit",
}

# Mapping from case to display labels
CASE_DISPLAY_LABELS = {
    "case1": "Case 1\n(No Rotation)",
    "case2": "Case 2\n(Rotate 90°)",
}


def parse_csv_summary(csv_path: str) -> Dict:
    """
    Parse a CSV file and extract summary statistics.
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        Dictionary with filename, success_rate, median_completion_time, std_completion_time
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    result = {
        "filename": path.stem,
        "success_rate": 0.0,
        "median_completion_time": 0.0,
        "std_completion_time": 0.0,
    }
    
    with open(path, "r") as f:
        content = f.read()
    
    # Extract success rate
    match = re.search(r"# Success rate:\s*([\d.]+)%", content)
    if match:
        result["success_rate"] = float(match.group(1))
    
    # Extract median completion time and std (new format)
    match = re.search(r"# Median completion time:\s*([\d.]+)\s*steps\s*\(std:\s*([\d.]+)\)", content)
    if match:
        result["median_completion_time"] = float(match.group(1))
        result["std_completion_time"] = float(match.group(2))
    else:
        # Fallback to old format (avg completion time)
        match = re.search(r"# Avg completion time:\s*([\d.]+)\s*steps\s*\(std:\s*([\d.]+)\)", content)
        if match:
            result["median_completion_time"] = float(match.group(1))
            result["std_completion_time"] = float(match.group(2))
    
    return result


def extract_experiment_and_case(filename: str) -> tuple:
    """
    Extract experiment name and case from filename.
    
    Args:
        filename: e.g., SmallExp7-1_rollout100_clearance112_yaw0_case1
    
    Returns:
        (experiment, case): e.g., ("7-1", "case1")
    """
    # Extract experiment number (e.g., 7-1, 7-2, 7-3)
    exp_match = re.search(r"SmallExp(\d+-\d+)", filename)
    experiment = exp_match.group(1) if exp_match else "unknown"
    
    # Extract case (case1 or case2)
    case_match = re.search(r"(case\d+)", filename)
    case = case_match.group(1) if case_match else "unknown"
    
    return experiment, case


def plot_experiment_success_rate(
    cases_data: Dict[str, float],
    experiment: str,
    title: str = None,
    output_path: str = None,
):
    """
    Plot success rate bar chart for a single experiment comparing case1 vs case2.
    
    Args:
        cases_data: {case: success_rate}
        experiment: Experiment name (e.g., "7-1")
        title: Plot title (optional)
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Sort cases
    cases = sorted(cases_data.keys())
    success_rates = [cases_data[c] for c in cases]
    
    x = np.arange(len(cases))
    width = 0.5
    
    # Define colors: case1 steelblue, case2 coral
    colors = ['steelblue', 'coral'][:len(cases)]
    
    # Create bars
    bars = ax.bar(x, success_rates, width, color=colors, edgecolor='black')
    
    # Add value labels on bars
    for bar, val in zip(bars, success_rates):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Get display labels
    exp_label = EXPERIMENT_DISPLAY_LABELS.get(experiment, experiment)
    case_labels = [CASE_DISPLAY_LABELS.get(c, c) for c in cases]
    
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f"{exp_label} - Success Rate by Case", fontsize=14, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, fontsize=11)
    ax.set_ylim(0, 115)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()


def plot_all_experiments_comparison(
    grouped_data: Dict[str, Dict[str, float]],
    experiments: List[str],
    cases: List[str],
    title: str = "SmallExp7 Success Rate - All Experiments",
    output_path: str = None,
):
    """
    Plot grouped success rate bar chart comparing all experiments.
    
    Args:
        grouped_data: {experiment: {case: success_rate}}
        experiments: List of experiment names (e.g., ["7-1", "7-2", "7-3"])
        cases: List of cases (e.g., ["case1", "case2"])
        title: Plot title
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(experiments))
    width = 0.35
    n_cases = len(cases)
    
    # Define colors for cases
    colors = ['steelblue', 'coral'][:n_cases]
    
    for i, case in enumerate(cases):
        values = [grouped_data[exp].get(case, 0) for exp in experiments]
        offset = (i - (n_cases - 1) / 2) * width
        case_label = CASE_DISPLAY_LABELS.get(case, case).replace('\n', ' ')
        bars = ax.bar(x + offset, values, width, label=case_label, color=colors[i], edgecolor='black')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Get display labels for experiments
    exp_labels = [EXPERIMENT_DISPLAY_LABELS.get(exp, exp) for exp in experiments]
    
    ax.set_xlabel("Experiment", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontsize=11)
    ax.set_ylim(0, 120)
    ax.legend(title="Condition", loc='upper right', fontsize=10)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot SmallExp7 success rate comparing case1 vs case2 for each experiment"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        nargs="+",
        required=True,
        help="Paths to input CSV files"
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="SmallExp7",
        help="Title prefix for plots"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output plots"
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Only generate the combined comparison plot"
    )
    
    args = parser.parse_args()
    
    # Parse all CSV files and organize by experiment and case
    # {experiment: {case: success_rate}}
    grouped_data = defaultdict(dict)
    experiments_set = set()
    cases_set = set()
    
    for csv_path in args.input:
        try:
            data = parse_csv_summary(csv_path)
            experiment, case = extract_experiment_and_case(data["filename"])
            
            if experiment == "unknown" or case == "unknown":
                print(f"[SKIP] Could not parse experiment/case from: {csv_path}")
                continue
            
            grouped_data[experiment][case] = data["success_rate"]
            experiments_set.add(experiment)
            cases_set.add(case)
            
            print(f"Parsed: {csv_path}")
            print(f"  -> Experiment: {experiment}, Case: {case}, Success Rate: {data['success_rate']:.1f}%")
        except Exception as e:
            print(f"[ERROR] Failed to parse {csv_path}: {e}")
    
    if not grouped_data:
        print("No valid CSV files to plot.")
        return
    
    # Sort experiments and cases
    experiments = sorted(experiments_set)
    cases = sorted(cases_set)
    
    print(f"\nExperiments: {experiments}")
    print(f"Cases: {cases}")
    
    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)
    
    # Generate individual plots for each experiment
    if not args.combined_only:
        for exp in experiments:
            if exp not in grouped_data:
                continue
            
            exp_label = EXPERIMENT_DISPLAY_LABELS.get(exp, exp)
            output_path = output_dir / f"{args.title_prefix}_{exp}_case_comparison.png"
            
            print(f"\n[{exp}] Generating plot...")
            print(f"  Cases: {grouped_data[exp]}")
            
            plot_experiment_success_rate(
                grouped_data[exp],
                exp,
                title=f"{exp_label} - Success Rate by Case",
                output_path=str(output_path)
            )
    
    # Generate combined comparison plot
    print("\n[Combined] Generating comparison plot...")
    combined_output_path = output_dir / f"{args.title_prefix}_all_experiments_comparison.png"
    
    plot_all_experiments_comparison(
        grouped_data,
        experiments,
        cases,
        title="Diagnosis3 - Success Rate Comparison",
        output_path=str(combined_output_path)
    )
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Experiment':<15} {'Case 1 (%)':<15} {'Case 2 (%)':<15}")
    print("-" * 45)
    for exp in experiments:
        case1_rate = grouped_data[exp].get('case1', 'N/A')
        case2_rate = grouped_data[exp].get('case2', 'N/A')
        case1_str = f"{case1_rate:.1f}" if isinstance(case1_rate, float) else case1_rate
        case2_str = f"{case2_rate:.1f}" if isinstance(case2_rate, float) else case2_rate
        exp_label = EXPERIMENT_DISPLAY_LABELS.get(exp, "")
        print(f"{exp} ({exp_label}){' '*(10-len(exp_label))} {case1_str:<15} {case2_str:<15}")
    print("=" * 70)


if __name__ == "__main__":
    main()
