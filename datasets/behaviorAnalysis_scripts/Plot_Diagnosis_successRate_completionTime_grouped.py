"""
Script to plot grouped success rate and completion time from multiple experiments.

This script reads CSV files from multiple experiments (e.g., SmallExp3-1, 3-2, 3-3)
and creates grouped bar charts where each condition (clearance_yaw) is a group.

Usage:
    python Plot_Diagnosis1_grouped.py --input ./smallExp3/SmallExp3-1_rollout100_clearance112_yaw-15.csv ... 
    python Plot_Diagnosis1_grouped.py --input ./smallExp3/*.csv --title "SmallExp3" --output-dir ./smallExp3/grouped_plots
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Mapping from experiment names to display labels
EXPERIMENT_DISPLAY_LABELS = {
    "3-1": "Explicit",
    "3-2": "Semi-implicit",
    "3-3": "Implicit",
    "4-1": "Explicit",
    "4-2": "Semi-implicit",
    "4-3": "Implicit",
    "5-1": "Explicit",
    "5-2": "Semi-implicit",
    "5-3": "Implicit",
    "6-1": "Explicit",
    "6-2": "Semi-implicit",
    "6-3": "Implicit",
    "7-1": "Explicit",
    "7-2": "Semi-implicit",
    "7-3": "Implicit",
}


def parse_csv_summary(csv_path: str) -> Dict:
    """
    Parse a CSV file and extract summary statistics.
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


def extract_experiment_and_condition(filename: str) -> tuple:
    """
    Extract experiment name and condition from filename.
    
    Args:
        filename: e.g., SmallExp3-1_rollout100_clearance112_yaw-15
    
    Returns:
        (experiment, condition): e.g., ("3-1", "C112_Y-15")
    """
    # Extract experiment number (e.g., 3-1, 3-2, 3-3)
    exp_match = re.search(r"SmallExp(\d+-\d+)", filename)
    experiment = exp_match.group(1) if exp_match else "unknown"
    
    # Extract condition (clearance and yaw)
    cond_match = re.search(r"clearance(\d+)_yaw(-?\d+)", filename)
    if cond_match:
        clearance = cond_match.group(1)
        yaw = cond_match.group(2)
        condition = f"C{clearance}_Y{yaw}"
    else:
        condition = "unknown"
    
    return experiment, condition


def plot_grouped_success_rate(
    grouped_data: Dict[str, Dict[str, float]],
    experiments: List[str],
    conditions: List[str],
    title: str = "SmallExp3 Success Rate (%)",
    output_path: str = None,
):
    """
    Plot grouped success rate bar chart.
    
    Args:
        grouped_data: {condition: {experiment: success_rate}}
        experiments: List of experiment names (e.g., ["3-1", "3-2", "3-3"])
        conditions: List of conditions (e.g., ["C112_Y-15", "C112_Y15", ...])
        title: Plot title
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(conditions))
    width = 0.25  # width of each bar
    n_experiments = len(experiments)
    
    # Define colors for each experiment
    colors = ['steelblue', 'coral', 'seagreen', 'orchid', 'gold'][:n_experiments]
    
    for i, exp in enumerate(experiments):
        values = [grouped_data[cond].get(exp, 0) for cond in conditions]
        offset = (i - (n_experiments - 1) / 2) * width
        display_label = EXPERIMENT_DISPLAY_LABELS.get(exp, exp)
        bars = ax.bar(x + offset, values, width, label=display_label, color=colors[i], edgecolor='black')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=0, ha='center')
    ax.set_ylim(0, 115)
    ax.legend(title="Experiment")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Success rate plot saved to: {output_path}")
    
    plt.show()


def plot_grouped_completion_time(
    grouped_data: Dict[str, Dict[str, tuple]],
    experiments: List[str],
    conditions: List[str],
    title: str = "SmallExp3 Completion Steps (Median, Success Only)",
    output_path: str = None,
):
    """
    Plot grouped completion time bar chart.
    
    Args:
        grouped_data: {condition: {experiment: (median, std)}}
        experiments: List of experiment names
        conditions: List of conditions
        title: Plot title
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(conditions))
    width = 0.25
    n_experiments = len(experiments)
    
    colors = ['steelblue', 'coral', 'seagreen', 'orchid', 'gold'][:n_experiments]
    
    for i, exp in enumerate(experiments):
        medians = [grouped_data[cond].get(exp, (0, 0))[0] for cond in conditions]
        stds = [grouped_data[cond].get(exp, (0, 0))[1] for cond in conditions]
        offset = (i - (n_experiments - 1) / 2) * width
        display_label = EXPERIMENT_DISPLAY_LABELS.get(exp, exp)
        bars = ax.bar(x + offset, medians, width, yerr=stds, capsize=3,
                      label=display_label, color=colors[i], edgecolor='black', error_kw={'elinewidth': 1.5})
        
        # Add value labels on bars
        for bar, med, std in zip(bars, medians, stds):
            height = bar.get_height()
            ax.annotate(f'{med:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height + std + 1),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Completion Steps", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=0, ha='center')
    ax.legend(title="Experiment")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Completion time plot saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot grouped success rate and completion time from multiple experiments"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        nargs="+",
        required=True,
        help="Paths to input CSV files"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="SmallExp3",
        help="Title prefix for plots"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output plots"
    )
    
    args = parser.parse_args()
    
    # Parse all CSV files and organize by experiment and condition
    success_rate_data = defaultdict(dict)  # {condition: {experiment: success_rate}}
    completion_time_data = defaultdict(dict)  # {condition: {experiment: (median, std)}}
    experiments_set = set()
    conditions_set = set()
    
    for csv_path in args.input:
        try:
            data = parse_csv_summary(csv_path)
            experiment, condition = extract_experiment_and_condition(data["filename"])
            
            success_rate_data[condition][experiment] = data["success_rate"]
            completion_time_data[condition][experiment] = (
                data["median_completion_time"],
                data["std_completion_time"]
            )
            
            experiments_set.add(experiment)
            conditions_set.add(condition)
            
            print(f"Parsed: {csv_path} -> Exp {experiment}, {condition}")
        except Exception as e:
            print(f"[ERROR] Failed to parse {csv_path}: {e}")
    
    if not success_rate_data:
        print("No valid CSV files to plot.")
        return
    
    # Sort experiments and conditions for consistent ordering
    experiments = sorted(experiments_set)
    # Custom sort conditions: C112 before C160, Y-15 before Y15
    def condition_sort_key(c):
        match = re.search(r"C(\d+)_Y(-?\d+)", c)
        if match:
            clearance = int(match.group(1))
            yaw = int(match.group(2))
            return (clearance, yaw)
        return (0, 0)
    
    conditions = sorted(conditions_set, key=condition_sort_key)
    
    print(f"\nExperiments: {experiments}")
    print(f"Conditions: {conditions}")
    
    # Determine output paths
    output_dir = Path(args.output_dir) if args.output_dir else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_rate_path = output_dir / f"{args.title}_grouped_success_rate.png"
    completion_time_path = output_dir / f"{args.title}_grouped_completion_steps.png"
    
    print("\n" + "=" * 70)
    print("Generating grouped plots...")
    print("=" * 70)
    
    # Plot grouped success rate
    plot_grouped_success_rate(
        success_rate_data,
        experiments,
        conditions,
        title=f"{args.title} Success Rate (%)",
        output_path=str(success_rate_path)
    )
    
    # Plot grouped completion time
    plot_grouped_completion_time(
        completion_time_data,
        experiments,
        conditions,
        title=f"{args.title} Completion Steps (Median, Success Only)",
        output_path=str(completion_time_path)
    )


if __name__ == "__main__":
    main()
