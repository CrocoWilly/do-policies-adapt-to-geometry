"""
Script to plot grouped alignment phase behavior analysis metrics from multiple experiments.

This script reads CSV files from multiple experiments (e.g., SmallExp3-1, 3-2, 3-3)
and creates grouped bar charts for:
    1. Final Yaw Error comparison across experiments
    2. Alignment Wiggle Index comparison across experiments
    3. Yaw Correction comparison across experiments

Usage:
    python Plot_Diagnosis2_alignment_grouped.py --input ./smallExp3/*_alignment.csv
    python Plot_Diagnosis2_alignment_grouped.py --input file1.csv file2.csv --title "SmallExp3"
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
        "final_yaw_error_median": 0.0,
        "final_yaw_error_min": 0.0,
        "final_yaw_error_max": 0.0,
        "alignment_steps_median": 0,
        "alignment_steps_min": 0,
        "alignment_steps_max": 0,
        "alignment_wiggle_median": 0.0,
        "alignment_wiggle_min": 0.0,
        "alignment_wiggle_max": 0.0,
        "yaw_correction_median": 0.0,
        "yaw_correction_min": 0.0,
        "yaw_correction_max": 0.0,
    }
    
    with open(path, "r") as f:
        content = f.read()
    
    # Extract final_yaw_error statistics
    match = re.search(r"# Median final_yaw_error:\s*([\d.]+)\s*rad", content)
    if match:
        result["final_yaw_error_median"] = float(match.group(1))
    
    match = re.search(r"# Min final_yaw_error:\s*([\d.]+)\s*rad", content)
    if match:
        result["final_yaw_error_min"] = float(match.group(1))
    
    match = re.search(r"# Max final_yaw_error:\s*([\d.]+)\s*rad", content)
    if match:
        result["final_yaw_error_max"] = float(match.group(1))
    
    # Extract alignment_steps statistics
    match = re.search(r"# Median alignment_steps:\s*(\d+)", content)
    if match:
        result["alignment_steps_median"] = int(match.group(1))
    
    match = re.search(r"# Min alignment_steps:\s*(\d+)", content)
    if match:
        result["alignment_steps_min"] = int(match.group(1))
    
    match = re.search(r"# Max alignment_steps:\s*(\d+)", content)
    if match:
        result["alignment_steps_max"] = int(match.group(1))
    
    # Extract alignment_wiggle statistics
    match = re.search(r"# Median alignment_wiggle:\s*([\d.]+)", content)
    if match:
        result["alignment_wiggle_median"] = float(match.group(1))
    
    match = re.search(r"# Min alignment_wiggle:\s*([\d.]+)", content)
    if match:
        result["alignment_wiggle_min"] = float(match.group(1))
    
    match = re.search(r"# Max alignment_wiggle:\s*([\d.]+)", content)
    if match:
        result["alignment_wiggle_max"] = float(match.group(1))
    
    # Extract yaw_correction statistics
    match = re.search(r"# Median yaw_correction:\s*([\d.]+)\s*rad", content)
    if match:
        result["yaw_correction_median"] = float(match.group(1))
    
    match = re.search(r"# Min yaw_correction:\s*([\d.]+)\s*rad", content)
    if match:
        result["yaw_correction_min"] = float(match.group(1))
    
    match = re.search(r"# Max yaw_correction:\s*([\d.]+)\s*rad", content)
    if match:
        result["yaw_correction_max"] = float(match.group(1))
    
    return result


def extract_experiment_and_condition(filename: str) -> tuple:
    """
    Extract experiment name and condition from filename.
    
    Args:
        filename: e.g., SmallExp3-1_rollout100_clearance112_yaw-15_alignment
    
    Returns:
        (experiment, condition): e.g., ("3-1", "C112_Y-15")
    """
    # Remove _alignment suffix if present
    filename = re.sub(r"_alignment$", "", filename)
    
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


def plot_grouped_final_yaw_error(
    grouped_data: Dict[str, Dict[str, tuple]],
    experiments: List[str],
    conditions: List[str],
    title: str = "SmallExp3 Final Yaw Error (Median)",
    output_path: str = None,
):
    """
    Plot grouped final yaw error bar chart.
    
    Args:
        grouped_data: {condition: {experiment: (median, min, max)}}
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
        # Convert from rad to deg
        medians = [np.degrees(grouped_data[cond].get(exp, (0, 0, 0))[0]) for cond in conditions]
        # mins = [np.degrees(grouped_data[cond].get(exp, (0, 0, 0))[1]) for cond in conditions]
        # maxs = [np.degrees(grouped_data[cond].get(exp, (0, 0, 0))[2]) for cond in conditions]
        
        # Calculate asymmetric error bars
        # lower_errors = [m - mn for m, mn in zip(medians, mins)]
        # upper_errors = [mx - m for m, mx in zip(medians, maxs)]
        
        offset = (i - (n_experiments - 1) / 2) * width
        display_label = EXPERIMENT_DISPLAY_LABELS.get(exp, exp)
        bars = ax.bar(x + offset, medians, width,
                      label=display_label, color=colors[i], edgecolor='black')
        
        # Add value labels on bars
        for bar, med in zip(bars, medians):
            height = bar.get_height()
            ax.annotate(f'{med:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Final Yaw Error (deg)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=0, ha='center')
    ax.legend(title="Experiment")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Final yaw error plot saved to: {output_path}")
    
    plt.show()


def plot_grouped_alignment_wiggle(
    grouped_data: Dict[str, Dict[str, tuple]],
    experiments: List[str],
    conditions: List[str],
    title: str = "SmallExp3 Alignment Wiggle Index (Median)",
    output_path: str = None,
):
    """
    Plot grouped alignment wiggle bar chart.
    
    Args:
        grouped_data: {condition: {experiment: (median, min, max)}}
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
        # Convert from m to cm
        medians = [grouped_data[cond].get(exp, (0, 0, 0))[0] * 100 for cond in conditions]
        # mins = [grouped_data[cond].get(exp, (0, 0, 0))[1] * 100 for cond in conditions]
        # maxs = [grouped_data[cond].get(exp, (0, 0, 0))[2] * 100 for cond in conditions]
        
        # Calculate asymmetric error bars
        # lower_errors = [m - mn for m, mn in zip(medians, mins)]
        # upper_errors = [mx - m for m, mx in zip(medians, maxs)]
        
        offset = (i - (n_experiments - 1) / 2) * width
        display_label = EXPERIMENT_DISPLAY_LABELS.get(exp, exp)
        bars = ax.bar(x + offset, medians, width,
                      label=display_label, color=colors[i], edgecolor='black')
        
        # Add value labels on bars
        for bar, med in zip(bars, medians):
            height = bar.get_height()
            ax.annotate(f'{med:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Alignment Wiggle Index (cm)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=0, ha='center')
    ax.legend(title="Experiment")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Alignment wiggle plot saved to: {output_path}")
    
    plt.show()


def plot_grouped_alignment_steps(
    grouped_data: Dict[str, Dict[str, tuple]],
    experiments: List[str],
    conditions: List[str],
    title: str = "SmallExp3 Alignment Steps (Median)",
    output_path: str = None,
):
    """
    Plot grouped alignment steps bar chart.
    
    Args:
        grouped_data: {condition: {experiment: (median, min, max)}}
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
        medians = [grouped_data[cond].get(exp, (0, 0, 0))[0] for cond in conditions]
        
        offset = (i - (n_experiments - 1) / 2) * width
        display_label = EXPERIMENT_DISPLAY_LABELS.get(exp, exp)
        bars = ax.bar(x + offset, medians, width,
                      label=display_label, color=colors[i], edgecolor='black')
        
        # Add value labels on bars
        for bar, med in zip(bars, medians):
            height = bar.get_height()
            ax.annotate(f'{int(med)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Alignment Steps", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=0, ha='center')
    ax.legend(title="Experiment")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Alignment steps plot saved to: {output_path}")
    
    plt.show()


def plot_grouped_yaw_correction(
    grouped_data: Dict[str, Dict[str, tuple]],
    experiments: List[str],
    conditions: List[str],
    title: str = "SmallExp3 Yaw Correction (Median)",
    output_path: str = None,
):
    """
    Plot grouped yaw correction bar chart.
    
    Args:
        grouped_data: {condition: {experiment: (median, min, max)}}
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
        # Convert from rad to deg
        medians = [np.degrees(grouped_data[cond].get(exp, (0, 0, 0))[0]) for cond in conditions]
        # mins = [np.degrees(grouped_data[cond].get(exp, (0, 0, 0))[1]) for cond in conditions]
        # maxs = [np.degrees(grouped_data[cond].get(exp, (0, 0, 0))[2]) for cond in conditions]
        
        # Calculate asymmetric error bars
        # lower_errors = [m - mn for m, mn in zip(medians, mins)]
        # upper_errors = [mx - m for m, mx in zip(medians, maxs)]
        
        offset = (i - (n_experiments - 1) / 2) * width
        display_label = EXPERIMENT_DISPLAY_LABELS.get(exp, exp)
        bars = ax.bar(x + offset, medians, width,
                      label=display_label, color=colors[i], edgecolor='black')
        
        # Add value labels on bars
        for bar, med in zip(bars, medians):
            height = bar.get_height()
            ax.annotate(f'{med:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Yaw Correction (deg)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=0, ha='center')
    ax.legend(title="Experiment")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Yaw correction plot saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot grouped alignment phase behavior analysis metrics from multiple experiments"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        nargs="+",
        required=True,
        help="Paths to input CSV files (alignment CSV files)"
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
    final_yaw_error_data = defaultdict(dict)  # {condition: {experiment: (median, min, max)}}
    alignment_steps_data = defaultdict(dict)
    alignment_wiggle_data = defaultdict(dict)
    yaw_correction_data = defaultdict(dict)
    experiments_set = set()
    conditions_set = set()
    
    for csv_path in args.input:
        try:
            data = parse_csv_summary(csv_path)
            experiment, condition = extract_experiment_and_condition(data["filename"])
            
            final_yaw_error_data[condition][experiment] = (
                data["final_yaw_error_median"],
                data["final_yaw_error_min"],
                data["final_yaw_error_max"]
            )
            alignment_steps_data[condition][experiment] = (
                data["alignment_steps_median"],
                data["alignment_steps_min"],
                data["alignment_steps_max"]
            )
            alignment_wiggle_data[condition][experiment] = (
                data["alignment_wiggle_median"],
                data["alignment_wiggle_min"],
                data["alignment_wiggle_max"]
            )
            yaw_correction_data[condition][experiment] = (
                data["yaw_correction_median"],
                data["yaw_correction_min"],
                data["yaw_correction_max"]
            )
            
            experiments_set.add(experiment)
            conditions_set.add(condition)
            
            print(f"Parsed: {csv_path} -> Exp {experiment}, {condition}")
        except Exception as e:
            print(f"[ERROR] Failed to parse {csv_path}: {e}")
    
    if not final_yaw_error_data:
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
    
    final_yaw_error_path = output_dir / f"{args.title}_grouped_final_yaw_error.png"
    alignment_steps_path = output_dir / f"{args.title}_grouped_alignment_steps.png"
    alignment_wiggle_path = output_dir / f"{args.title}_grouped_alignment_wiggle.png"
    yaw_correction_path = output_dir / f"{args.title}_grouped_yaw_correction.png"
    
    print("\n" + "=" * 70)
    print("Generating grouped plots...")
    print("=" * 70)
    
    # Plot grouped final yaw error
    plot_grouped_final_yaw_error(
        final_yaw_error_data,
        experiments,
        conditions,
        title=f"{args.title} Final Yaw Error (Median)",
        output_path=str(final_yaw_error_path)
    )
    
    # Plot grouped alignment steps
    plot_grouped_alignment_steps(
        alignment_steps_data,
        experiments,
        conditions,
        title=f"{args.title} Alignment Steps (Median)",
        output_path=str(alignment_steps_path)
    )
    
    # Plot grouped alignment wiggle
    plot_grouped_alignment_wiggle(
        alignment_wiggle_data,
        experiments,
        conditions,
        title=f"{args.title} Alignment Wiggle Index (Median)",
        output_path=str(alignment_wiggle_path)
    )
    
    # Plot grouped yaw correction
    plot_grouped_yaw_correction(
        yaw_correction_data,
        experiments,
        conditions,
        title=f"{args.title} Yaw Correction (Median)",
        output_path=str(yaw_correction_path)
    )


if __name__ == "__main__":
    main()
