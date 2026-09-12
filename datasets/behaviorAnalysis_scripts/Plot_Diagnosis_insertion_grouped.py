"""
Script to plot grouped insertion phase behavior analysis metrics from multiple experiments.

This script reads CSV files from multiple experiments (e.g., SmallExp3-1, 3-2, 3-3)
and creates grouped bar charts for SUCCESS EPISODES ONLY:
    1. Contact Wiggle Index comparison across experiments
    2. Contact Wiggle Per Step comparison across experiments
    3. Contact Steps comparison across experiments
    4. Contact Push Speed comparison across experiments
    5. Mean dZ Per Step comparison across experiments
    6. Contact Pause Count comparison across experiments

Usage:
    python Plot_Diagnosis2_insertion_grouped.py --input ./smallExp3/*_insertion.csv
    python Plot_Diagnosis2_insertion_grouped.py --input file1.csv file2.csv --title "SmallExp3"
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
}


def parse_csv_summary(csv_path: str) -> Dict:
    """
    Parse a CSV file and extract summary statistics for SUCCESS episodes only.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    result = {
        "filename": path.stem,
        "contact_wiggle_median": 0.0,
        "contact_wiggle_min": 0.0,
        "contact_wiggle_max": 0.0,
        "contact_wiggle_per_step_median": 0.0,
        "contact_wiggle_per_step_min": 0.0,
        "contact_wiggle_per_step_max": 0.0,
        "contact_steps_median": 0.0,
        "contact_steps_min": 0.0,
        "contact_steps_max": 0.0,
        "contact_push_speed_median": 0.0,
        "contact_push_speed_min": 0.0,
        "contact_push_speed_max": 0.0,
        "mean_dz_per_step_median": 0.0,
        "mean_dz_per_step_min": 0.0,
        "mean_dz_per_step_max": 0.0,
        "contact_pause_count_median": 0.0,
        "contact_pause_count_min": 0.0,
        "contact_pause_count_max": 0.0,
        "success_count": 0,
        "total_count": 0,
    }
    
    with open(path, "r") as f:
        content = f.read()
    
    # Extract SUCCESS EPISODES section only
    success_match = re.search(
        r"\[SUCCESS EPISODES\].*?(?=\[FAILED EPISODES\]|$)",
        content,
        re.DOTALL
    )
    if not success_match:
        print(f"[WARNING] No SUCCESS EPISODES section found in {csv_path}")
        return result
    
    success_section = success_match.group(0)
    
    # Extract episode counts
    match = re.search(r"# Total Episodes:\s*(\d+)", content)
    if match:
        result["total_count"] = int(match.group(1))
    
    match = re.search(r"# Successful Episodes:\s*(\d+)", content)
    if match:
        result["success_count"] = int(match.group(1))
    
    # Extract contact_wiggle statistics from SUCCESS section
    match = re.search(r"# Median contact_wiggle:\s*([\d.]+)", success_section)
    if match:
        result["contact_wiggle_median"] = float(match.group(1))
    
    match = re.search(r"# Min contact_wiggle:\s*([\d.]+)", success_section)
    if match:
        result["contact_wiggle_min"] = float(match.group(1))
    
    match = re.search(r"# Max contact_wiggle:\s*([\d.]+)", success_section)
    if match:
        result["contact_wiggle_max"] = float(match.group(1))
    
    # Extract contact_wiggle_per_step statistics from SUCCESS section
    match = re.search(r"# Median contact_wiggle_per_step:\s*([\d.]+)", success_section)
    if match:
        result["contact_wiggle_per_step_median"] = float(match.group(1))
    
    match = re.search(r"# Min contact_wiggle_per_step:\s*([\d.]+)", success_section)
    if match:
        result["contact_wiggle_per_step_min"] = float(match.group(1))
    
    match = re.search(r"# Max contact_wiggle_per_step:\s*([\d.]+)", success_section)
    if match:
        result["contact_wiggle_per_step_max"] = float(match.group(1))
    
    # Extract contact_steps statistics from SUCCESS section
    match = re.search(r"# Median contact_steps:\s*(\d+)", success_section)
    if match:
        result["contact_steps_median"] = float(match.group(1))
    
    match = re.search(r"# Min contact_steps:\s*(\d+)", success_section)
    if match:
        result["contact_steps_min"] = float(match.group(1))
    
    match = re.search(r"# Max contact_steps:\s*(\d+)", success_section)
    if match:
        result["contact_steps_max"] = float(match.group(1))
    
    # Extract contact_push_speed statistics from SUCCESS section
    match = re.search(r"# Median contact_push_speed:\s*([\d.]+)", success_section)
    if match:
        result["contact_push_speed_median"] = float(match.group(1))
    
    match = re.search(r"# Min contact_push_speed:\s*([\d.]+)", success_section)
    if match:
        result["contact_push_speed_min"] = float(match.group(1))
    
    match = re.search(r"# Max contact_push_speed:\s*([\d.]+)", success_section)
    if match:
        result["contact_push_speed_max"] = float(match.group(1))
    
    # Extract mean_dz_per_step statistics from SUCCESS section
    match = re.search(r"# Median mean_dz_per_step:\s*([\d.]+)", success_section)
    if match:
        result["mean_dz_per_step_median"] = float(match.group(1))
    
    match = re.search(r"# Min mean_dz_per_step:\s*([\d.]+)", success_section)
    if match:
        result["mean_dz_per_step_min"] = float(match.group(1))
    
    match = re.search(r"# Max mean_dz_per_step:\s*([\d.]+)", success_section)
    if match:
        result["mean_dz_per_step_max"] = float(match.group(1))
    
    # Extract contact_pause_count statistics from SUCCESS section
    match = re.search(r"# Median contact_pause_count:\s*(\d+)", success_section)
    if match:
        result["contact_pause_count_median"] = float(match.group(1))
    
    match = re.search(r"# Min contact_pause_count:\s*(\d+)", success_section)
    if match:
        result["contact_pause_count_min"] = float(match.group(1))
    
    match = re.search(r"# Max contact_pause_count:\s*(\d+)", success_section)
    if match:
        result["contact_pause_count_max"] = float(match.group(1))
    
    return result


def extract_experiment_and_condition(filename: str) -> tuple:
    """
    Extract experiment name and condition from filename.
    
    Args:
        filename: e.g., SmallExp3-1_rollout100_clearance112_yaw-15_insertion
    
    Returns:
        (experiment, condition): e.g., ("3-1", "C112_Y-15")
    """
    # Remove _insertion suffix if present
    filename = re.sub(r"_insertion$", "", filename)
    
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


def plot_grouped_contact_wiggle(
    grouped_data: Dict[str, Dict[str, tuple]],
    experiments: List[str],
    conditions: List[str],
    title: str = "SmallExp3 Contact Wiggle Index (Median) - Success Only",
    output_path: str = None,
):
    """
    Plot grouped contact wiggle bar chart.
    
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
    ax.set_ylabel("Contact Wiggle Index (cm)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=0, ha='center')
    ax.legend(title="Experiment")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Contact wiggle plot saved to: {output_path}")
    
    plt.show()


def plot_grouped_contact_push_speed(
    grouped_data: Dict[str, Dict[str, tuple]],
    experiments: List[str],
    conditions: List[str],
    title: str = "SmallExp3 Contact Push Speed (Median) - Success Only",
    output_path: str = None,
):
    """
    Plot grouped contact push speed bar chart.
    
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
        # Convert from m/s to cm/s
        medians = [grouped_data[cond].get(exp, (0, 0, 0))[0] * 100 for cond in conditions]
        
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
    ax.set_ylabel("Contact Push Speed (cm/step)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=0, ha='center')
    ax.legend(title="Experiment")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Contact push speed plot saved to: {output_path}")
    
    plt.show()


def plot_grouped_contact_pause_count(
    grouped_data: Dict[str, Dict[str, tuple]],
    experiments: List[str],
    conditions: List[str],
    title: str = "SmallExp3 Contact Pause Count (Median) - Success Only",
    output_path: str = None,
):
    """
    Plot grouped contact pause count bar chart.
    
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
        # No unit conversion needed for count
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
    ax.set_ylabel("Contact Pause Count", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=0, ha='center')
    ax.legend(title="Experiment")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Contact pause count plot saved to: {output_path}")
    
    plt.show()


def plot_grouped_contact_steps(
    grouped_data: Dict[str, Dict[str, tuple]],
    experiments: List[str],
    conditions: List[str],
    title: str = "SmallExp3 Contact Steps (Median) - Success Only",
    output_path: str = None,
):
    """
    Plot grouped contact steps bar chart.
    
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
    ax.set_ylabel("Contact Steps", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=0, ha='center')
    ax.legend(title="Experiment")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Contact steps plot saved to: {output_path}")
    
    plt.show()


def plot_grouped_contact_wiggle_per_step(
    grouped_data: Dict[str, Dict[str, tuple]],
    experiments: List[str],
    conditions: List[str],
    title: str = "SmallExp3 Contact Wiggle Per Step (Median) - Success Only",
    output_path: str = None,
):
    """
    Plot grouped contact wiggle per step bar chart.
    
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
        
        offset = (i - (n_experiments - 1) / 2) * width
        display_label = EXPERIMENT_DISPLAY_LABELS.get(exp, exp)
        bars = ax.bar(x + offset, medians, width,
                      label=display_label, color=colors[i], edgecolor='black')
        
        # Add value labels on bars
        for bar, med in zip(bars, medians):
            height = bar.get_height()
            ax.annotate(f'{med:.6f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Contact Wiggle Per Step (cm/step)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=0, ha='center')
    ax.legend(title="Experiment")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Contact wiggle per step plot saved to: {output_path}")
    
    plt.show()


def plot_grouped_mean_dz_per_step(
    grouped_data: Dict[str, Dict[str, tuple]],
    experiments: List[str],
    conditions: List[str],
    title: str = "SmallExp3 Mean dZ Per Step (Median) - Success Only",
    output_path: str = None,
):
    """
    Plot grouped mean dZ per step bar chart.
    
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
        
        offset = (i - (n_experiments - 1) / 2) * width
        display_label = EXPERIMENT_DISPLAY_LABELS.get(exp, exp)
        bars = ax.bar(x + offset, medians, width,
                      label=display_label, color=colors[i], edgecolor='black')
        
        # Add value labels on bars
        for bar, med in zip(bars, medians):
            height = bar.get_height()
            ax.annotate(f'{med:.6f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Mean dZ Per Step (cm/step)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=0, ha='center')
    ax.legend(title="Experiment")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Mean dZ per step plot saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot grouped insertion phase behavior analysis metrics from multiple experiments"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        nargs="+",
        required=True,
        help="Paths to input CSV files (insertion CSV files)"
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
    contact_wiggle_data = defaultdict(dict)  # {condition: {experiment: (median, min, max)}}
    contact_wiggle_per_step_data = defaultdict(dict)
    contact_steps_data = defaultdict(dict)
    contact_push_speed_data = defaultdict(dict)
    mean_dz_per_step_data = defaultdict(dict)
    contact_pause_count_data = defaultdict(dict)
    experiments_set = set()
    conditions_set = set()
    
    for csv_path in args.input:
        try:
            data = parse_csv_summary(csv_path)
            experiment, condition = extract_experiment_and_condition(data["filename"])
            
            contact_wiggle_data[condition][experiment] = (
                data["contact_wiggle_median"],
                data["contact_wiggle_min"],
                data["contact_wiggle_max"]
            )
            contact_wiggle_per_step_data[condition][experiment] = (
                data["contact_wiggle_per_step_median"],
                data["contact_wiggle_per_step_min"],
                data["contact_wiggle_per_step_max"]
            )
            contact_steps_data[condition][experiment] = (
                data["contact_steps_median"],
                data["contact_steps_min"],
                data["contact_steps_max"]
            )
            contact_push_speed_data[condition][experiment] = (
                data["contact_push_speed_median"],
                data["contact_push_speed_min"],
                data["contact_push_speed_max"]
            )
            mean_dz_per_step_data[condition][experiment] = (
                data["mean_dz_per_step_median"],
                data["mean_dz_per_step_min"],
                data["mean_dz_per_step_max"]
            )
            contact_pause_count_data[condition][experiment] = (
                data["contact_pause_count_median"],
                data["contact_pause_count_min"],
                data["contact_pause_count_max"]
            )
            
            experiments_set.add(experiment)
            conditions_set.add(condition)
            
            print(f"Parsed: {csv_path} -> Exp {experiment}, {condition}")
            print(f"  Success Rate: {data['success_count']}/{data['total_count']} ({100*data['success_count']/max(1,data['total_count']):.1f}%)")
        except Exception as e:
            print(f"[ERROR] Failed to parse {csv_path}: {e}")
    
    if not contact_wiggle_data:
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
    
    contact_wiggle_path = output_dir / f"{args.title}_grouped_contact_wiggle_success.png"
    contact_wiggle_per_step_path = output_dir / f"{args.title}_grouped_contact_wiggle_per_step_success.png"
    contact_steps_path = output_dir / f"{args.title}_grouped_contact_steps_success.png"
    contact_push_speed_path = output_dir / f"{args.title}_grouped_contact_push_speed_success.png"
    mean_dz_per_step_path = output_dir / f"{args.title}_grouped_mean_dz_per_step_success.png"
    contact_pause_count_path = output_dir / f"{args.title}_grouped_contact_pause_count_success.png"
    
    print("\n" + "=" * 70)
    print("Generating grouped plots...")
    print("=" * 70)
    
    # Plot grouped contact wiggle
    plot_grouped_contact_wiggle(
        contact_wiggle_data,
        experiments,
        conditions,
        title=f"{args.title} Contact Wiggle Index (Median) - Success Only",
        output_path=str(contact_wiggle_path)
    )
    
    # Plot grouped contact wiggle per step
    plot_grouped_contact_wiggle_per_step(
        contact_wiggle_per_step_data,
        experiments,
        conditions,
        title=f"{args.title} Contact Wiggle Per Step (Median) - Success Only",
        output_path=str(contact_wiggle_per_step_path)
    )
    
    # Plot grouped contact steps
    plot_grouped_contact_steps(
        contact_steps_data,
        experiments,
        conditions,
        title=f"{args.title} Contact Steps (Median) - Success Only",
        output_path=str(contact_steps_path)
    )
    
    # Plot grouped contact push speed
    plot_grouped_contact_push_speed(
        contact_push_speed_data,
        experiments,
        conditions,
        title=f"{args.title} Contact Push Speed (Median) - Success Only",
        output_path=str(contact_push_speed_path)
    )
    
    # Plot grouped mean dZ per step
    plot_grouped_mean_dz_per_step(
        mean_dz_per_step_data,
        experiments,
        conditions,
        title=f"{args.title} Mean dZ Per Step (Median) - Success Only",
        output_path=str(mean_dz_per_step_path)
    )
    
    # Plot grouped contact pause count
    plot_grouped_contact_pause_count(
        contact_pause_count_data,
        experiments,
        conditions,
        title=f"{args.title} Contact Pause Count (Median) - Success Only",
        output_path=str(contact_pause_count_path)
    )


if __name__ == "__main__":
    main()
