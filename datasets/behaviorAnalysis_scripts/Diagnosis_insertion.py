#!/usr/bin/env python3
"""
Script to run insertion phase behavior analysis metrics on episodes in an HDF5 dataset.

This script runs insertion/contact-related behavior analysis scripts and aggregates results:
    - contact_wiggle: Wiggle during contact/push phase
    - contact_push_speed: Push speed during contact phase
    - contact_pause_count: Number of pauses during contact phase

Usage:
    python Diagnosis2_behaviorAnalysis_insertion.py --input rollout_dataset.hdf5
    python Diagnosis2_behaviorAnalysis_insertion.py --input rollout_dataset.hdf5 --output results.csv
    python Diagnosis2_behaviorAnalysis_insertion.py --input rollout_dataset.hdf5 --quiet
"""

import subprocess
import argparse
import os
import sys
import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import h5py
import numpy as np
import pandas as pd


# Insertion/contact phase analysis scripts
ANALYSIS_SCRIPTS = {
    "contact_wiggle": "BehaviorAnalysis_CalcContactWiggle.py",
    "contact_push_speed": "BehaviorAnalysis_CalcContactPushSpeed.py",
    "contact_pause_count": "BehaviorAnalysis_CalcContactPauseCount.py",
}


def run_and_capture(script_name: str, input_file: str, verbose: bool = True) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, int], str]:
    """
    Run a script and capture its stdout.
    
    Args:
        script_name: Name of the script to run
        input_file: Path to input HDF5 file
        verbose: Print detailed output
    
    Returns:
        Tuple of (parsed value results dict, per_step dict, mean_dz_per_step dict, contact_steps dict, raw output string)
    """
    header = f"\nRunning: {script_name}\n" + "-" * 50 + "\n"
    if verbose:
        print(header, end="")

    cmd = [
        "python",
        script_name,
        "--input",
        input_file,
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        error_msg = f"[ERROR] {script_name} failed: {e.stderr}"
        print(error_msg)
        return {}, {}, {}, {}, header + error_msg

    output = result.stdout
    if verbose:
        print(output)

    value_results, per_step_results, mean_dz_results, contact_steps_results = parse_output(output)
    return value_results, per_step_results, mean_dz_results, contact_steps_results, header + output


def parse_output(text: str) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, int]]:
    """
    Parse episode-value pairs and additional metrics from stdout.
    
    Accepts formats like:
        demo_0: 0.123
        demo_0 = 0.123
        demo_0: metric_name = 0.123 rad (0.03 deg)
        demo_0: contact_wiggle_index = 0.025758, per_step = 0.000157 (contact_start=36, contact_steps=164, ...)
        demo_0: contact_push_speed = 0.000017, mean_dz_per_step = 0.000040 (z_disp=0.0028, ...)
    
    Args:
        text: Output text from script
    
    Returns:
        Tuple of (value_results dict, per_step_results dict, mean_dz_per_step_results dict, contact_steps_results dict)
    """
    value_results = {}
    per_step_results = {}
    mean_dz_results = {}
    contact_steps_results = {}
    
    # Pattern to match: demo_X: metric_name = value or demo_X: value
    pattern = re.compile(r"(demo_\d+)\s*:\s*(?:\w+\s*=\s*)?([-+]?\d*\.\d+|\d+)")
    # Pattern to match per_step = value (but NOT mean_dz_per_step)
    # Use word boundary or comma before to ensure it's the standalone per_step
    per_step_pattern = re.compile(r",\s*per_step\s*=\s*([-+]?\d*\.\d+|\d+)")
    # Pattern to match mean_dz_per_step = value
    mean_dz_pattern = re.compile(r"mean_dz_per_step\s*=\s*([-+]?\d*\.\d+|\d+)")
    # Pattern to match contact_steps = value
    contact_steps_pattern = re.compile(r"contact_steps\s*=\s*(\d+)")

    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            episode = match.group(1)
            value = float(match.group(2))
            value_results[episode] = value
            
            # Try to extract per_step from the same line
            per_step_match = per_step_pattern.search(line)
            if per_step_match:
                per_step_results[episode] = float(per_step_match.group(1))
            
            # Try to extract mean_dz_per_step from the same line
            mean_dz_match = mean_dz_pattern.search(line)
            if mean_dz_match:
                mean_dz_results[episode] = float(mean_dz_match.group(1))
            
            # Try to extract contact_steps from the same line
            contact_steps_match = contact_steps_pattern.search(line)
            if contact_steps_match:
                contact_steps_results[episode] = int(contact_steps_match.group(1))

    return value_results, per_step_results, mean_dz_results, contact_steps_results


def format_metric_value(metric_name: str, value: float) -> str:
    """
    Format metric value with appropriate units.
    
    Args:
        metric_name: Name of the metric
        value: The metric value
    
    Returns:
        Formatted string with units
    """
    if "speed" in metric_name.lower():
        # Speed metrics: show m/s
        return f"{value:.6f} m/s"
    elif "count" in metric_name.lower() or "contact_steps" in metric_name.lower():
        # Count metrics and contact_steps: show as integer
        return f"{int(value)}"
    elif "per_step" in metric_name.lower() or "dz_per_step" in metric_name.lower():
        # Per-step metrics: show with more precision
        return f"{value:.6f}"
    else:
        # Other metrics: just show value
        return f"{value:.6f}"


def get_success_episodes(hdf5_path: str, demo_prefix: str = "demo_") -> set:
    """
    Get set of successful episode names from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        demo_prefix: Prefix for demo episode names
    
    Returns:
        Set of successful episode names
    """
    success_episodes = set()
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            return success_episodes
        data_grp = f["data"]
        for demo_name in data_grp.keys():
            if demo_name.startswith(demo_prefix):
                demo = data_grp[demo_name]
                success = demo.attrs.get("success", False)
                if isinstance(success, np.bool_):
                    success = bool(success)
                if success:
                    success_episodes.add(demo_name)
    return success_episodes


def calculate_summary_statistics(df: pd.DataFrame, success_episodes: set = None) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Calculate summary statistics for each metric separately for successful and failed episodes.
    
    Args:
        df: DataFrame with episodes as rows and metrics as columns
        success_episodes: Set of successful episode names (if None, use all as success)
    
    Returns:
        Tuple of (success_summary, failed_summary) dictionaries with statistics for each metric
    """
    # Split into success and failed dataframes
    if success_episodes is not None:
        df_success = df[df.index.isin(success_episodes)]
        df_failed = df[~df.index.isin(success_episodes)]
    else:
        df_success = df
        df_failed = pd.DataFrame()
    
    def compute_stats(df_subset):
        summary = {}
        for col in df.columns:  # Use original df columns to ensure all metrics present
            if col in df_subset.columns:
                values = df_subset[col].dropna()
            else:
                values = pd.Series([], dtype=float)
            if len(values) > 0:
                summary[col] = {
                    "median": float(np.median(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }
            else:
                summary[col] = {
                    "median": 0.0,
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "count": 0,
                }
        return summary
    
    success_summary = compute_stats(df_success)
    failed_summary = compute_stats(df_failed)
    
    return success_summary, failed_summary


def process_hdf5_file(
    input_hdf5: str,
    output_csv: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Process an HDF5 file and run insertion behavior analysis scripts.
    
    Args:
        input_hdf5: Path to input HDF5 file
        output_csv: Optional path to save results as CSV
        verbose: Print detailed output
    
    Returns:
        Dictionary with 'episodes' (DataFrame) and 'summary' (statistics)
    """
    input_path = Path(input_hdf5)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Collect all terminal output
    all_output_logs: List[str] = []

    header = f"Analyzing: {input_path.name}\n" + "=" * 70 + "\n"
    if verbose:
        print(header, end="")
    all_output_logs.append(header)

    # Get script directory (assume scripts are in same directory as this file)
    script_dir = Path(__file__).parent

    all_results = {}

    for metric_name, script in ANALYSIS_SCRIPTS.items():
        script_path = script_dir / script

        if not script_path.exists():
            warning = f"[WARNING] Script not found: {script}\n"
            print(warning, end="")
            all_output_logs.append(warning)
            continue

        metric_results, per_step_results, mean_dz_results, contact_steps_results, raw_output = run_and_capture(str(script_path), input_hdf5, verbose)
        all_output_logs.append(raw_output)

        for ep, val in metric_results.items():
            if ep not in all_results:
                all_results[ep] = {}
            all_results[ep][metric_name] = val
            
            # Collect per_step metric (from contact_wiggle script)
            if ep in per_step_results:
                all_results[ep]["contact_wiggle_per_step"] = per_step_results[ep]
            
            # Collect mean_dz_per_step metric (from contact_push_speed script)
            if ep in mean_dz_results:
                all_results[ep]["mean_dz_per_step"] = mean_dz_results[ep]
            
            # Collect contact_steps metric (from any script that outputs it)
            if ep in contact_steps_results and "contact_steps" not in all_results[ep]:
                all_results[ep]["contact_steps"] = contact_steps_results[ep]

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(all_results, orient="index")
    df = df.sort_index()

    # Get successful episodes from HDF5
    success_episodes = get_success_episodes(input_hdf5)
    num_success = len(success_episodes & set(df.index))
    num_failed = len(df) - num_success

    # Calculate summary statistics (separately for success and failed)
    success_summary, failed_summary = calculate_summary_statistics(df, success_episodes)

    # Generate summary output
    summary_output = "\n" + "=" * 70 + "\n"
    summary_output += "SUMMARY (Insertion Phase)\n"
    summary_output += "=" * 70 + "\n"
    summary_output += f"Total Episodes: {len(df)}\n"
    summary_output += f"Successful Episodes: {num_success}\n"
    summary_output += f"Failed Episodes: {num_failed}\n"
    
    # Success statistics
    summary_output += "\n" + "-" * 70 + "\n"
    summary_output += "[SUCCESS EPISODES]\n"
    summary_output += "-" * 70 + "\n"
    if num_success > 0:
        for metric, stats in success_summary.items():
            summary_output += f"Median {metric}: {format_metric_value(metric, stats['median'])}\n"
            summary_output += f"Min {metric}: {format_metric_value(metric, stats['min'])}\n"
            summary_output += f"Max {metric}: {format_metric_value(metric, stats['max'])}\n"
            summary_output += "\n"
    else:
        summary_output += "No successful episodes.\n\n"
    
    # Failed statistics
    summary_output += "-" * 70 + "\n"
    summary_output += "[FAILED EPISODES]\n"
    summary_output += "-" * 70 + "\n"
    if num_failed > 0:
        for metric, stats in failed_summary.items():
            summary_output += f"Median {metric}: {format_metric_value(metric, stats['median'])}\n"
            summary_output += f"Min {metric}: {format_metric_value(metric, stats['min'])}\n"
            summary_output += f"Max {metric}: {format_metric_value(metric, stats['max'])}\n"
            summary_output += "\n"
    else:
        summary_output += "No failed episodes.\n\n"

    if verbose:
        print(summary_output, end="")
    all_output_logs.append(summary_output)

    # Save results to CSV
    if output_csv:
        output_path = Path(output_csv)
        
        # Write per-episode results
        df.to_csv(output_path)
        
        # Append terminal output (which includes SUMMARY)
        with open(output_path, "a") as f:
            f.write("\n# ========== Terminal Output Log ==========\n")
            for log in all_output_logs:
                for line in log.splitlines():
                    f.write(f"# {line}\n")

        save_msg = f"\nResults saved to: {output_path}\n"
        if verbose:
            print(save_msg, end="")

    return {
        "episodes": df,
        "success_summary": success_summary,
        "failed_summary": failed_summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run insertion phase behavior analysis metrics on HDF5 rollout dataset"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input HDF5 file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Optional: Path to save results as CSV file"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress detailed output"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    # Without --output, reuse the input file name with an _insertion.csv extension
    output_csv = args.output
    if output_csv is None:
        input_path = Path(args.input)
        output_csv = str(input_path.with_suffix('')) + "_insertion.csv"

    results = process_hdf5_file(
        input_hdf5=args.input,
        output_csv=output_csv,
        verbose=not args.quiet,
    )

    return results


if __name__ == "__main__":
    main()
