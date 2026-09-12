"""
Script to plot alignment phase histogram for success and failed episodes.

This script reads alignment CSV files and corresponding HDF5 files to:
    1. Plot alignment_steps histogram for success episodes
    2. Plot yaw_correction histogram for success episodes
    3. Plot alignment_steps histogram for failed episodes
    4. Plot yaw_correction histogram for failed episodes

Usage:
    python SmallExp7_Plot_Diagnosis2_alignment.py --input SmallExp7-2_rollout100_clearance112_yaw0_case1_alignment.csv
    python SmallExp7_Plot_Diagnosis2_alignment.py --input file1.csv file2.csv
    python SmallExp7_Plot_Diagnosis2_alignment.py --input file1.csv --output-dir ./plots
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py


def get_hdf5_path_from_csv(csv_path: str) -> Path:
    """
    Derive the HDF5 file path from CSV file path.
    
    Args:
        csv_path: Path to the alignment CSV file
    
    Returns:
        Path to the corresponding HDF5 file
    """
    csv_path = Path(csv_path)
    # Remove '_alignment' suffix from filename
    hdf5_name = csv_path.stem.replace("_alignment", "") + ".hdf5"
    hdf5_path = csv_path.parent / hdf5_name
    return hdf5_path


def get_success_episodes(hdf5_path: str, demo_prefix: str = "demo_") -> Set[str]:
    """
    Get set of successful episode names from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        demo_prefix: Prefix for demo episode names
    
    Returns:
        Set of successful episode names
    """
    success_episodes = set()
    hdf5_path = Path(hdf5_path)
    
    if not hdf5_path.exists():
        print(f"[WARNING] HDF5 file not found: {hdf5_path}")
        return success_episodes
    
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


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """
    Load alignment CSV data.
    
    Args:
        csv_path: Path to the alignment CSV file
    
    Returns:
        DataFrame with alignment data
    """
    # Read CSV, skipping comment lines
    df = pd.read_csv(csv_path, comment='#', index_col=0)
    return df


def extract_label_from_filename(filename: str) -> str:
    """
    Extract a shorter label from the filename for display.
    
    Args:
        filename: The filename (without extension)
    
    Returns:
        A shorter label string
    """
    # Remove _alignment suffix if present
    filename = re.sub(r"_alignment$", "", filename)
    
    # Try to extract SmallExp, clearance, yaw, and case info
    # Example: SmallExp7-2_rollout100_clearance112_yaw0_case1 -> SE7-2 C112_Y0_case1
    match = re.search(r"SmallExp(\d+)-(\d+).*clearance(\d+)_yaw(-?\d+)_?(case\d+)?", filename)
    if match:
        exp = match.group(1)
        run = match.group(2)
        clearance = match.group(3)
        yaw = match.group(4)
        case = match.group(5) if match.group(5) else ""
        label = f"SE{exp}-{run}_C{clearance}_Y{yaw}"
        if case:
            label += f"_{case}"
        return label
    
    # Try to extract clearance and yaw info only
    match = re.search(r"clearance(\d+)_yaw(-?\d+)", filename)
    if match:
        clearance = match.group(1)
        yaw = match.group(2)
        return f"C{clearance}_Y{yaw}"
    
    # Fallback: return last part of filename
    parts = filename.split("_")
    if len(parts) >= 2:
        return "_".join(parts[-2:])
    return filename


def plot_histograms(
    df: pd.DataFrame,
    success_episodes: Set[str],
    title_prefix: str = "",
    output_dir: str = None,
    filename_stem: str = "alignment",
):
    """
    Plot histograms for alignment_steps and yaw_correction for success and failed episodes.
    
    Args:
        df: DataFrame with alignment data
        success_episodes: Set of successful episode names
        title_prefix: Prefix for the plot title
        output_dir: Directory to save the figures
        filename_stem: Stem for the output filenames
    """
    # Split data into success and failed
    success_mask = df.index.isin(success_episodes)
    df_success = df[success_mask]
    df_failed = df[~success_mask]
    
    n_success = len(df_success)
    n_failed = len(df_failed)
    
    print(f"  Success episodes: {n_success}")
    print(f"  Failed episodes: {n_failed}")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Colors
    success_color = '#2ecc71'  # Green
    failed_color = '#e74c3c'   # Red
    
    # Compute shared bins for alignment_steps (fixed range: 0 to 130)
    bins_steps = np.arange(-0.5, 130 + 1.5, 1)  # Max bin value = 130
    if len(bins_steps) > 50:  # Too many bins, use fixed number
        bins_steps = np.linspace(-0.5, 130 + 0.5, 31)
    
    # Compute shared bins for yaw_correction (fixed range: 0 to 110 degrees)
    bins_yaw = np.linspace(0, 110, 31)  # Min = 0, Max = 110 degrees
    
    # --- Success: alignment_steps ---
    ax = axes[0, 0]
    if n_success > 0:
        steps_success = df_success['alignment_steps'].values
        ax.hist(steps_success, bins=bins_steps, color=success_color, edgecolor='black', alpha=0.8)
        ax.axvline(np.median(steps_success), color='darkgreen', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(steps_success):.1f}')
        ax.legend()
        ax.set_xlabel('Alignment Steps', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Success Episodes (n={n_success})\nAlignment Steps', fontsize=12)
    else:
        ax.text(0.5, 0.5, 'No success episodes', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'Success Episodes (n=0)\nAlignment Steps', fontsize=12)
    ax.set_xlim(bins_steps[0], bins_steps[-1])
    
    # --- Success: yaw_correction ---
    ax = axes[0, 1]
    if n_success > 0:
        yaw_success = np.degrees(df_success['yaw_correction'].values)  # Convert to degrees
        ax.hist(yaw_success, bins=bins_yaw, color=success_color, edgecolor='black', alpha=0.8)
        ax.axvline(np.median(yaw_success), color='darkgreen', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(yaw_success):.2f}°')
        ax.legend()
        ax.set_xlabel('Yaw Correction (deg)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Success Episodes (n={n_success})\nYaw Correction', fontsize=12)
    else:
        ax.text(0.5, 0.5, 'No success episodes', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'Success Episodes (n=0)\nYaw Correction', fontsize=12)
    ax.set_xlim(bins_yaw[0], bins_yaw[-1])
    
    # --- Failed: alignment_steps ---
    ax = axes[1, 0]
    if n_failed > 0:
        steps_failed = df_failed['alignment_steps'].values
        ax.hist(steps_failed, bins=bins_steps, color=failed_color, edgecolor='black', alpha=0.8)
        ax.axvline(np.median(steps_failed), color='darkred', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(steps_failed):.1f}')
        ax.legend()
        ax.set_xlabel('Alignment Steps', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Failed Episodes (n={n_failed})\nAlignment Steps', fontsize=12)
    else:
        ax.text(0.5, 0.5, 'No failed episodes', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'Failed Episodes (n=0)\nAlignment Steps', fontsize=12)
    ax.set_xlim(bins_steps[0], bins_steps[-1])
    
    # --- Failed: yaw_correction ---
    ax = axes[1, 1]
    if n_failed > 0:
        yaw_failed = np.degrees(df_failed['yaw_correction'].values)  # Convert to degrees
        ax.hist(yaw_failed, bins=bins_yaw, color=failed_color, edgecolor='black', alpha=0.8)
        ax.axvline(np.median(yaw_failed), color='darkred', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(yaw_failed):.2f}°')
        ax.legend()
        ax.set_xlabel('Yaw Correction (deg)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Failed Episodes (n={n_failed})\nYaw Correction', fontsize=12)
    else:
        ax.text(0.5, 0.5, 'No failed episodes', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'Failed Episodes (n=0)\nYaw Correction', fontsize=12)
    ax.set_xlim(bins_yaw[0], bins_yaw[-1])
    
    # Main title
    if title_prefix:
        fig.suptitle(f'{title_prefix}\nAlignment Phase Histogram Analysis', fontsize=14, fontweight='bold')
    else:
        fig.suptitle('Alignment Phase Histogram Analysis', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / f"{filename_stem}_histogram.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to: {output_path}")
    
    plt.show()


def process_single_file(csv_path: str, output_dir: str = None, title: str = None):
    """
    Process a single CSV file and generate histograms.
    
    Args:
        csv_path: Path to the alignment CSV file
        output_dir: Directory to save the figures
        title: Custom title for the plot
    """
    csv_path = Path(csv_path)
    print(f"\nProcessing: {csv_path.name}")
    
    # Load CSV data
    df = load_csv_data(csv_path)
    print(f"  Loaded {len(df)} episodes from CSV")
    
    # Get corresponding HDF5 path and success episodes
    hdf5_path = get_hdf5_path_from_csv(csv_path)
    success_episodes = get_success_episodes(hdf5_path)
    print(f"  Found {len(success_episodes)} success episodes from HDF5")
    
    # Generate label/title
    if title:
        title_prefix = title
    else:
        title_prefix = extract_label_from_filename(csv_path.stem)
    
    # Determine output filename stem
    filename_stem = csv_path.stem.replace("_alignment", "")
    
    # Plot histograms
    plot_histograms(
        df=df,
        success_episodes=success_episodes,
        title_prefix=title_prefix,
        output_dir=output_dir,
        filename_stem=filename_stem,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot alignment histograms for success and failed episodes."
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        help="Path(s) to alignment CSV file(s)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./",
        help="Directory to save output plots. Default: current directory (./)."
    )
    parser.add_argument(
        "--title", "-t",
        default=None,
        help="Custom title for the plot (only effective for single file input)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if specified
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Process each file
    for csv_path in args.input:
        process_single_file(
            csv_path=csv_path,
            output_dir=output_dir,
            title=args.title if len(args.input) == 1 else None,
        )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
