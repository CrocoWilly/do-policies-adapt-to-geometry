#!/usr/bin/env python3
"""
Script to analyze and plot multi-modal behavior patterns (+45° rotation vs -45° rotation).

Classification is based on `final_yaw_error` (radians), using its absolute value:
    - |final_yaw_error| < low_threshold  (default 0.35 rad ≈ 20°) -> Mode 1: Rotate +45°
    - |final_yaw_error| > high_threshold (default 1.22 rad ≈ 70°) -> Mode 2: Rotate -45°
    - otherwise                                                   -> Unclassified (excluded from plot)

Usage:
    python Plot_multi-modal_yaw45.py --input alignment.csv
    python Plot_multi-modal_yaw45.py --input alignment.csv --base-csv success_info.csv
    python Plot_multi-modal_yaw45.py --input alignment.csv --low-threshold 0.35 --high-threshold 1.22
    python Plot_multi-modal_yaw45.py --input alignment.csv --output-dir ./plots
"""

import argparse
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_alignment_data(csv_path: str) -> pd.DataFrame:
    """Load alignment CSV (with final_yaw_error column)."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(path, comment='#')

    if df.columns[0] == 'Unnamed: 0' or df.columns[0] == '':
        df = df.rename(columns={df.columns[0]: 'episode'})

    if 'episode' in df.columns:
        df = df.set_index('episode')

    return df


def load_success_data(csv_path: str) -> Dict[str, bool]:
    """Load base CSV and extract success/failure information."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(path, comment='#')

    success_dict = {}
    for _, row in df.iterrows():
        episode = row['episode']
        success = row['success']
        if isinstance(success, str):
            success = success.lower() == 'true'
        success_dict[episode] = success

    return success_dict


def find_base_csv(alignment_csv_path: str) -> str:
    """Try to find the corresponding base CSV file."""
    path = Path(alignment_csv_path)

    base_name = path.stem.replace('_alignment', '')
    base_csv = path.parent / f"{base_name}.csv"

    if base_csv.exists():
        return str(base_csv)

    raise FileNotFoundError(
        f"Could not find base CSV file. Tried: {base_csv}\n"
        "Please specify --base-csv manually."
    )


def classify_behavior_mode(
    final_yaw_error: float,
    low_threshold: float = 0.35,
    high_threshold: float = 1.22,
) -> Optional[str]:
    """
    Classify episode based on |final_yaw_error| (radians).

    Returns 'rotate_pos45', 'rotate_neg45', or None (ambiguous range).
    """
    err = abs(final_yaw_error)
    if err < low_threshold:
        return 'rotate_pos45'
    elif err > high_threshold:
        return 'rotate_neg45'
    else:
        return None


def analyze_multimodal_behavior(
    alignment_df: pd.DataFrame,
    success_dict: Dict[str, bool],
    low_threshold: float = 0.35,
    high_threshold: float = 1.22,
) -> Dict[str, Dict[str, int]]:
    """Analyze multi-modal behavior patterns using final_yaw_error."""
    if 'final_yaw_error' not in alignment_df.columns:
        raise KeyError(
            "Column 'final_yaw_error' not found in alignment CSV. "
            f"Available columns: {list(alignment_df.columns)}"
        )

    results = {
        'rotate_pos45': {'success': 0, 'failure': 0},
        'rotate_neg45': {'success': 0, 'failure': 0},
        'unclassified': {'success': 0, 'failure': 0},
    }

    for episode in alignment_df.index:
        if episode not in success_dict:
            print(f"Warning: {episode} not found in success data, skipping")
            continue

        final_yaw_error = alignment_df.loc[episode, 'final_yaw_error']
        success = success_dict[episode]
        mode = classify_behavior_mode(final_yaw_error, low_threshold, high_threshold)

        key = mode if mode is not None else 'unclassified'
        if success:
            results[key]['success'] += 1
        else:
            results[key]['failure'] += 1

    return results


def plot_multimodal_analysis(
    results: Dict[str, Dict[str, int]],
    title: str = "Multi-Modal Behavior Analysis (±45°)",
    output_path: str = None,
    low_threshold: float = 0.35,
    high_threshold: float = 1.22,
):
    """Plot bar chart showing success/failure counts for each behavior mode."""
    fig, ax = plt.subplots(figsize=(8, 5))

    modes = ['rotate_pos45', 'rotate_neg45']
    mode_labels = [
        'Rotate +45°',
        'Rotate -45°',
    ]

    success_counts = [results[mode]['success'] for mode in modes]
    failure_counts = [results[mode]['failure'] for mode in modes]
    total_counts = [s + f for s, f in zip(success_counts, failure_counts)]

    x = np.arange(len(modes))
    width = 0.35

    bars_success = ax.bar(x - width/2, success_counts, width, label='Success', color='#4CAF50', edgecolor='black')
    bars_failure = ax.bar(x + width/2, failure_counts, width, label='Failure', color='#F44336', edgecolor='black')

    ax.set_xlabel('Behavior Mode', fontsize=12)
    ax.set_ylabel('Number of Episodes', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=11, fontweight='bold')

    add_labels(bars_success)
    add_labels(bars_failure)

    for i, mode in enumerate(modes):
        total = total_counts[i]
        if total > 0:
            success_rate = success_counts[i] / total * 100
            ax.text(x[i], -0.12,
                   f'Total: {total}, Success Rate: {success_rate:.1f}%',
                   ha='center', va='top', fontsize=9,
                   transform=ax.get_xaxis_transform())

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and plot multi-modal behavior patterns (+45° vs -45°) using final_yaw_error"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the alignment CSV file"
    )
    parser.add_argument(
        "--base-csv", "-b",
        default=None,
        help="Path to the base CSV file with success info (auto-detected if not specified)"
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.35,
        help="Lower threshold (rad). |final_yaw_error| < this -> Mode 1 (+45°). Default: 0.35 rad (~20°)"
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=1.22,
        help="Upper threshold (rad). |final_yaw_error| > this -> Mode 2 (-45°). Default: 1.22 rad (~70°)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory to save the plot (default: same as input file)"
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom plot title"
    )

    args = parser.parse_args()

    print(f"Loading alignment data from: {args.input}")
    alignment_df = load_alignment_data(args.input)
    print(f"  Found {len(alignment_df)} episodes")

    if args.base_csv:
        base_csv_path = args.base_csv
    else:
        base_csv_path = find_base_csv(args.input)

    print(f"Loading success data from: {base_csv_path}")
    success_dict = load_success_data(base_csv_path)
    print(f"  Found {len(success_dict)} episodes")

    print(
        f"\nClassification thresholds (based on |final_yaw_error|):"
        f"\n  < {args.low_threshold:.2f} rad ({np.degrees(args.low_threshold):.1f}°) -> Mode 1: Rotate +45°"
        f"\n  > {args.high_threshold:.2f} rad ({np.degrees(args.high_threshold):.1f}°) -> Mode 2: Rotate -45°"
    )
    results = analyze_multimodal_behavior(
        alignment_df, success_dict, args.low_threshold, args.high_threshold
    )

    print("\n" + "=" * 60)
    print("MULTI-MODAL BEHAVIOR ANALYSIS SUMMARY (±45°)")
    print("=" * 60)

    for mode, mode_label in [
        ('rotate_pos45', 'Rotate +45° (Mode 1)'),
        ('rotate_neg45', 'Rotate -45° (Mode 2)'),
        ('unclassified', 'Unclassified'),
    ]:
        success = results[mode]['success']
        failure = results[mode]['failure']
        total = success + failure
        rate = success / total * 100 if total > 0 else 0
        print(f"\n[{mode_label}]")
        print(f"  Success: {success}")
        print(f"  Failure: {failure}")
        print(f"  Total: {total}")
        print(f"  Success Rate: {rate:.1f}%")

    print("\n" + "=" * 60)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.input).parent

    input_stem = Path(args.input).stem.replace('_alignment', '')
    output_path = output_dir / f"{input_stem}_multimodal_yaw45.png"

    if args.title:
        title = args.title
    else:
        title = f"Multi-Modal Behavior Analysis (±45°)\n{input_stem}"

    plot_multimodal_analysis(
        results, title, str(output_path), args.low_threshold, args.high_threshold
    )


if __name__ == "__main__":
    main()
