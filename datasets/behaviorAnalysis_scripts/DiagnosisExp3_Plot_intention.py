#!/usr/bin/env python3
"""
Script to compute the "intention score" of a rollout dataset.

For each rollout, read `final_yaw_error` (radians) from a `*_alignment.csv` file.
    - If |final_yaw_error| < threshold (default 10 degrees) -> score 1
    - Otherwise                                             -> score 0

The total intention score = sum of per-rollout scores.
e.g. 20 out of 100 rollouts satisfy the condition -> intention score = 20.

Usage:
    python SmallExp7_Plot_intention.py --input alignment.csv
    python SmallExp7_Plot_intention.py --input alignment.csv --threshold-deg 10
    python SmallExp7_Plot_intention.py --input alignment.csv --output-dir ./plots
"""

import argparse
from pathlib import Path
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


def compute_intention_scores(
    alignment_df: pd.DataFrame,
    threshold_rad: float,
) -> pd.DataFrame:
    """Compute per-rollout intention score (1 if |final_yaw_error| < threshold else 0)."""
    if 'final_yaw_error' not in alignment_df.columns:
        raise KeyError(
            "Column 'final_yaw_error' not found in alignment CSV. "
            f"Available columns: {list(alignment_df.columns)}"
        )

    df = alignment_df[['final_yaw_error']].copy()
    df['abs_final_yaw_error'] = df['final_yaw_error'].abs()
    df['intention_score'] = (df['abs_final_yaw_error'] < threshold_rad).astype(int)
    return df


def plot_intention_summary(
    scores_df: pd.DataFrame,
    threshold_rad: float,
    title: str,
    output_path: str = None,
):
    """Plot a bar chart summarizing intention score counts (pass vs fail)."""
    n_total = len(scores_df)
    n_pass = int(scores_df['intention_score'].sum())
    n_fail = n_total - n_pass
    pass_rate = (n_pass / n_total * 100) if n_total > 0 else 0.0
    threshold_deg = np.degrees(threshold_rad)

    fig, ax = plt.subplots(figsize=(7, 5))

    labels = [f'Intention\n(|final_yaw_differences| < {threshold_deg:.1f}°)',
              f'No Intention\n(|final_yaw_differences| ≥ {threshold_deg:.1f}°)']
    counts = [n_pass, n_fail]
    colors = ['#4CAF50', '#F44336']

    bars = ax.bar(labels, counts, color=colors, edgecolor='black', width=0.55)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    ax.set_ylabel('Number of Rollouts', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title(
        f"{title}\nIntention Score: {n_pass} / {n_total}  ({pass_rate:.1f}%)",
        fontsize=13, fontweight='bold', pad=30,
    )
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compute intention score from a *_alignment.csv (per-rollout final_yaw_error)."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the alignment CSV file (must contain 'final_yaw_error' column)",
    )
    parser.add_argument(
        "--threshold-deg",
        type=float,
        default=10.0,
        help="Threshold in degrees. |final_yaw_error| < threshold -> score 1. Default: 10°",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory to save the plot (default: same as input file)",
    )
    parser.add_argument(
        "--title",
        default="SmallExp7-1",
        help="Plot title (default: SmallExp7-1)",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Also save a per-rollout intention CSV next to the plot.",
    )

    args = parser.parse_args()

    threshold_rad = np.radians(args.threshold_deg)

    print(f"Loading alignment data from: {args.input}")
    alignment_df = load_alignment_data(args.input)
    print(f"  Found {len(alignment_df)} episodes")

    print(
        f"\nThreshold: {args.threshold_deg:.2f}° ({threshold_rad:.4f} rad)"
        f"  -> |final_yaw_error| < threshold counts as intention (score=1)"
    )

    scores_df = compute_intention_scores(alignment_df, threshold_rad)

    n_total = len(scores_df)
    n_pass = int(scores_df['intention_score'].sum())
    pass_rate = (n_pass / n_total * 100) if n_total > 0 else 0.0

    print("\n" + "=" * 60)
    print("INTENTION SCORE SUMMARY")
    print("=" * 60)
    print(f"  Total rollouts:    {n_total}")
    print(f"  Intention (score=1): {n_pass}")
    print(f"  No intention (score=0): {n_total - n_pass}")
    print(f"  Intention Score:   {n_pass} / {n_total}  ({pass_rate:.1f}%)")
    print("=" * 60)

    print("\nPer-rollout breakdown:")
    print("-" * 60)
    for ep, row in scores_df.iterrows():
        deg = np.degrees(row['abs_final_yaw_error'])
        flag = "✓" if row['intention_score'] == 1 else "✗"
        print(f"  {flag} {ep}: |final_yaw_error| = {row['abs_final_yaw_error']:.4f} rad ({deg:.2f}°)"
              f"  score={int(row['intention_score'])}")
    print("-" * 60)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.input).parent

    input_stem = Path(args.input).stem.replace('_alignment', '')
    output_path = output_dir / f"{input_stem}_intention_score.png"

    title = args.title if args.title else f"Intention Score Analysis\n{input_stem}"

    plot_intention_summary(scores_df, threshold_rad, title, str(output_path))

    if args.save_csv:
        csv_out = output_dir / f"{input_stem}_intention_score.csv"
        scores_df.to_csv(csv_out)
        print(f"Per-rollout intention CSV saved to: {csv_out}")


if __name__ == "__main__":
    main()
