"""
Script to plot grouped alignment phase behavior analysis metrics by
(color, texture, bg) combination, with one subplot per clearance value.

Each bar in a subplot represents one combination (e.g. red_metal_dark).
The combination is parsed directly from the input CSV filename, e.g.:
    MainExp2-3_rollout100_clearance112_yaw-15_red_metal_dark_epoch20_alignment.csv
                              clearance ^^^      color texture bg

Metrics produced:
    1. Final Yaw Error (deg)
    2. Alignment Steps
    3. Alignment Wiggle Index (cm)
    4. Yaw Correction (deg)

Usage:
    python MainExp_Plot_Diagnosis_alignment_byColorTextureBg_mean.py \
        --input ./bc_results/mainExp2/analyzed_csv/*_alignment.csv \
        --title "MainExp2-3" --output-dir ./bc_results/mainExp2/
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Color palettes per object color (used to tint bars within a clearance subplot)
COLOR_BASE = {
    "blue":  "#4477CC",
    "red":   "#DD3344",
    "green": "#88CC44",
}
DEFAULT_COLOR = "#888888"

TEXTURE_HATCH = {
    "metal": "",
    "wood":  "//",
}

BG_ALPHA = {
    "light": 1.0,
    "dark":  0.55,
}


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def parse_csv_means(csv_path: Path) -> Dict[str, float]:
    """Read a per-demo alignment CSV and return means for each metric."""
    finals, steps, wiggle, yawcorr = [], [], [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                finals.append(float(row["final_yaw_error"]))
                steps.append(float(row["alignment_steps"]))
                wiggle.append(float(row["alignment_wiggle"]))
                yawcorr.append(float(row["yaw_correction"]))
            except (TypeError, ValueError, KeyError):
                # Skip appended terminal-log lines or malformed rows
                continue

    return {
        "final_yaw_error":  mean(finals),
        "alignment_steps":  mean(steps),
        "alignment_wiggle": mean(wiggle),
        "yaw_correction":   mean(yawcorr),
    }


def parse_filename(filename: str) -> Tuple[int, str, str, str, str]:
    """
    Extract (clearance, yaw, color, texture, bg) from a filename like:
        MainExp2-3_rollout100_clearance112_yaw-15_red_metal_dark_epoch20_alignment
    """
    name = re.sub(r"_alignment$", "", filename)
    name = re.sub(r"_epoch\d+$", "", name)

    clearance_match = re.search(r"clearance(\d+)", name)
    clearance = int(clearance_match.group(1)) if clearance_match else 0

    yaw_match = re.search(r"yaw(-?\d+)", name)
    yaw = yaw_match.group(1) if yaw_match else "?"

    combo_match = re.search(
        r"yaw-?\d+_([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)", name
    )
    if combo_match:
        color, texture, bg = combo_match.groups()
    else:
        color, texture, bg = "unknown", "unknown", "unknown"

    return clearance, yaw, color, texture, bg


def combo_label(color: str, texture: str, bg: str) -> str:
    return f"{color}\n{texture}\n{bg}"


def combo_sort_key(combo: Tuple[str, str, str]) -> Tuple[int, int, int]:
    color_order = {"blue": 0, "red": 1, "green": 2}
    texture_order = {"metal": 0, "wood": 1}
    bg_order = {"light": 0, "dark": 1}
    color, texture, bg = combo
    return (
        color_order.get(color, 99),
        texture_order.get(texture, 99),
        bg_order.get(bg, 99),
    )


def bar_style_for_combo(color: str, texture: str, bg: str) -> Tuple[str, float, str]:
    base = COLOR_BASE.get(color, DEFAULT_COLOR)
    alpha = BG_ALPHA.get(bg, 1.0)
    hatch = TEXTURE_HATCH.get(texture, "")
    return base, alpha, hatch


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metric_by_clearance(
    grouped: Dict[int, Dict[Tuple[str, str, str], float]],
    ylabel: str,
    title: str,
    output_path: Path,
    value_transform=lambda v: v,
    annotation_fmt: str = "{:.2f}",
    ylim: Tuple[float, float] = None,
):
    clearances = sorted(grouped.keys())
    n_sub = len(clearances)
    if n_sub == 0:
        print(f"[WARN] No data to plot for {title}.")
        return

    # Use union of combinations across clearances so x-axis is comparable.
    all_combos = sorted(
        {combo for cmap in grouped.values() for combo in cmap.keys()},
        key=combo_sort_key,
    )

    fig, axes = plt.subplots(
        1, n_sub,
        figsize=(max(8, 1.4 * len(all_combos) * n_sub + 3), 8),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for ax, clearance in zip(axes, clearances):
        cmap = grouped[clearance]
        x = np.arange(len(all_combos))
        values = [value_transform(cmap.get(combo, 0.0)) for combo in all_combos]

        for i, combo in enumerate(all_combos):
            color, texture, bg = combo
            face, alpha, hatch = bar_style_for_combo(color, texture, bg)
            present = combo in cmap
            ax.bar(
                x[i], values[i],
                width=0.8,
                color=face,
                alpha=alpha if present else 0.15,
                edgecolor="black",
                hatch=hatch,
            )
            if present:
                ax.annotate(
                    annotation_fmt.format(values[i]),
                    xy=(x[i], values[i]),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=18,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [combo_label(*c) for c in all_combos],
            rotation=0, ha="center", fontsize=20,
        )
        ax.tick_params(axis="y", labelsize=20)
        # ax.set_title(f"Hole size {clearance/100:.2f}", fontsize=18)
        ax.set_title(f"Tight Hole ({clearance/100:.2f}x peg size)", fontsize=22, pad=15)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if ylim is not None:
            ax.set_ylim(*ylim)

    axes[0].set_ylabel(ylabel, fontsize=22)
    fig.suptitle(title, fontsize=24, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(top=0.88)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot grouped alignment metrics: one subplot per clearance, "
            "one bar per (color, texture, bg) combination."
        )
    )
    parser.add_argument(
        "--input", "-i",
        type=str, nargs="+", required=True,
        help="Input alignment CSV files (supports shell glob expansion).",
    )
    parser.add_argument("--title", type=str, default="MainExp2",
                        help="Title prefix for plots.")
    parser.add_argument("--output-dir", type=str, default="./",
                        help="Directory to save output plots.")

    args = parser.parse_args()

    # grouped[metric][clearance][(color, texture, bg)] = mean value
    grouped: Dict[str, Dict[int, Dict[Tuple[str, str, str], float]]] = {
        "final_yaw_error":  defaultdict(dict),
        "alignment_steps":  defaultdict(dict),
        "alignment_wiggle": defaultdict(dict),
        "yaw_correction":   defaultdict(dict),
    }

    for csv_path_str in args.input:
        csv_path = Path(csv_path_str)
        if not csv_path.exists():
            print(f"[WARN] Not found: {csv_path}")
            continue
        try:
            means = parse_csv_means(csv_path)
            clearance, yaw, color, texture, bg = parse_filename(csv_path.stem)
            combo = (color, texture, bg)
            for metric, value in means.items():
                grouped[metric][clearance][combo] = value
            print(
                f"Parsed: {csv_path.name} -> clearance={clearance} "
                f"yaw={yaw} combo={color}_{texture}_{bg}"
            )
        except Exception as e:
            print(f"[ERROR] Failed to parse {csv_path}: {e}")

    if not any(grouped[m] for m in grouped):
        print("No valid CSV files to plot.")
        return

    output_dir = Path(args.output_dir)

    print("\n" + "=" * 70)
    print("Generating grouped plots...")
    print("=" * 70)

    plot_metric_by_clearance(
        grouped["final_yaw_error"],
        ylabel="Final yaw differences (deg)",
        title=f"{args.title} final yaw differences under different appearance variations (Mean)",
        output_path=output_dir / f"{args.title}_grouped_final_yaw_error.png",
        value_transform=lambda v: float(np.degrees(v)),
        annotation_fmt="{:.2f}",
        ylim=(0.0, 5.0),
    )
    plot_metric_by_clearance(
        grouped["alignment_steps"],
        ylabel="Alignment steps",
        title=f"{args.title} alignment steps under different appearance variations (Mean)",
        output_path=output_dir / f"{args.title}_grouped_alignment_steps.png",
        value_transform=lambda v: v,
        annotation_fmt="{:.0f}",
        ylim=(0.0, 150.0),
    )
    plot_metric_by_clearance(
        grouped["alignment_wiggle"],
        ylabel="Alignment wiggle index (cm)",
        title=f"{args.title} alignment wiggle index under different appearance variations (Mean)",
        output_path=output_dir / f"{args.title}_grouped_alignment_wiggle.png",
        value_transform=lambda v: v * 100.0,
        annotation_fmt="{:.3f}",
    )
    plot_metric_by_clearance(
        grouped["yaw_correction"],
        ylabel="Yaw correction (deg)",
        title=f"{args.title} yaw correction under different appearance variations (Mean)",
        output_path=output_dir / f"{args.title}_grouped_yaw_correction.png",
        value_transform=lambda v: float(np.degrees(v)),
        annotation_fmt="{:.1f}",
    )


if __name__ == "__main__":
    main()
