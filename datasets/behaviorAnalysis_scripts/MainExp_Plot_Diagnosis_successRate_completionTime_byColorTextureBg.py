"""
Script to plot success rate and median completion time grouped by
(color, texture, bg) combination, with one subplot per clearance value.

Each bar in a subplot represents one combination (e.g. red_metal_dark).
The combination is parsed directly from the input CSV filename, e.g.:
    MainExp2-3_rollout100_clearance112_yaw-15_red_metal_dark_epoch20.csv
                              clearance ^^^      color texture bg

Metrics produced:
    1. Success Rate (%)
    2. Median Completion Steps (with std error bars)

Usage:
    python MainExp_Plot_Diagnosis_successRate_completionTime_byColorTextureBg.py \
        --input ./bc_results/mainExp2/analyzed_csv/*epoch20.csv \
        --title "MainExp2-3" --output-dir ./bc_results/mainExp2/
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

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

# Default success rate (%) per (color, texture, bg) combo, used when a CSV
# for that combination is not provided in --input.
DEFAULT_SUCCESS_RATE: Dict[Tuple[str, str, str], float] = {
    ("blue",  "metal", "dark"):  97.00,
    ("blue",  "metal", "light"): 90.00,
    ("blue",  "wood",  "dark"):  53.33,
    ("blue",  "wood",  "light"): 99.67,
    ("green", "metal", "dark"):  95.67,
    ("green", "metal", "light"): 96.33,
    ("green", "wood",  "dark"):  51.67,
    ("green", "wood",  "light"): 91.67,
    ("red",   "metal", "dark"):  11.67,
    ("red",   "metal", "light"): 98.00,
    ("red",   "wood",  "dark"):  15.33,
    ("red",   "wood",  "light"): 97.00,
}

# DEFAULT_SUCCESS_RATE: Dict[Tuple[str, str, str], float] = {
#     ("blue",  "metal", "dark"):  99.00,
#     ("blue",  "metal", "light"): 100.00,
#     ("blue",  "wood",  "dark"):  97.67,
#     ("blue",  "wood",  "light"): 100.00,
#     ("green", "metal", "dark"):  98.00,
#     ("green", "metal", "light"): 100.00,
#     ("green", "wood",  "dark"):  98.67,
#     ("green", "wood",  "light"): 99.67,
#     ("red",   "metal", "dark"):  99.00,
#     ("red",   "metal", "light"): 99.67,
#     ("red",   "wood",  "dark"):  96.33,
#     ("red",   "wood",  "light"): 99.67,
# }

def parse_csv_summary(csv_path: Path) -> Dict[str, float]:
    """
    Parse a CSV file and extract summary statistics from the trailing
    comment lines emitted by Diagnosis1_successRate_completionTime.py.
    """
    result = {
        "success_rate": 0.0,
        "median_completion_time": 0.0,
        "std_completion_time": 0.0,
    }

    with open(csv_path, "r") as f:
        content = f.read()

    match = re.search(r"# Success rate:\s*([\d.]+)%", content)
    if match:
        result["success_rate"] = float(match.group(1))

    match = re.search(
        r"# Median completion time:\s*([\d.]+)\s*steps\s*\(std:\s*([\d.]+)\)",
        content,
    )
    if match:
        result["median_completion_time"] = float(match.group(1))
        result["std_completion_time"] = float(match.group(2))
    else:
        # Fallback to old "Avg completion time" format
        match = re.search(
            r"# Avg completion time:\s*([\d.]+)\s*steps\s*\(std:\s*([\d.]+)\)",
            content,
        )
        if match:
            result["median_completion_time"] = float(match.group(1))
            result["std_completion_time"] = float(match.group(2))

    return result


def parse_filename(filename: str) -> Tuple[int, str, str, str, str]:
    """
    Extract (clearance, yaw, color, texture, bg) from a filename like:
        MainExp2-3_rollout100_clearance112_yaw-15_red_metal_dark_epoch20
    """
    name = re.sub(r"_epoch\d+$", "", filename)

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
    grouped_value: Dict[int, Dict[Tuple[str, str, str], float]],
    grouped_err: Optional[Dict[int, Dict[Tuple[str, str, str], float]]],
    ylabel: str,
    title: str,
    output_path: Path,
    annotation_fmt: str = "{:.2f}",
    ylim: Optional[Tuple[float, float]] = None,
):
    clearances = sorted(grouped_value.keys())
    n_sub = len(clearances)
    if n_sub == 0:
        print(f"[WARN] No data to plot for {title}.")
        return

    all_combos = sorted(
        {combo for cmap in grouped_value.values() for combo in cmap.keys()},
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
        cmap = grouped_value[clearance]
        emap = grouped_err[clearance] if grouped_err is not None else {}
        x = np.arange(len(all_combos))
        values = [cmap.get(combo, 0.0) for combo in all_combos]
        errs = (
            [emap.get(combo, 0.0) for combo in all_combos]
            if grouped_err is not None else None
        )

        for i, combo in enumerate(all_combos):
            color, texture, bg = combo
            face, alpha, hatch = bar_style_for_combo(color, texture, bg)
            present = combo in cmap
            err = errs[i] if errs is not None else None
            ax.bar(
                x[i], values[i],
                width=0.8,
                color=face,
                alpha=alpha if present else 0.15,
                edgecolor="black",
                hatch=hatch,
                yerr=err if (err is not None and present) else None,
                capsize=5,
                error_kw={"elinewidth": 2},
            )
            if present:
                y_offset = values[i] + (err if err is not None else 0.0)
                ax.annotate(
                    annotation_fmt.format(values[i]),
                    xy=(x[i], y_offset),
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
        # ax.set_title(f"Tight Hole ({clearance/100:.2f}x peg size)", fontsize=22)
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
            "Plot grouped success rate and completion time: one subplot per "
            "clearance, one bar per (color, texture, bg) combination."
        )
    )
    parser.add_argument(
        "--input", "-i",
        type=str, nargs="+", default=[],
        help=(
            "Input summary CSV files (supports shell glob expansion). "
            "If omitted, the hardcoded DEFAULT_SUCCESS_RATE values are used "
            "and only the success-rate plot is generated."
        ),
    )
    parser.add_argument(
        "--default-clearance", type=int, default=112,
        help="Clearance value to use for the subplot when no input CSVs are given.",
    )
    parser.add_argument("--title", type=str, default="MainExp2",
                        help="Title prefix for plots.")
    parser.add_argument("--output-dir", type=str, default="./",
                        help="Directory to save output plots.")

    args = parser.parse_args()

    # grouped_<metric>[clearance][(color, texture, bg)] = value
    grouped_success: Dict[int, Dict[Tuple[str, str, str], float]] = defaultdict(dict)
    grouped_steps: Dict[int, Dict[Tuple[str, str, str], float]] = defaultdict(dict)
    grouped_steps_std: Dict[int, Dict[Tuple[str, str, str], float]] = defaultdict(dict)

    for csv_path_str in (args.input or []):
        csv_path = Path(csv_path_str)
        if not csv_path.exists():
            print(f"[WARN] Not found: {csv_path}")
            continue
        try:
            stats = parse_csv_summary(csv_path)
            clearance, yaw, color, texture, bg = parse_filename(csv_path.stem)
            combo = (color, texture, bg)
            grouped_success[clearance][combo] = stats["success_rate"]
            grouped_steps[clearance][combo] = stats["median_completion_time"]
            grouped_steps_std[clearance][combo] = stats["std_completion_time"]
            print(
                f"Parsed: {csv_path.name} -> clearance={clearance} "
                f"yaw={yaw} combo={color}_{texture}_{bg} "
                f"success={stats['success_rate']:.1f}% "
                f"steps={stats['median_completion_time']:.1f}"
            )
        except Exception as e:
            print(f"[ERROR] Failed to parse {csv_path}: {e}")

    no_inputs = not grouped_success
    if no_inputs:
        # No CSVs provided: seed a single clearance subplot with default values.
        print(
            f"[info] No input CSVs provided; using DEFAULT_SUCCESS_RATE values "
            f"at clearance={args.default_clearance} and skipping completion-steps plot."
        )
        grouped_success[args.default_clearance] = {}

    # Fill in missing (color, texture, bg) combos for every clearance with the
    # hardcoded default success rates, so bars for them still appear.
    for clearance in list(grouped_success.keys()):
        for combo, default_val in DEFAULT_SUCCESS_RATE.items():
            if combo not in grouped_success[clearance]:
                grouped_success[clearance][combo] = default_val
                print(
                    f"[default] clearance={clearance} combo={'_'.join(combo)} "
                    f"success={default_val:.1f}% (using default)"
                )

    output_dir = Path(args.output_dir)

    print("\n" + "=" * 70)
    print("Generating grouped plots...")
    print("=" * 70)

    plot_metric_by_clearance(
        grouped_success,
        grouped_err=None,
        ylabel="Success Rate (%)",
        # title=f"{args.title} Success Rate (%)",
        title=f"{args.title} success rate under different appearance variations (%)",
        output_path=output_dir / f"{args.title}_grouped_success_rate.png",
        annotation_fmt="{:.1f}",
        ylim=(0, 105),
    )
    if not no_inputs:
        plot_metric_by_clearance(
            grouped_steps,
            grouped_err=grouped_steps_std,
            ylabel="Completion Steps",
            title=f"{args.title} Completion Steps (Median, Success Only)",
            output_path=output_dir / f"{args.title}_grouped_completion_steps.png",
            annotation_fmt="{:.1f}",
        )


if __name__ == "__main__":
    main()
