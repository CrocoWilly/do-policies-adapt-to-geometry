"""
Script to compute per-file mean & std of alignment_steps and final_yaw_error,
plus a Score using the formulas in the attached image.

For each input CSV file:
    - Compute mean and std of `alignment_steps` and `final_yaw_error`
      (final_yaw_error is reported in both rad and deg).
    - Determine score type by filename:
        * filename contains "clearance112" -> score_tight  (eps_yaw = 3.2 deg)
        * filename contains "clearance160" -> score_loose  (eps_yaw = 15  deg)
    - Compute Score using the file's MEAN values:

        score_loose = 1[ |yaw_mean| <= eps_yaw  AND  5 <= steps_mean < 100 ]
                    * (|yaw_mean| / eps_yaw)
                    * (1 - (steps_mean - 5) / 95)

        score_tight = 1[ |yaw_mean| <= eps_yaw  AND 55 <= steps_mean < 100 ]
                    * (1 - |yaw_mean| / eps_yaw)
                    * (1 - (steps_mean - 55) / 45)

Additionally, when multiple files of the same clearance group are provided,
this script aggregates the per-file MEAN values within each group and
computes:
    - mean & std of those per-file means (for both final_yaw_error and
      alignment_steps),
    - a single Score using the aggregated mean values.

The aggregated rows (`ALL_clearance112`, `ALL_clearance160`) are appended to
the output CSV after the per-file rows.

Usage:
    python behaviorAnalysis_scripts/MainExp_Plot_Diagnosis2_alignment_averageOfAll_meanAndStd.py \
        --input ./bc_results/mainExp1/*.csv \
        --output-dir ./
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# Score formula thresholds (yaw error tolerance, in degrees)
EPS_YAW_TIGHT_DEG = 3.2   # used when filename contains "clearance112"
EPS_YAW_LOOSE_DEG = 15.0  # used when filename contains "clearance160"


def parse_csv_demos(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a CSV file and return arrays of (final_yaw_error, alignment_steps)
    over all demo rows. Comment lines (starting with '#') are ignored.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    final_yaw_error = []
    alignment_steps = []

    with open(path, "r") as f:
        reader = csv.DictReader(
            (line for line in f if not line.lstrip().startswith("#"))
        )
        for row in reader:
            try:
                final_yaw_error.append(float(row["final_yaw_error"]))
                alignment_steps.append(float(row["alignment_steps"]))
            except (KeyError, ValueError, TypeError):
                continue

    return np.asarray(final_yaw_error, dtype=float), np.asarray(alignment_steps, dtype=float)


def detect_clearance_type(filename: str) -> Optional[str]:
    """Return 'tight' for clearance112, 'loose' for clearance160, else None."""
    if re.search(r"clearance112", filename, flags=re.IGNORECASE):
        return "tight"
    if re.search(r"clearance160", filename, flags=re.IGNORECASE):
        return "loose"
    return None


def compute_score_loose(yaw_deg: float, steps: float) -> float:
    """
    score_loose = 1[ |yaw| <= eps_yaw AND 5 <= steps < 100 ]
                * (|yaw| / eps_yaw)
                * (1 - (steps - 5) / 95)
    """
    eps_yaw = EPS_YAW_LOOSE_DEG
    yaw_abs = abs(yaw_deg)
    if yaw_abs <= eps_yaw and 5.0 <= steps < 100.0:
        return (yaw_abs / eps_yaw) * (1.0 - (steps - 5.0) / 95.0)
    return 0.0


def compute_score_tight(yaw_deg: float, steps: float) -> float:
    """
    score_tight = 1[ |yaw| <= eps_yaw AND 55 <= steps < 100 ]
                * (1 - |yaw| / eps_yaw)
                * (1 - (steps - 55) / 45)
    """
    eps_yaw = EPS_YAW_TIGHT_DEG
    yaw_abs = abs(yaw_deg)
    if yaw_abs <= eps_yaw and 55.0 <= steps < 100.0:
        return (1.0 - yaw_abs / eps_yaw) * (1.0 - (steps - 55.0) / 45.0)
    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-file mean/std of alignment_steps and final_yaw_error, "
                    "and a Score using score_tight (clearance112) or score_loose (clearance160)."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        nargs="+",
        required=True,
        help="Paths to input alignment CSV files."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./",
        help="Directory to save the summary CSV (default: ./)."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional explicit output CSV filename. "
             "Default: <output-dir>/alignment_meanStdScore_summary.csv"
    )
    parser.add_argument(
        "--ddof",
        type=int,
        default=0,
        help="Delta Degrees of Freedom for std (0 = population std, 1 = sample std). Default: 0."
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = (
        Path(args.output_csv)
        if args.output_csv
        else output_dir / "alignment_meanStdScore_summary.csv"
    )

    rows = []
    # Per-clearance accumulators of per-file MEAN values (yaw in rad, steps in raw count).
    group_yaw_means_rad = defaultdict(list)
    group_steps_means = defaultdict(list)
    # Per-clearance accumulators of per-file Score values.
    group_scores = defaultdict(list)

    print("=" * 100)
    print("Per-file statistics (final_yaw_error & alignment_steps) and Score")
    print("=" * 100)

    for csv_path in args.input:
        try:
            yaw_arr, steps_arr = parse_csv_demos(csv_path)
            if yaw_arr.size == 0 or steps_arr.size == 0:
                print(f"[WARN] No valid rows in: {csv_path}")
                continue

            yaw_mean_rad = float(np.mean(yaw_arr))
            yaw_std_rad = float(np.std(yaw_arr, ddof=args.ddof))
            yaw_mean_deg = float(np.degrees(yaw_mean_rad))
            yaw_std_deg = float(np.degrees(yaw_std_rad))

            steps_mean = float(np.mean(steps_arr))
            steps_std = float(np.std(steps_arr, ddof=args.ddof))

            filename = Path(csv_path).stem
            ctype = detect_clearance_type(filename)

            if ctype == "tight":
                score_kind = "score_tight"
                eps_yaw_used = EPS_YAW_TIGHT_DEG
                score = compute_score_tight(yaw_mean_deg, steps_mean)
            elif ctype == "loose":
                score_kind = "score_loose"
                eps_yaw_used = EPS_YAW_LOOSE_DEG
                score = compute_score_loose(yaw_mean_deg, steps_mean)
            else:
                score_kind = "unknown"
                eps_yaw_used = float("nan")
                score = float("nan")
                print(f"[WARN] Cannot detect clearance112/clearance160 in: {filename}; "
                      f"score skipped.")

            row = {
                "filename": filename,
                "clearance_type": ctype if ctype else "unknown",
                "n_demos": int(yaw_arr.size),
                "final_yaw_error_mean_rad": yaw_mean_rad,
                "final_yaw_error_std_rad": yaw_std_rad,
                "final_yaw_error_mean_deg": yaw_mean_deg,
                "final_yaw_error_std_deg": yaw_std_deg,
                "alignment_steps_mean": steps_mean,
                "alignment_steps_std": steps_std,
                "score_kind": score_kind,
                "eps_yaw_deg": eps_yaw_used,
                "score": score,
                "score_mean": score,
                "score_std": float("nan"),
            }
            rows.append(row)

            # Accumulate per-file means into the group of the same clearance.
            if ctype in ("tight", "loose"):
                group_yaw_means_rad[ctype].append(yaw_mean_rad)
                group_steps_means[ctype].append(steps_mean)
                if not np.isnan(score):
                    group_scores[ctype].append(score)

            print(f"\n{filename}")
            print(f"  type            : {row['clearance_type']:5s}   n_demos = {row['n_demos']}")
            print(f"  final_yaw_error : mean = {yaw_mean_deg:8.4f} deg   std = {yaw_std_deg:8.4f} deg")
            print(f"                    mean = {yaw_mean_rad:8.6f} rad   std = {yaw_std_rad:8.6f} rad")
            print(f"  alignment_steps : mean = {steps_mean:8.4f}       std = {steps_std:8.4f}")
            if score_kind == "unknown":
                print(f"  {score_kind:11s}     : N/A")
            else:
                print(f"  {score_kind:11s}     : {score:.6f}   (eps_yaw = {eps_yaw_used} deg)")
        except Exception as e:
            print(f"[ERROR] Failed to process {csv_path}: {e}")

    print("\n" + "=" * 100)

    # ------------------------------------------------------------------
    # Aggregate per-clearance summary across all input files of that group.
    # For each clearance group, compute mean & std of the per-file means,
    # then compute a single Score from those aggregated means.
    # ------------------------------------------------------------------
    aggregate_rows = []
    # Track per-group score mean/std for the final_score row.
    group_score_stats = {}
    if group_yaw_means_rad:
        print("Per-clearance aggregated statistics (across input files)")
        print("=" * 100)

    for ctype in ("tight", "loose"):
        yaw_list_rad = group_yaw_means_rad.get(ctype, [])
        steps_list = group_steps_means.get(ctype, [])
        if not yaw_list_rad or not steps_list:
            continue

        yaw_arr_rad = np.asarray(yaw_list_rad, dtype=float)
        steps_arr = np.asarray(steps_list, dtype=float)

        agg_yaw_mean_rad = float(np.mean(yaw_arr_rad))
        agg_yaw_std_rad = float(np.std(yaw_arr_rad, ddof=args.ddof))
        agg_yaw_mean_deg = float(np.degrees(agg_yaw_mean_rad))
        agg_yaw_std_deg = float(np.degrees(agg_yaw_std_rad))

        agg_steps_mean = float(np.mean(steps_arr))
        agg_steps_std = float(np.std(steps_arr, ddof=args.ddof))

        if ctype == "tight":
            score_kind = "score_tight"
            eps_yaw_used = EPS_YAW_TIGHT_DEG
            agg_score = compute_score_tight(agg_yaw_mean_deg, agg_steps_mean)
            label = "ALL_clearance112"
        else:
            score_kind = "score_loose"
            eps_yaw_used = EPS_YAW_LOOSE_DEG
            agg_score = compute_score_loose(agg_yaw_mean_deg, agg_steps_mean)
            label = "ALL_clearance160"

        # Mean / std of per-file Score values within this clearance group.
        score_list = group_scores.get(ctype, [])
        if score_list:
            score_arr = np.asarray(score_list, dtype=float)
            score_mean = float(np.mean(score_arr))
            score_std = float(np.std(score_arr, ddof=args.ddof))
        else:
            score_mean = float("nan")
            score_std = float("nan")
        group_score_stats[ctype] = (score_mean, score_std)

        agg_row = {
            "filename": label,
            "clearance_type": ctype,
            "n_demos": int(yaw_arr_rad.size),  # here: number of files aggregated
            "final_yaw_error_mean_rad": agg_yaw_mean_rad,
            "final_yaw_error_std_rad": agg_yaw_std_rad,
            "final_yaw_error_mean_deg": agg_yaw_mean_deg,
            "final_yaw_error_std_deg": agg_yaw_std_deg,
            "alignment_steps_mean": agg_steps_mean,
            "alignment_steps_std": agg_steps_std,
            "score_kind": score_kind,
            "eps_yaw_deg": eps_yaw_used,
            "score": agg_score,
            "score_mean": score_mean,
            "score_std": score_std,
        }
        aggregate_rows.append(agg_row)

        print(f"\n{label}  (n_files = {agg_row['n_demos']})")
        print(f"  final_yaw_error : mean = {agg_yaw_mean_deg:8.4f} deg   std = {agg_yaw_std_deg:8.4f} deg")
        print(f"                    mean = {agg_yaw_mean_rad:8.6f} rad   std = {agg_yaw_std_rad:8.6f} rad")
        print(f"  alignment_steps : mean = {agg_steps_mean:8.4f}       std = {agg_steps_std:8.4f}")
        print(f"  {score_kind:11s}     : {agg_score:.6f}   (eps_yaw = {eps_yaw_used} deg)  [from aggregated means]")
        print(f"  score (per-file): mean = {score_mean:.6f}   std = {score_std:.6f}")

    if aggregate_rows:
        print("\n" + "=" * 100)

    # ------------------------------------------------------------------
    # Final score = (score_tight + score_loose) / 2
    # Score field      : (aggregated tight + aggregated loose) / 2
    # score_mean       : (mean_tight + mean_loose) / 2
    # score_std        : sqrt(std_tight^2 + std_loose^2) / 2
    #                    (assuming tight & loose per-file scores are independent)
    # ------------------------------------------------------------------
    final_row = None
    tight_agg = next((r for r in aggregate_rows if r["clearance_type"] == "tight"), None)
    loose_agg = next((r for r in aggregate_rows if r["clearance_type"] == "loose"), None)
    if tight_agg is not None and loose_agg is not None:
        final_score = (tight_agg["score"] + loose_agg["score"]) / 2.0
        tight_mean, tight_std = group_score_stats["tight"]
        loose_mean, loose_std = group_score_stats["loose"]
        final_score_mean = (tight_mean + loose_mean) / 2.0
        final_score_std = float(np.sqrt(tight_std ** 2 + loose_std ** 2) / 2.0)

        final_row = {
            "filename": "FINAL_SCORE",
            "clearance_type": "combined",
            "n_demos": int(tight_agg["n_demos"] + loose_agg["n_demos"]),
            "final_yaw_error_mean_rad": float("nan"),
            "final_yaw_error_std_rad": float("nan"),
            "final_yaw_error_mean_deg": float("nan"),
            "final_yaw_error_std_deg": float("nan"),
            "alignment_steps_mean": float("nan"),
            "alignment_steps_std": float("nan"),
            "score_kind": "final_score",
            "eps_yaw_deg": float("nan"),
            "score": final_score,
            "score_mean": final_score_mean,
            "score_std": final_score_std,
        }

        print("FINAL SCORE  =  (score_tight + score_loose) / 2")
        print("=" * 100)
        print(f"  score (from aggregated means) : {final_score:.6f}")
        print(f"  score_mean (per-file based)   : {final_score_mean:.6f}")
        print(f"  score_std  (per-file based)   : {final_score_std:.6f}")
        print("=" * 100)
    elif aggregate_rows:
        print("[INFO] FINAL_SCORE not computed: need both clearance112 and clearance160 groups.")
        print("=" * 100)

    # Combine per-file rows, aggregated rows, and final score into the same CSV.
    all_rows = rows + aggregate_rows + ([final_row] if final_row is not None else [])

    if all_rows:
        with open(output_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Summary written to: {output_csv_path}")
    else:
        print("No valid files processed; no summary written.")


if __name__ == "__main__":
    main()
