"""
Script to segment the alignment phase for each episode in an HDF5 dataset.

For each episode, finds the time-step where yaw rotation begins, and records:
    - peg_bottom_z, hole_top_z, z_diff (= peg_bottom_z - hole_top_z) at that time-step
    - yaw_change values between consecutive time-steps from that point onward

Start yaw rotation condition:
    First time-step t where |ee_yaw_t - ee_yaw_{t-1}| >= yaw_change_threshold (default: 0.02)

Where:
    - peg_bottom_z = peg_center_z - 0.5 * peg_length
    - hole_top_z = hole_bottom_z + hole_height
    - peg_pos_world is at peg center
    - hole_pos_world is at hole bottom

Usage:
    python MainExp_alignmentPhaseCheck.py --input MainExp_dataset_forTraining.hdf5
    python MainExp_alignmentPhaseCheck.py --input MainExp_dataset_forTraining.hdf5 --output segmentation.csv
    python MainExp_alignmentPhaseCheck.py --input MainExp_dataset_forTraining.hdf5 --yaw-change-threshold 0.002
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def find_yaw_rotation_start(
    ee_yaw: np.ndarray,
    yaw_change_threshold: float = 0.002,
) -> Tuple[int, np.ndarray]:
    """
    Find the first time-step where yaw rotation begins.

    Args:
        ee_yaw: (T,) array of end-effector yaw angles
        yaw_change_threshold: threshold for detecting yaw change start

    Returns:
        (start_index, yaw_changes) where yaw_changes[i] = wrapped(yaw[i+1] - yaw[i])
    """
    if len(ee_yaw) < 2:
        return -1, np.array([])

    yaw_changes = wrap_to_pi(np.diff(ee_yaw))
    abs_yaw_changes = np.abs(yaw_changes)

    # Find first index where |yaw_change| >= threshold
    # yaw_changes[i] is the change from step i to step i+1
    # So if yaw_changes[i] triggers, the rotation starts at step i+1
    # But we define "start" as the step where the change is observed,
    # i.e. the transition from step i to step i+1 means step i+1 is the first "rotated" step
    candidates = np.where(abs_yaw_changes >= yaw_change_threshold)[0]

    if len(candidates) == 0:
        return -1, yaw_changes

    # The rotation starts at candidate[0] + 1 (the step that moved)
    # But conventionally, we report the step index where the change is first detected
    # yaw_changes[t] = yaw[t+1] - yaw[t], so t is the step before rotation
    # We report t as the start step (the last step before rotation begins)
    # Actually, let's report t+1 as the start (the step that first shows rotation)
    # But to match the reference script convention, let's use t+1
    # Actually the reference uses index directly from yaw_diff_padded where pad[0]=0
    # and pad[i] = |yaw[i] - yaw[i-1]|, so index i means step i shows change from i-1.
    # Let's do the same: start_index is the step that first shows yaw change.
    start_index = int(candidates[0] + 1)  # +1 because diff shifts by 1

    return start_index, yaw_changes


def process_episode(
    obs_group,
    peg_length: float,
    hole_height: float,
    yaw_change_threshold: float,
    ee_yaw_key: str,
    peg_pos_key: str,
    hole_pos_key: str,
) -> dict:
    """Process a single episode and return segmentation results."""
    ee_yaw = obs_group[ee_yaw_key][()]
    if ee_yaw.ndim == 2:
        ee_yaw = ee_yaw.flatten()

    peg_pos = obs_group[peg_pos_key][()]
    hole_pos = obs_group[hole_pos_key][()]

    T = len(ee_yaw)
    if hole_pos.ndim == 1:
        hole_pos = np.tile(hole_pos[None, :], (T, 1))

    # Compute z values for all steps
    peg_center_z = peg_pos[:, 2]
    peg_bottom_z = peg_center_z - 0.5 * peg_length
    hole_bottom_z = hole_pos[:, 2]
    hole_top_z = hole_bottom_z + hole_height
    z_diff = peg_bottom_z - hole_top_z

    # Find yaw rotation start
    start_idx, yaw_changes = find_yaw_rotation_start(ee_yaw, yaw_change_threshold)

    result = {
        "total_steps": T,
        "yaw_rotation_start_step": start_idx,
        "peg_bottom_z_at_start": None,
        "hole_top_z_at_start": None,
        "z_diff_at_start": None,
        "yaw_changes_after_start": [],
    }

    if start_idx >= 0 and start_idx < T:
        result["peg_bottom_z_at_start"] = float(peg_bottom_z[start_idx])
        result["hole_top_z_at_start"] = float(hole_top_z[start_idx])
        result["z_diff_at_start"] = float(z_diff[start_idx])

        # yaw_changes[i] is yaw[i+1] - yaw[i], valid for i in [0, T-2]
        # From start_idx onward: changes from start_idx to start_idx+1, etc.
        # yaw_changes index for step start_idx -> start_idx+1 is yaw_changes[start_idx]
        # But we want from start_idx-1 -> start_idx (that's the first change), which is yaw_changes[start_idx-1]
        # Actually start_idx = candidates[0]+1, so yaw_changes[start_idx-1] is the triggering change
        # Let's include from the triggering change onward
        change_start = max(start_idx - 1, 0)
        result["yaw_changes_after_start"] = yaw_changes[change_start:].tolist()

    return result


def process_hdf5_file(
    input_hdf5: str,
    output_csv: Optional[str] = None,
    demo_prefix: str = "demo_",
    ee_yaw_key: str = "ee_yaw_rel",
    peg_pos_key: str = "peg_pos_world",
    hole_pos_key: str = "hole_pos_world",
    peg_length: float = 0.12,
    hole_height: float = 0.08,
    yaw_change_threshold: float = 0.002,
    verbose: bool = True,
) -> dict:
    """Process an HDF5 file and find yaw rotation start for each episode."""
    input_path = Path(input_hdf5)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    all_results = {}

    with h5py.File(input_path, "r") as f:
        if "data" not in f:
            raise KeyError("HDF5 file missing 'data' group")

        data_grp = f["data"]
        demo_names = sorted([k for k in data_grp.keys() if k.startswith(demo_prefix)])

        if verbose:
            print(f"Found {len(demo_names)} episodes in {input_path.name}")
            print(f"Peg length: {peg_length}, Hole height: {hole_height}")
            print(f"Yaw change threshold: {yaw_change_threshold}")
            print("=" * 120)

        z_diffs_at_start = []
        detected_count = 0

        for demo_name in demo_names:
            demo = data_grp[demo_name]
            if "obs" not in demo:
                if verbose:
                    print(f"[WARNING] {demo_name}: No 'obs' group found, skipping")
                continue

            obs = demo["obs"]
            missing = [k for k in [ee_yaw_key, peg_pos_key, hole_pos_key] if k not in obs]
            if missing:
                if verbose:
                    print(f"[WARNING] {demo_name}: Missing keys in obs: {missing}, skipping")
                continue

            try:
                result = process_episode(
                    obs, peg_length, hole_height, yaw_change_threshold,
                    ee_yaw_key, peg_pos_key, hole_pos_key,
                )
                all_results[demo_name] = result

                start = result["yaw_rotation_start_step"]
                T = result["total_steps"]

                if start >= 0:
                    detected_count += 1
                    z_diffs_at_start.append(result["z_diff_at_start"])
                    yaw_changes = result["yaw_changes_after_start"]

                    if verbose:
                        # Show first few yaw changes
                        n_show = min(10, len(yaw_changes))
                        yaw_str = ", ".join(
                            f"{np.degrees(yc):+.4f}" for yc in yaw_changes[:n_show]
                        )
                        if len(yaw_changes) > n_show:
                            yaw_str += f", ... ({len(yaw_changes)} total)"
                        print(
                            f"{demo_name}: yaw_start_step={start:3d}/{T}  "
                            f"peg_bottom_z={result['peg_bottom_z_at_start']:.6f}  "
                            f"hole_top_z={result['hole_top_z_at_start']:.6f}  "
                            f"z_diff={result['z_diff_at_start']:+.6f}  "
                            f"yaw_changes(deg)=[{yaw_str}]"
                        )
                else:
                    if verbose:
                        print(f"{demo_name}: No yaw rotation detected (T={T})")

            except Exception as e:
                if verbose:
                    print(f"[ERROR] {demo_name}: {e}")
                continue

        # Summary
        summary = {
            "total_episodes": len(all_results),
            "detected_count": detected_count,
            "avg_z_diff_at_start": float(np.mean(z_diffs_at_start)) if z_diffs_at_start else None,
            "min_z_diff_at_start": float(np.min(z_diffs_at_start)) if z_diffs_at_start else None,
            "max_z_diff_at_start": float(np.max(z_diffs_at_start)) if z_diffs_at_start else None,
            "std_z_diff_at_start": float(np.std(z_diffs_at_start)) if z_diffs_at_start else None,
        }

        if verbose:
            print("=" * 120)
            print(f"Total episodes: {summary['total_episodes']}")
            print(f"Episodes with yaw rotation detected: {summary['detected_count']}")
            if z_diffs_at_start:
                avg_start_steps = np.mean([
                    r["yaw_rotation_start_step"] for r in all_results.values()
                    if r["yaw_rotation_start_step"] >= 0
                ])
                print(f"Average yaw rotation start step: {avg_start_steps:.1f}")
                print(
                    f"z_diff at start — "
                    f"mean: {summary['avg_z_diff_at_start']:+.6f}, "
                    f"min: {summary['min_z_diff_at_start']:+.6f}, "
                    f"max: {summary['max_z_diff_at_start']:+.6f}, "
                    f"std: {summary['std_z_diff_at_start']:.6f}"
                )

    # Save CSV
    if output_csv:
        output_path = Path(output_csv)
        with open(output_path, "w") as fout:
            fout.write(
                "episode,yaw_rotation_start_step,total_steps,"
                "peg_bottom_z,hole_top_z,z_diff,"
                "yaw_changes_deg\n"
            )
            for demo_name, data in sorted(all_results.items()):
                start = data["yaw_rotation_start_step"]
                T = data["total_steps"]
                if start >= 0:
                    yaw_str = ";".join(f"{np.degrees(yc):.6f}" for yc in data["yaw_changes_after_start"])
                    fout.write(
                        f"{demo_name},{start},{T},"
                        f"{data['peg_bottom_z_at_start']:.6f},"
                        f"{data['hole_top_z_at_start']:.6f},"
                        f"{data['z_diff_at_start']:.6f},"
                        f"{yaw_str}\n"
                    )
                else:
                    fout.write(f"{demo_name},{start},{T},,,,\n")

            # Summary
            fout.write(f"\n# Summary\n")
            fout.write(f"# Peg length: {peg_length}, Hole height: {hole_height}\n")
            fout.write(f"# Yaw change threshold: {yaw_change_threshold}\n")
            fout.write(f"# Total episodes: {summary['total_episodes']}\n")
            fout.write(f"# Detected: {summary['detected_count']}\n")
            if summary['avg_z_diff_at_start'] is not None:
                fout.write(f"# z_diff at start — mean: {summary['avg_z_diff_at_start']:+.6f}, "
                           f"min: {summary['min_z_diff_at_start']:+.6f}, "
                           f"max: {summary['max_z_diff_at_start']:+.6f}\n")

        if verbose:
            print(f"\nResults saved to: {output_path}")

    return {"episodes": all_results, "summary": summary}


def main():
    parser = argparse.ArgumentParser(
        description="Segment alignment phase: find yaw rotation start and record z-heights and yaw changes"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to input HDF5 file",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Optional: path to save results as CSV file",
    )
    parser.add_argument(
        "--demo-prefix", type=str, default="demo_",
        help="Prefix for demo episode names (default: 'demo_')",
    )
    parser.add_argument(
        "--ee-yaw-key", type=str, default="ee_yaw_rel",
        help="Key for end-effector yaw in obs group (default: 'ee_yaw_rel')",
    )
    parser.add_argument(
        "--peg-pos-key", type=str, default="peg_pos_world",
        help="Key for peg world position in obs group (default: 'peg_pos_world')",
    )
    parser.add_argument(
        "--hole-pos-key", type=str, default="hole_pos_world",
        help="Key for hole position in obs group (default: 'hole_pos_world')",
    )
    parser.add_argument(
        "--peg-length", type=float, default=0.12,
        help="Length of the peg (default: 0.12)",
    )
    parser.add_argument(
        "--hole-height", type=float, default=0.08,
        help="Height of the hole (default: 0.08)",
    )
    parser.add_argument(
        "--yaw-change-threshold", type=float, default=0.002,
        help="Threshold for yaw change to detect rotation start (default: 0.002 rad)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress detailed output",
    )

    args = parser.parse_args()

    process_hdf5_file(
        input_hdf5=args.input,
        output_csv=args.output,
        demo_prefix=args.demo_prefix,
        ee_yaw_key=args.ee_yaw_key,
        peg_pos_key=args.peg_pos_key,
        hole_pos_key=args.hole_pos_key,
        peg_length=args.peg_length,
        hole_height=args.hole_height,
        yaw_change_threshold=args.yaw_change_threshold,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
