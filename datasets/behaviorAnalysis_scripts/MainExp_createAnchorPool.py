"""
Script to create an anchor pool from a human demo HDF5 dataset.

STEP 1: Scan each demo step-by-step and collect time-steps that satisfy anchor conditions:
    1. z_diff ∈ [0, +0.0125]  (peg close to hole top, not yet in contact / inserted)
    2. yaw_error = abs(ee_yaw_rel[t]) > 3.5 deg  (significant yaw error remains)

    Where:
        z_diff = peg_bottom_z - hole_top_z
        peg_bottom_z = peg_center_z - 0.5 * peg_length
        hole_top_z = hole_bottom_z + hole_height

    Alignment start condition (same as BehaviorAnalysis_CalcAlignmentYawCorrection.py):
        first time (z_diff <= start_threshold) OR (|yaw_change| >= yaw_change_threshold)

STEP 1.5 (Second-round filter): For each demo, find the rotation start anchor:
    - Walk consecutive anchors; if |yaw_error[i+1] - yaw_error[i]| <= yaw_rotation_diff_threshold,
      anchor[i] is before rotation → remove it.
    - Once the difference > threshold, that anchor is the rotation start.
      Keep it and all subsequent anchors from that demo.

STEP 2: Save each anchor as metadata:
    A_i = (demo_id, t_i, size_i, color_i, obs_i, Δz_i, yaw_error_i)

Output: HDF5 file with anchor pool.

Usage:
    python MainExp_createAnchorPool.py --input MainExp_dataset_forTraining.hdf5
    python MainExp_createAnchorPool.py --input dataset.hdf5 --output MainExp_dataset_anchorPool_forTraining.hdf5
"""

import argparse
import h5py
import numpy as np
from pathlib import Path


# Map box_color_no to color label
_COLOR_MAP = {0: "red", 1: "green", 2: "blue"}
_PEG_TEXTURE_MAP = {0: "Metal", 1: "Wood"}
_TABLE_TEXTURE_MAP = {0: "dark", 1: "light"}


def color_no_to_label(box_color_no: int) -> str:
    """Map box_color_no {0, 1, 2} to 'red', 'green', 'blue'."""
    return _COLOR_MAP.get(int(box_color_no), "unknown")


def peg_texture_no_to_label(peg_texture_no: int) -> str:
    """Map peg_texture_no {0, 1} to 'Metal', 'Wood'."""
    return _PEG_TEXTURE_MAP.get(int(peg_texture_no), "unknown")


def table_texture_no_to_label(table_texture_no: int) -> str:
    """Map table_texture_no {0, 1} to 'dark', 'light'."""
    return _TABLE_TEXTURE_MAP.get(int(table_texture_no), "unknown")


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def find_alignment_start(
    z_diff: np.ndarray,
    ee_yaw_world: np.ndarray,
    start_threshold: float = 0.01,
    yaw_change_threshold: float = 0.002,
) -> int:
    """
    Find the alignment start step using the hybrid condition:
        first time (z_diff <= start_threshold) OR (|yaw_change| >= yaw_change_threshold)

    Returns:
        alignment_start index, or -1 if not found.
    """
    T = len(z_diff)
    if T < 2:
        return -1

    z_mask = z_diff <= start_threshold

    yaw_diff = np.abs(wrap_to_pi(np.diff(ee_yaw_world)))
    yaw_diff_padded = np.concatenate([[0], yaw_diff])
    yaw_mask = yaw_diff_padded >= yaw_change_threshold

    combined = z_mask | yaw_mask
    indices = np.where(combined)[0]

    if len(indices) == 0:
        return -1
    return int(indices[0])


def create_anchor_pool(
    input_hdf5: str,
    output_hdf5: str = None,
    peg_length: float = 0.12,
    hole_height: float = 0.08,
    z_diff_min: float = 0.0,
    z_diff_max: float = 0.0125,
    yaw_error_threshold_deg: float = 4.2,
    start_threshold: float = 0.01,
    yaw_change_threshold: float = 0.002,
    yaw_rotation_diff_threshold: float = 0.03,
    demo_prefix: str = "demo_",
    verbose: bool = True,
) -> dict:
    """
    Create anchor pool from human demo dataset.

    Args:
        input_hdf5: Path to input HDF5 dataset.
        output_hdf5: Path to output anchor pool HDF5. If None, derived from input.
        peg_length: Peg length (default: 0.12).
        hole_height: Hole height (default: 0.08).
        z_diff_min: Minimum z_diff (inclusive) for anchor condition (default: 0.0).
        z_diff_max: Maximum z_diff (inclusive) for anchor condition (default: 0.0125).
        yaw_error_threshold_deg: Minimum remaining yaw error in degrees (default: 4.2).
        start_threshold: z_diff threshold for alignment start detection.
        yaw_change_threshold: Yaw change threshold for alignment start detection.
        yaw_rotation_diff_threshold: Consecutive yaw_error_deg diff threshold for rotation start (default: 0.03).
        demo_prefix: Prefix for demo names in HDF5.
        verbose: Print detailed output.

    Returns:
        Summary dict with anchor count and statistics.
    """
    input_path = Path(input_hdf5)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_hdf5 is None:
        output_hdf5 = str(input_path.parent / (input_path.stem + "_anchorPool.hdf5"))
    output_path = Path(output_hdf5)
    output_csv_path = output_path.with_suffix(".csv")

    yaw_error_threshold_rad = np.radians(yaw_error_threshold_deg)

    anchors = []  # Collect anchor metadata before writing

    with h5py.File(input_path, "r") as fin:
        if "data" not in fin:
            raise KeyError("HDF5 file missing 'data' group")

        data_grp = fin["data"]
        demo_names = sorted([k for k in data_grp.keys() if k.startswith(demo_prefix)])

        if verbose:
            print(f"Input: {input_path.name}")
            print(f"Found {len(demo_names)} demos")
            print(f"Anchor conditions: z_diff ∈ [{z_diff_min}, {z_diff_max}], "
                  f"yaw_error = |ee_yaw_rel| > {yaw_error_threshold_deg}°")
            print("-" * 90)

        for demo_name in demo_names:
            demo = data_grp[demo_name]
            obs = demo["obs"]

            # Load required data
            ee_yaw_world = obs["ee_yaw_world"][()].flatten()
            ee_yaw_rel = obs["ee_yaw_rel"][()].flatten()
            peg_pos_world = obs["peg_pos_world"][()]
            hole_pos_world = obs["hole_pos_world"][()]

            T = len(ee_yaw_world)
            if T < 2:
                continue

            # Compute z_diff for all timesteps
            peg_bottom_z = peg_pos_world[:, 2] - 0.5 * peg_length
            hole_top_z = hole_pos_world[:, 2] + hole_height
            z_diff = peg_bottom_z - hole_top_z

            # Find alignment start
            alignment_start = find_alignment_start(
                z_diff, ee_yaw_world, start_threshold, yaw_change_threshold
            )
            if alignment_start < 0:
                if verbose:
                    print(f"  {demo_name}: no alignment phase detected, skipping")
                continue

            # Get metadata: clearance (scaled), color, and textures
            clearance_scaled = float(obs["clearance"][0, 0])
            color_label = color_no_to_label(obs["box_color_no"][0])
            peg_texture_label = peg_texture_no_to_label(obs["peg_texture_no"][0])
            table_texture_label = table_texture_no_to_label(obs["table_texture_no"][0])

            # Scan timesteps from alignment_start to end
            demo_anchor_count = 0
            for t in range(alignment_start, T):
                z_t = z_diff[t]
                yaw_err_t = abs(ee_yaw_rel[t])

                # Condition 1: z_diff in [z_diff_min, z_diff_max]
                if z_t < z_diff_min or z_t > z_diff_max:
                    continue

                # Condition 2: yaw_error > threshold
                if yaw_err_t <= yaw_error_threshold_rad:
                    continue

                # This timestep is an anchor
                anchors.append({
                    "demo_id": demo_name,
                    "t": t,
                    "size": clearance_scaled,
                    "color": color_label,
                    "peg_texture": peg_texture_label,
                    "table_texture": table_texture_label,
                    "delta_z": float(z_t),
                    "yaw_error_deg": float(np.degrees(yaw_err_t)),
                })
                demo_anchor_count += 1

            if verbose:
                if demo_anchor_count > 0:
                    print(f"  {demo_name}: {demo_anchor_count} anchors "
                          f"(size={clearance_scaled:.4f}, color={color_label}, "
                          f"alignment_start={alignment_start}, T={T})")
                else:
                    print(f"  {demo_name}: 0 anchors "
                          f"(size={clearance_scaled:.4f}, color={color_label}, "
                          f"alignment_start={alignment_start}, T={T})")

    if verbose:
        print("-" * 90)
        print(f"Round 1 anchors collected: {len(anchors)}")

    # --- STEP 1.5: Second-round filter — find rotation start per demo ---
    from collections import defaultdict
    demo_anchors = defaultdict(list)
    for anc in anchors:
        demo_anchors[anc["demo_id"]].append(anc)

    filtered_anchors = []
    for demo_id in sorted(demo_anchors.keys()):
        demo_list = demo_anchors[demo_id]  # already sorted by t from STEP 1
        # Only apply rotation start filter when clearance < 0
        if demo_list[0]["size"] < 0:
            rotation_start_idx = 0  # default: keep all if only 1 anchor
            for j in range(len(demo_list) - 1):
                diff = abs(demo_list[j + 1]["yaw_error_deg"] - demo_list[j]["yaw_error_deg"])
                if diff > yaw_rotation_diff_threshold:
                    # Rotation starts at anchor j — keep j and all after
                    rotation_start_idx = j
                    break
                else:
                    # Not rotating yet, skip anchor j
                    rotation_start_idx = j + 1
            kept = demo_list[rotation_start_idx:]
        else:
            kept = demo_list
        filtered_anchors.extend(kept)
        if verbose:
            print(f"  {demo_id}: {len(demo_list)} -> {len(kept)} anchors "
                  f"(rotation start at t={kept[0]['t'] if kept else 'N/A'})")

    anchors = filtered_anchors
    if verbose:
        print(f"Round 2 anchors after rotation filter: {len(anchors)}")

    # --- STEP 2: Write anchor pool to output HDF5 ---
    if len(anchors) == 0:
        print("No anchors found. Output file will not be created.")
        return {"total_anchors": 0, "anchors": []}

    with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
        # Store parameters as attributes
        fout.attrs["source_file"] = str(input_path.name)
        fout.attrs["z_diff_min"] = z_diff_min
        fout.attrs["z_diff_max"] = z_diff_max
        fout.attrs["yaw_error_threshold_deg"] = yaw_error_threshold_deg
        fout.attrs["yaw_rotation_diff_threshold"] = yaw_rotation_diff_threshold
        fout.attrs["peg_length"] = peg_length
        fout.attrs["hole_height"] = hole_height
        fout.attrs["total_anchors"] = len(anchors)

        anchor_grp = fout.create_group("anchors")

        for i, anc in enumerate(anchors):
            demo_name = anc["demo_id"]
            t = anc["t"]
            obs_src = fin["data"][demo_name]["obs"]

            a_grp = anchor_grp.create_group(f"anchor_{i}")

            # Metadata
            a_grp.attrs["demo_id"] = demo_name
            a_grp.attrs["t"] = t
            a_grp.attrs["size"] = anc["size"]
            a_grp.attrs["color"] = anc["color"]
            a_grp.attrs["peg_texture"] = anc["peg_texture"]
            a_grp.attrs["table_texture"] = anc["table_texture"]
            a_grp.attrs["delta_z"] = anc["delta_z"]
            a_grp.attrs["yaw_error_deg"] = anc["yaw_error_deg"]

            # Observation at timestep t
            obs_grp = a_grp.create_group("obs")
            obs_grp.create_dataset("ee_pos_world", data=obs_src["ee_pos_world"][t])
            obs_grp.create_dataset("ee_yaw_world", data=obs_src["ee_yaw_world"][t])
            obs_grp.create_dataset("table_side_cam", data=obs_src["table_side_cam"][t])

        if verbose:
            print(f"\nAnchor pool saved to: {output_path}")
            print(f"  Total anchors: {len(anchors)}")

            # Summary by size and color
            from collections import Counter
            size_counts = Counter(a["size"] for a in anchors)
            color_counts = Counter(a["color"] for a in anchors)
            combo_counts = Counter((a["size"], a["color"]) for a in anchors)
            print(f"  By size:  {dict(size_counts)}")
            print(f"  By color: {dict(color_counts)}")
            print(f"  By (size, color): {dict(combo_counts)}")

            yaw_errors = [a["yaw_error_deg"] for a in anchors]
            dz_values = [a["delta_z"] for a in anchors]
            print(f"  yaw_error range: [{min(yaw_errors):.2f}, {max(yaw_errors):.2f}] deg")
            print(f"  delta_z range:   [{min(dz_values):.6f}, {max(dz_values):.6f}]")

    # --- Also write anchor pool to CSV ---
    with open(output_csv_path, "w") as csvf:
        csvf.write("anchor_id,demo_id,t,size,color,peg_texture,table_texture,delta_z,yaw_error_deg\n")
        for i, anc in enumerate(anchors):
            csvf.write(f"{i},{anc['demo_id']},{anc['t']},{anc['size']:.6f},"
                       f"{anc['color']},{anc['peg_texture']},{anc['table_texture']},"
                       f"{anc['delta_z']:.6f},"
                       f"{anc['yaw_error_deg']:.2f}\n")
    if verbose:
        print(f"  CSV saved to: {output_csv_path}")

    return {
        "total_anchors": len(anchors),
        "output_file": str(output_path),
        "output_csv": str(output_csv_path),
        "anchors": anchors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create anchor pool from human demo HDF5 dataset"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to input HDF5 file"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to output anchor pool HDF5 file (default: <input>_anchorPool.hdf5)"
    )
    parser.add_argument(
        "--peg-length", type=float, default=0.12,
        help="Peg length (default: 0.12)"
    )
    parser.add_argument(
        "--hole-height", type=float, default=0.08,
        help="Hole height (default: 0.08)"
    )
    parser.add_argument(
        "--z-diff-min", type=float, default=0.0,
        help="Minimum z_diff for anchor condition (default: 0.0)"
    )
    parser.add_argument(
        "--z-diff-max", type=float, default=0.0125,
        help="Maximum z_diff for anchor condition (default: 0.0125)"
    )
    parser.add_argument(
        "--yaw-error-threshold", type=float, default=3.5,
        help="Minimum remaining yaw error in degrees for anchor (default: 3.5)"
    )
    parser.add_argument(
        "--start-threshold", type=float, default=0.01,
        help="z_diff threshold for alignment start (default: 0.01)"
    )
    parser.add_argument(
        "--yaw-change-threshold", type=float, default=0.002,
        help="Yaw change threshold for alignment start (default: 0.002)"
    )
    parser.add_argument(
        "--yaw-rotation-diff-threshold", type=float, default=0.03,
        help="Consecutive yaw_error_deg diff threshold for rotation start filter (default: 0.03)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress detailed output"
    )

    args = parser.parse_args()

    create_anchor_pool(
        input_hdf5=args.input,
        output_hdf5=args.output,
        peg_length=args.peg_length,
        hole_height=args.hole_height,
        z_diff_min=args.z_diff_min,
        z_diff_max=args.z_diff_max,
        yaw_error_threshold_deg=args.yaw_error_threshold,
        start_threshold=args.start_threshold,
        yaw_change_threshold=args.yaw_change_threshold,
        yaw_rotation_diff_threshold=args.yaw_rotation_diff_threshold,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
