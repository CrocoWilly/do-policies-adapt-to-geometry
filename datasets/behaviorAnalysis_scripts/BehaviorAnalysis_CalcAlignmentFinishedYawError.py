"""
Script to calculate the alignment final yaw error for each episode in an HDF5 dataset.

Alignment Final Yaw Error Formula:
    final_yaw_error = abs(ee_yaw_rel[end_alignment_time])

This measures the yaw error at the end of alignment phase (just before contact/insertion).

Start alignment condition: first time (z_diff <= start_threshold) OR (|ee_yaw_t - ee_yaw_{t-1}| >= yaw_change_threshold)
    - start_threshold default: 0.01
    - yaw_change_threshold default: 0.002
End alignment condition: peg_bottom_z - hole_top_z <= end_threshold (default: -0.01)

Where:
    - peg_bottom_z = peg_center_z - 0.5 * peg_length
    - hole_top_z = hole_bottom_z + hole_height
    - peg_pos_world is at peg center
    - hole_pos is at hole bottom

Usage:
    python BehaviorAnalysis_CalcAlignmentFinishedYawError.py --input test_insert_dataset.hdf5
    python BehaviorAnalysis_CalcAlignmentFinishedYawError.py --input test_insert_dataset.hdf5 --output final_yaw_error.csv
    python BehaviorAnalysis_CalcAlignmentFinishedYawError.py --input test_insert_dataset.hdf5 --peg-length 0.12 --hole-height 0.08
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def calculate_alignment_final_yaw_error(
    ee_yaw_rel: np.ndarray,
    peg_pos_world: np.ndarray,
    hole_pos: np.ndarray,
    peg_length: float = 0.12,
    hole_height: float = 0.08,
    start_threshold: float = 0.01,
    yaw_change_threshold: float = 0.002,
    end_threshold: float = -0.01,
) -> Tuple[float, float, int, int, int]:
    """
    Calculate the alignment final yaw error from end-effector yaw data.
    
    Alignment phase is defined as:
    - Start: first time (z_diff <= start_threshold) OR (|ee_yaw_t - ee_yaw_{t-1}| >= yaw_change_threshold)
    - End: peg_bottom_z - hole_top_z <= end_threshold (peg bottom entered hole)
    
    Args:
        ee_yaw_rel: (T,) or (T, 1) array of end-effector relative yaw angles
        peg_pos_world: (T, 3) array of peg world positions (peg center)
        hole_pos: (T, 3) or (3,) array of hole world positions (hole bottom)
        peg_length: Length of the peg (default: 0.12)
        hole_height: Height of the hole (default: 0.08)
        start_threshold: Threshold for alignment start based on z position (default: 0.01)
        yaw_change_threshold: Threshold for alignment start based on yaw change (default: 0.005)
        end_threshold: Threshold for alignment end / contact start detection (default: -0.01)
    
    Returns:
        Tuple of (final_yaw_error, raw_yaw_at_end, alignment_start_step, alignment_end_step, total_steps)
    
    Formula:
        final_yaw_error = abs(ee_yaw_rel[alignment_end])
    """
    # Flatten yaw if needed
    if ee_yaw_rel.ndim == 2:
        ee_yaw_rel = ee_yaw_rel.flatten()
    
    if peg_pos_world.ndim != 2 or peg_pos_world.shape[1] < 3:
        raise ValueError(f"peg_pos_world expected (T, >=3) but got {peg_pos_world.shape}")
    
    T = len(ee_yaw_rel)
    if T < 2:
        return 0.0, 0.0, -1, -1, T
    
    # Handle hole_pos: can be (T, 3) or (3,) - if static, broadcast
    if hole_pos.ndim == 1:
        hole_pos = np.tile(hole_pos[None, :], (T, 1))
    
    # Calculate peg_bottom_z and hole_top_z for each time step
    peg_center_z = peg_pos_world[:, 2]
    peg_bottom_z = peg_center_z - 0.5 * peg_length
    
    hole_bottom_z = hole_pos[:, 2]
    hole_top_z = hole_bottom_z + hole_height
    
    # Calculate relative height: peg_bottom_z - hole_top_z
    # Positive: peg bottom is above hole top
    # Negative: peg bottom is below hole top (inside hole)
    z_diff = peg_bottom_z - hole_top_z
    
    # Hybrid start alignment condition:
    # (z_diff <= start_threshold) OR (|ee_yaw_t - ee_yaw_{t-1}| >= yaw_change_threshold)
    
    # Condition 1: z position condition
    z_start_mask = z_diff <= start_threshold
    
    # Condition 2: yaw change condition
    # yaw_diff[i] corresponds to transition from step i to step i+1
    yaw_diff = np.abs(wrap_to_pi(np.diff(ee_yaw_rel)))
    # Pad yaw_diff to match length T (first element has no previous, set to 0)
    yaw_diff_padded = np.concatenate([[0], yaw_diff])
    yaw_start_mask = yaw_diff_padded >= yaw_change_threshold
    
    # Combined condition: OR
    alignment_start_mask = z_start_mask | yaw_start_mask
    alignment_start_indices = np.where(alignment_start_mask)[0]
    
    if len(alignment_start_indices) == 0:
        # No alignment phase detected
        return 0.0, 0.0, -1, -1, T
    
    alignment_start = alignment_start_indices[0]
    
    # Find alignment end: first time step (after alignment_start) where z_diff <= end_threshold
    # (peg bottom has entered the hole)
    alignment_end_mask = z_diff <= end_threshold
    alignment_end_indices = np.where(alignment_end_mask)[0]
    
    # Filter to only indices after alignment_start
    alignment_end_indices = alignment_end_indices[alignment_end_indices > alignment_start]
    
    if len(alignment_end_indices) == 0:
        # Alignment started but never reached contact/end condition
        # Use the end of episode as alignment_end
        alignment_end = T - 1
    else:
        alignment_end = alignment_end_indices[0]
    
    if alignment_start >= alignment_end:
        # No valid alignment phase
        return 0.0, 0.0, alignment_start, alignment_end, T
    
    # Get yaw at end of alignment phase
    raw_yaw_at_end = float(ee_yaw_rel[alignment_end])
    
    # Wrap to [-pi, pi] and take absolute value
    wrapped_yaw = wrap_to_pi(np.array([raw_yaw_at_end]))[0]
    final_yaw_error = abs(wrapped_yaw)
    
    return float(final_yaw_error), float(raw_yaw_at_end), int(alignment_start), int(alignment_end), T


def process_hdf5_file(
    input_hdf5: str,
    output_csv: Optional[str] = None,
    add_to_hdf5: bool = False,
    demo_prefix: str = "demo_",
    ee_yaw_key: str = "ee_yaw_rel",
    peg_pos_key: str = "peg_pos_world",
    hole_pos_key: str = "hole_pos_world",
    error_attr_name: str = "alignment_final_yaw_error",
    peg_length: float = 0.12,
    hole_height: float = 0.08,
    start_threshold: float = 0.01,
    yaw_change_threshold: float = 0.002,
    end_threshold: float = -0.01,
    verbose: bool = True,
) -> dict:
    """
    Process an HDF5 file and calculate alignment final yaw error for each episode.
    
    Args:
        input_hdf5: Path to input HDF5 file
        output_csv: Optional path to save results as CSV
        add_to_hdf5: If True, add alignment_final_yaw_error as attribute to each episode
        demo_prefix: Prefix for demo episode names
        ee_yaw_key: Key for end-effector relative yaw in obs group
        peg_pos_key: Key for peg world position in obs group
        hole_pos_key: Key for hole position in obs group
        error_attr_name: Attribute name for storing alignment final yaw error
        peg_length: Length of the peg
        hole_height: Height of the hole
        start_threshold: Threshold for alignment start based on z position
        yaw_change_threshold: Threshold for alignment start based on yaw change
        end_threshold: Threshold for alignment end / contact start detection
        verbose: Print detailed output
    
    Returns:
        Dictionary with 'episodes' (mapping episode names to error data) and 'summary' (statistics)
    """
    input_path = Path(input_hdf5)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    results = {}
    
    mode = "r+" if add_to_hdf5 else "r"
    
    with h5py.File(input_path, mode) as f:
        if "data" not in f:
            raise KeyError("HDF5 file missing 'data' group")
        
        data_grp = f["data"]
        demo_names = sorted([k for k in data_grp.keys() if k.startswith(demo_prefix)])
        
        if verbose:
            print(f"Found {len(demo_names)} episodes in {input_path.name}")
            print(f"Peg length: {peg_length}, Hole height: {hole_height}")
            print(f"Alignment start: (z_diff <= {start_threshold}) OR (yaw_change >= {yaw_change_threshold}), End: {end_threshold}")
            print("-" * 110)
        
        total_error = 0.0
        alignment_count = 0
        
        for demo_name in demo_names:
            demo = data_grp[demo_name]
            
            # Try to find required keys in obs group
            if "obs" not in demo:
                print(f"[WARNING] {demo_name}: No 'obs' group found, skipping")
                continue
            
            obs = demo["obs"]
            
            missing_keys = []
            if ee_yaw_key not in obs:
                missing_keys.append(ee_yaw_key)
            if peg_pos_key not in obs:
                missing_keys.append(peg_pos_key)
            if hole_pos_key not in obs:
                missing_keys.append(hole_pos_key)
            
            if missing_keys:
                print(f"[WARNING] {demo_name}: Missing keys in obs: {missing_keys}, skipping")
                continue
            
            ee_yaw_rel = obs[ee_yaw_key][()]
            peg_pos_world = obs[peg_pos_key][()]
            hole_pos = obs[hole_pos_key][()]
            
            try:
                final_yaw_error, raw_yaw, alignment_start, alignment_end, total_steps = calculate_alignment_final_yaw_error(
                    ee_yaw_rel, peg_pos_world, hole_pos,
                    peg_length, hole_height, start_threshold, yaw_change_threshold, end_threshold
                )
                
                alignment_steps = alignment_end - alignment_start if alignment_start >= 0 and alignment_end >= 0 else 0
                
                results[demo_name] = {
                    "alignment_final_yaw_error": final_yaw_error,
                    "raw_yaw_at_end": raw_yaw,
                    "alignment_start_step": alignment_start,
                    "alignment_end_step": alignment_end,
                    "alignment_steps": alignment_steps,
                    "total_steps": total_steps,
                }
                
                if alignment_start >= 0:
                    total_error += final_yaw_error
                    alignment_count += 1
                
                # Optionally add as attribute to the episode
                if add_to_hdf5:
                    demo.attrs[error_attr_name] = final_yaw_error
                    demo.attrs["alignment_start_step"] = alignment_start
                    demo.attrs["alignment_end_step"] = alignment_end
                
                if verbose:
                    if alignment_start >= 0:
                        print(f"{demo_name}: {error_attr_name} = {final_yaw_error:.6f} rad "
                              f"({np.degrees(final_yaw_error):.2f} deg) "
                              f"[raw={raw_yaw:.4f}] "
                              f"(start={alignment_start}, end={alignment_end}, alignment_steps={alignment_steps}, T={total_steps})")
                    else:
                        print(f"{demo_name}: No alignment phase detected (T={total_steps})")
                    
            except Exception as e:
                print(f"[ERROR] {demo_name}: Failed to calculate alignment final yaw error: {e}")
                continue
        
        # Calculate summary statistics
        error_values = [r["alignment_final_yaw_error"] for r in results.values() if r["alignment_start_step"] >= 0]
        summary = {
            "total_episodes": len(results),
            "episodes_with_alignment": alignment_count,
            "avg_final_yaw_error": 0.0,
            "min_final_yaw_error": 0.0,
            "max_final_yaw_error": 0.0,
            "total_final_yaw_error": total_error,
        }
        
        if error_values:
            summary["avg_final_yaw_error"] = total_error / len(error_values)
            summary["min_final_yaw_error"] = min(error_values)
            summary["max_final_yaw_error"] = max(error_values)
        
        if verbose:
            print("-" * 110)
            print(f"Total episodes processed: {summary['total_episodes']}")
            print(f"Episodes with alignment phase: {summary['episodes_with_alignment']}")
            if error_values:
                print(f"Average final yaw error: {summary['avg_final_yaw_error']:.6f} rad "
                      f"({np.degrees(summary['avg_final_yaw_error']):.2f} deg)")
                print(f"Min final yaw error: {summary['min_final_yaw_error']:.6f} rad "
                      f"({np.degrees(summary['min_final_yaw_error']):.2f} deg)")
                print(f"Max final yaw error: {summary['max_final_yaw_error']:.6f} rad "
                      f"({np.degrees(summary['max_final_yaw_error']):.2f} deg)")
            
            if add_to_hdf5:
                print(f"\nAlignment final yaw errors saved as '{error_attr_name}' attribute in each episode")
    
    # Optionally save results to CSV
    if output_csv:
        output_path = Path(output_csv)
        with open(output_path, "w") as f:
            f.write(f"episode,{error_attr_name}_rad,{error_attr_name}_deg,raw_yaw_at_end_rad,alignment_start_step,alignment_end_step,alignment_steps,total_steps\n")
            for demo_name, data in sorted(results.items()):
                f.write(f"{demo_name},{data['alignment_final_yaw_error']:.6f},"
                        f"{np.degrees(data['alignment_final_yaw_error']):.2f},"
                        f"{data['raw_yaw_at_end']:.6f},"
                        f"{data['alignment_start_step']},{data['alignment_end_step']},"
                        f"{data['alignment_steps']},{data['total_steps']}\n")
            # Add summary row
            f.write(f"\n# Summary Statistics\n")
            f.write(f"# Peg length: {peg_length}, Hole height: {hole_height}\n")
            f.write(f"# Alignment start: (z_diff <= {start_threshold}) OR (yaw_change >= {yaw_change_threshold}), End: {end_threshold}\n")
            f.write(f"# Total episodes: {summary['total_episodes']}\n")
            f.write(f"# Episodes with alignment phase: {summary['episodes_with_alignment']}\n")
            f.write(f"# Average final yaw error: {summary['avg_final_yaw_error']:.6f} rad "
                    f"({np.degrees(summary['avg_final_yaw_error']):.2f} deg)\n")
            f.write(f"# Min final yaw error: {summary['min_final_yaw_error']:.6f} rad "
                    f"({np.degrees(summary['min_final_yaw_error']):.2f} deg)\n")
            f.write(f"# Max final yaw error: {summary['max_final_yaw_error']:.6f} rad "
                    f"({np.degrees(summary['max_final_yaw_error']):.2f} deg)\n")
        if verbose:
            print(f"\nResults saved to: {output_path}")
    
    return {
        "episodes": results,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate alignment final yaw error for each episode in an HDF5 dataset"
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
        "--add-to-hdf5",
        action="store_true",
        help="Add alignment_final_yaw_error as attribute to each episode in the HDF5 file"
    )
    parser.add_argument(
        "--demo-prefix",
        type=str,
        default="demo_",
        help="Prefix for demo episode names (default: 'demo_')"
    )
    parser.add_argument(
        "--ee-yaw-key",
        type=str,
        default="ee_yaw_rel",
        help="Key for end-effector relative yaw in obs group (default: 'ee_yaw_rel')"
    )
    parser.add_argument(
        "--peg-pos-key",
        type=str,
        default="peg_pos_world",
        help="Key for peg world position in obs group (default: 'peg_pos_world')"
    )
    parser.add_argument(
        "--hole-pos-key",
        type=str,
        default="hole_pos_world",
        help="Key for hole position in obs group (default: 'hole_pos_world')"
    )
    parser.add_argument(
        "--peg-length",
        type=float,
        default=0.12,
        help="Length of the peg (default: 0.12)"
    )
    parser.add_argument(
        "--hole-height",
        type=float,
        default=0.08,
        help="Height of the hole (default: 0.08)"
    )
    parser.add_argument(
        "--start-threshold",
        type=float,
        default=0.01,
        help="Threshold for alignment start (z condition): peg_bottom_z - hole_top_z <= threshold (default: 0.01)"
    )
    parser.add_argument(
        "--yaw-change-threshold",
        type=float,
        default=0.002,
        help="Threshold for alignment start (yaw condition): |ee_yaw_t - ee_yaw_{t-1}| >= threshold (default: 0.002)"
    )
    parser.add_argument(
        "--end-threshold",
        type=float,
        default=-0.01,
        help="Threshold for alignment end / contact start: peg_bottom_z - hole_top_z <= threshold (default: -0.01)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress detailed output"
    )
    
    args = parser.parse_args()
    
    results = process_hdf5_file(
        input_hdf5=args.input,
        output_csv=args.output,
        add_to_hdf5=args.add_to_hdf5,
        demo_prefix=args.demo_prefix,
        ee_yaw_key=args.ee_yaw_key,
        peg_pos_key=args.peg_pos_key,
        hole_pos_key=args.hole_pos_key,
        peg_length=args.peg_length,
        hole_height=args.hole_height,
        start_threshold=args.start_threshold,
        yaw_change_threshold=args.yaw_change_threshold,
        end_threshold=args.end_threshold,
        verbose=not args.quiet,
    )
    
    return results


if __name__ == "__main__":
    main()
