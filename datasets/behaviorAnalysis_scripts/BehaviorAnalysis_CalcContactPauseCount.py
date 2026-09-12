"""
Script to calculate the contact pause count for each episode in an HDF5 dataset.

Contact Pause Count Formula:
    Count the number of pause events during contact phase (from start_contact to end of episode).
    A pause is detected when ee_pos_world (x, y, z) AND ee_yaw_world all have changes < threshold
    for a consecutive window of time steps.

Start contact condition: peg_bottom_z - hole_top_z <= contact_threshold (default: -0.003)
    (peg bottom has entered the hole)

Pause condition (consecutive window_size steps):
    - abs(x_t - x_{t-1}) < pos_threshold (default: 0.001)
    - abs(y_t - y_{t-1}) < pos_threshold (default: 0.001)
    - abs(z_t - z_{t-1}) < pos_threshold (default: 0.001)
    - abs(yaw_t - yaw_{t-1}) < yaw_threshold (default: 0.001)

Where:
    - peg_bottom_z = peg_center_z - 0.5 * peg_length
    - hole_top_z = hole_bottom_z + hole_height
    - peg_pos_world is at peg center
    - hole_pos is at hole bottom

Usage:
    python BehaviorAnalysis_CalcContactPauseCount.py --input test_insert_dataset.hdf5
    python BehaviorAnalysis_CalcContactPauseCount.py --input test_insert_dataset.hdf5 --output pause_count.csv
    python BehaviorAnalysis_CalcContactPauseCount.py --input test_insert_dataset.hdf5 --window-size 5 --pos-threshold 0.001
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def calculate_contact_pause_count(
    ee_pos_world: np.ndarray,
    ee_yaw_world: np.ndarray,
    peg_pos_world: np.ndarray,
    hole_pos: np.ndarray,
    peg_length: float = 0.12,
    hole_height: float = 0.08,
    contact_threshold: float = -0.003,
    window_size: int = 3,
    pos_threshold: float = 0.001,
    yaw_threshold: float = 0.001,
) -> Tuple[int, int, int, List[int]]:
    """
    Calculate the contact pause count from end-effector position and yaw data.
    
    Contact phase starts when peg bottom enters hole:
    - Start: peg_bottom_z - hole_top_z <= contact_threshold
    
    A pause is detected when all changes are below thresholds for window_size consecutive steps.
    
    Args:
        ee_pos_world: (T, 3) array of end-effector world positions [x, y, z]
        ee_yaw_world: (T,) or (T, 1) array of end-effector yaw angles
        peg_pos_world: (T, 3) array of peg world positions (peg center)
        hole_pos: (T, 3) or (3,) array of hole world positions (hole bottom)
        peg_length: Length of the peg (default: 0.12)
        hole_height: Height of the hole (default: 0.08)
        contact_threshold: Threshold for contact start detection (default: -0.003)
        window_size: Number of consecutive steps to detect a pause (default: 3)
        pos_threshold: Threshold for position change detection (default: 0.001)
        yaw_threshold: Threshold for yaw change detection (default: 0.001)
    
    Returns:
        Tuple of (pause_count, contact_start_step, total_steps, pause_start_indices)
    """
    if ee_pos_world.ndim != 2 or ee_pos_world.shape[1] < 3:
        raise ValueError(f"ee_pos_world expected (T, >=3) but got {ee_pos_world.shape}")
    if peg_pos_world.ndim != 2 or peg_pos_world.shape[1] < 3:
        raise ValueError(f"peg_pos_world expected (T, >=3) but got {peg_pos_world.shape}")
    
    # Flatten yaw if needed
    if ee_yaw_world.ndim == 2:
        ee_yaw_world = ee_yaw_world.flatten()
    
    T = ee_pos_world.shape[0]
    if T < window_size + 1:
        return 0, -1, T, []
    
    # Handle hole_pos: can be (T, 3) or (3,) - if static, broadcast
    if hole_pos.ndim == 1:
        hole_pos = np.tile(hole_pos[None, :], (T, 1))
    
    # Calculate peg_bottom_z and hole_top_z for each time step
    peg_center_z = peg_pos_world[:, 2]
    peg_bottom_z = peg_center_z - 0.5 * peg_length
    
    hole_bottom_z = hole_pos[:, 2]
    hole_top_z = hole_bottom_z + hole_height
    
    # Calculate relative height: peg_bottom_z - hole_top_z
    z_diff = peg_bottom_z - hole_top_z
    
    # Find contact start: first time step where z_diff <= contact_threshold
    contact_start_mask = z_diff <= contact_threshold
    contact_start_indices = np.where(contact_start_mask)[0]
    
    if len(contact_start_indices) == 0:
        # No contact phase detected
        return 0, -1, T, []
    
    contact_start = contact_start_indices[0]
    
    # Calculate changes for each time step (from contact_start to end)
    # We need at least window_size steps after contact_start
    if contact_start + window_size >= T:
        return 0, int(contact_start), T, []
    
    # Extract data from contact_start to end
    pos_data = ee_pos_world[contact_start:]
    yaw_data = ee_yaw_world[contact_start:]
    
    # Calculate step-by-step changes
    dx = np.abs(np.diff(pos_data[:, 0]))
    dy = np.abs(np.diff(pos_data[:, 1]))
    dz = np.abs(np.diff(pos_data[:, 2]))
    dyaw = np.abs(wrap_to_pi(np.diff(yaw_data)))
    
    # Check if each step is a "still" step (all changes below threshold)
    is_still = (dx < pos_threshold) & (dy < pos_threshold) & (dz < pos_threshold) & (dyaw < yaw_threshold)
    
    # Find consecutive windows of "still" steps
    # A pause is detected when we have window_size consecutive "still" steps
    pause_count = 0
    pause_start_indices = []
    
    i = 0
    while i <= len(is_still) - window_size:
        # Check if the next window_size steps are all "still"
        if np.all(is_still[i:i + window_size]):
            # Found a pause
            pause_count += 1
            pause_start_indices.append(contact_start + i)
            # Skip to the end of this pause window to avoid counting overlapping pauses
            # Find where the pause ends
            end_of_pause = i + window_size
            while end_of_pause < len(is_still) and is_still[end_of_pause]:
                end_of_pause += 1
            i = end_of_pause
        else:
            i += 1
    
    return pause_count, int(contact_start), T, pause_start_indices


def process_hdf5_file(
    input_hdf5: str,
    output_csv: Optional[str] = None,
    add_to_hdf5: bool = False,
    demo_prefix: str = "demo_",
    ee_pos_key: str = "ee_pos_rel",
    ee_yaw_key: str = "ee_yaw_rel",
    peg_pos_key: str = "peg_pos_world",
    hole_pos_key: str = "hole_pos_world",
    pause_attr_name: str = "contact_pause_count",
    peg_length: float = 0.12,
    hole_height: float = 0.08,
    contact_threshold: float = -0.003,
    window_size: int = 3,
    pos_threshold: float = 0.001,
    yaw_threshold: float = 0.001,
    verbose: bool = True,
) -> dict:
    """
    Process an HDF5 file and calculate contact pause count for each episode.
    
    Args:
        input_hdf5: Path to input HDF5 file
        output_csv: Optional path to save results as CSV
        add_to_hdf5: If True, add contact_pause_count as attribute to each episode
        demo_prefix: Prefix for demo episode names
        ee_pos_key: Key for end-effector world position in obs group
        ee_yaw_key: Key for end-effector yaw in obs group
        peg_pos_key: Key for peg world position in obs group
        hole_pos_key: Key for hole position in obs group
        pause_attr_name: Attribute name for storing contact pause count
        peg_length: Length of the peg
        hole_height: Height of the hole
        contact_threshold: Threshold for contact start detection
        window_size: Number of consecutive steps to detect a pause
        pos_threshold: Threshold for position change detection
        yaw_threshold: Threshold for yaw change detection
        verbose: Print detailed output
    
    Returns:
        Dictionary with 'episodes' (mapping episode names to pause data) and 'summary' (statistics)
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
            print(f"Contact threshold: {contact_threshold}")
            print(f"Window size: {window_size}, Pos threshold: {pos_threshold}, Yaw threshold: {yaw_threshold}")
            print("-" * 100)
        
        total_pause_count = 0
        contact_count = 0
        
        for demo_name in demo_names:
            demo = data_grp[demo_name]
            
            # Try to find required keys in obs group
            if "obs" not in demo:
                print(f"[WARNING] {demo_name}: No 'obs' group found, skipping")
                continue
            
            obs = demo["obs"]
            
            missing_keys = []
            if ee_pos_key not in obs:
                missing_keys.append(ee_pos_key)
            if ee_yaw_key not in obs:
                missing_keys.append(ee_yaw_key)
            if peg_pos_key not in obs:
                missing_keys.append(peg_pos_key)
            if hole_pos_key not in obs:
                missing_keys.append(hole_pos_key)
            
            if missing_keys:
                print(f"[WARNING] {demo_name}: Missing keys in obs: {missing_keys}, skipping")
                continue
            
            ee_pos_world = obs[ee_pos_key][()]
            ee_yaw_world = obs[ee_yaw_key][()]
            peg_pos_world = obs[peg_pos_key][()]
            hole_pos = obs[hole_pos_key][()]
            
            try:
                pause_count, contact_start, total_steps, pause_indices = calculate_contact_pause_count(
                    ee_pos_world, ee_yaw_world, peg_pos_world, hole_pos,
                    peg_length, hole_height, contact_threshold,
                    window_size, pos_threshold, yaw_threshold
                )
                
                contact_steps = total_steps - contact_start if contact_start >= 0 else 0
                
                results[demo_name] = {
                    "contact_pause_count": pause_count,
                    "contact_start_step": contact_start,
                    "contact_steps": contact_steps,
                    "total_steps": total_steps,
                    "pause_start_indices": pause_indices,
                }
                
                if contact_start >= 0:
                    total_pause_count += pause_count
                    contact_count += 1
                
                # Optionally add as attribute to the episode
                if add_to_hdf5:
                    demo.attrs[pause_attr_name] = pause_count
                    demo.attrs["contact_start_step"] = contact_start
                
                if verbose:
                    if contact_start >= 0:
                        pause_info = f"pauses at steps: {pause_indices}" if pause_indices else "no pauses"
                        print(f"{demo_name}: {pause_attr_name} = {pause_count} "
                              f"(contact_start={contact_start}, contact_steps={contact_steps}, T={total_steps}) "
                              f"[{pause_info}]")
                    else:
                        print(f"{demo_name}: No contact phase detected (T={total_steps})")
                    
            except Exception as e:
                print(f"[ERROR] {demo_name}: Failed to calculate contact pause count: {e}")
                continue
        
        # Calculate summary statistics
        pause_values = [r["contact_pause_count"] for r in results.values() if r["contact_start_step"] >= 0]
        summary = {
            "total_episodes": len(results),
            "episodes_with_contact": contact_count,
            "avg_pause_count": 0.0,
            "min_pause_count": 0,
            "max_pause_count": 0,
            "total_pause_count": total_pause_count,
            "episodes_with_pauses": sum(1 for v in pause_values if v > 0),
        }
        
        if pause_values:
            summary["avg_pause_count"] = total_pause_count / len(pause_values)
            summary["min_pause_count"] = min(pause_values)
            summary["max_pause_count"] = max(pause_values)
        
        if verbose:
            print("-" * 100)
            print(f"Total episodes processed: {summary['total_episodes']}")
            print(f"Episodes with contact phase: {summary['episodes_with_contact']}")
            print(f"Episodes with pauses: {summary['episodes_with_pauses']}")
            if pause_values:
                print(f"Average pause count: {summary['avg_pause_count']:.2f}")
                print(f"Min pause count: {summary['min_pause_count']}")
                print(f"Max pause count: {summary['max_pause_count']}")
                print(f"Total pauses across all episodes: {summary['total_pause_count']}")
            
            if add_to_hdf5:
                print(f"\nContact pause counts saved as '{pause_attr_name}' attribute in each episode")
    
    # Optionally save results to CSV
    if output_csv:
        output_path = Path(output_csv)
        with open(output_path, "w") as f:
            f.write(f"episode,{pause_attr_name},contact_start_step,contact_steps,total_steps,pause_start_indices\n")
            for demo_name, data in sorted(results.items()):
                indices_str = ";".join(map(str, data["pause_start_indices"]))
                f.write(f"{demo_name},{data['contact_pause_count']},"
                        f"{data['contact_start_step']},{data['contact_steps']},"
                        f"{data['total_steps']},\"{indices_str}\"\n")
            # Add summary row
            f.write(f"\n# Summary Statistics\n")
            f.write(f"# Peg length: {peg_length}, Hole height: {hole_height}\n")
            f.write(f"# Contact threshold: {contact_threshold}\n")
            f.write(f"# Window size: {window_size}, Pos threshold: {pos_threshold}, Yaw threshold: {yaw_threshold}\n")
            f.write(f"# Total episodes: {summary['total_episodes']}\n")
            f.write(f"# Episodes with contact phase: {summary['episodes_with_contact']}\n")
            f.write(f"# Episodes with pauses: {summary['episodes_with_pauses']}\n")
            f.write(f"# Average pause count: {summary['avg_pause_count']:.2f}\n")
            f.write(f"# Min pause count: {summary['min_pause_count']}\n")
            f.write(f"# Max pause count: {summary['max_pause_count']}\n")
            f.write(f"# Total pauses: {summary['total_pause_count']}\n")
        if verbose:
            print(f"\nResults saved to: {output_path}")
    
    return {
        "episodes": results,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate contact pause count for each episode in an HDF5 dataset"
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
        help="Add contact_pause_count as attribute to each episode in the HDF5 file"
    )
    parser.add_argument(
        "--demo-prefix",
        type=str,
        default="demo_",
        help="Prefix for demo episode names (default: 'demo_')"
    )
    parser.add_argument(
        "--ee-pos-key",
        type=str,
        default="ee_pos_rel",
        help="Key for end-effector world position in obs group (default: 'ee_pos_rel')"
    )
    parser.add_argument(
        "--ee-yaw-key",
        type=str,
        default="ee_yaw_rel",
        help="Key for end-effector yaw in obs group (default: 'ee_yaw_rel')"
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
        "--contact-threshold",
        type=float,
        default=-0.003,
        help="Threshold for contact start: peg_bottom_z - hole_top_z <= threshold (default: -0.003)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="Number of consecutive steps to detect a pause (default: 3)"
    )
    parser.add_argument(
        "--pos-threshold",
        type=float,
        default=0.001,
        help="Threshold for position change detection (default: 0.001)"
    )
    parser.add_argument(
        "--yaw-threshold",
        type=float,
        default=0.001,
        help="Threshold for yaw change detection (default: 0.001)"
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
        ee_pos_key=args.ee_pos_key,
        ee_yaw_key=args.ee_yaw_key,
        peg_pos_key=args.peg_pos_key,
        hole_pos_key=args.hole_pos_key,
        peg_length=args.peg_length,
        hole_height=args.hole_height,
        contact_threshold=args.contact_threshold,
        window_size=args.window_size,
        pos_threshold=args.pos_threshold,
        yaw_threshold=args.yaw_threshold,
        verbose=not args.quiet,
    )
    
    return results


if __name__ == "__main__":
    main()
