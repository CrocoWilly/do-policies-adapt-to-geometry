"""
Script to calculate the pause count for each episode in an HDF5 dataset.

Pause is defined as: consecutive 5 time-steps where ee_pos_world and ee_yaw_world 
have minimal changes (|dx|<0.001 && |dy|<0.001 && |dz|<0.001 && |dyaw|<0.001).

Usage:
    python BehaviorAnalysis_CalcPauseCount.py --input test_insert_dataset.hdf5
    python BehaviorAnalysis_CalcPauseCount.py --input test_insert_dataset.hdf5 --output pause_count.csv
    python BehaviorAnalysis_CalcPauseCount.py --input test_insert_dataset.hdf5 --add-to-hdf5
    python BehaviorAnalysis_CalcPauseCount.py --input test_insert_dataset.hdf5 --threshold 0.002 --window 3
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def calculate_pause_count(
    ee_pos_world: np.ndarray,
    ee_yaw_world: np.ndarray,
    threshold: float = 0.001,
    window: int = 5,
) -> Tuple[int, list]:
    """
    Calculate the number of pauses in an episode trajectory.
    
    A pause is defined as consecutive `window` time-steps where:
    - |dx| < threshold
    - |dy| < threshold
    - |dz| < threshold
    - |dyaw| < threshold
    
    Args:
        ee_pos_world: (T, 3) array of end-effector positions [x, y, z]
        ee_yaw_world: (T,) or (T, 1) array of end-effector yaw angles
        threshold: Maximum change threshold to be considered "no change"
        window: Number of consecutive time-steps required for a pause
    
    Returns:
        Tuple of (pause_count, list of pause start indices)
    """
    if ee_pos_world.ndim != 2 or ee_pos_world.shape[1] < 3:
        raise ValueError(f"ee_pos_world expected (T, >=3) but got {ee_pos_world.shape}")
    
    # Flatten yaw if needed
    if ee_yaw_world.ndim == 2:
        ee_yaw_world = ee_yaw_world.flatten()
    
    T = ee_pos_world.shape[0]
    if T < window + 1:
        return 0, []
    
    # Calculate changes between consecutive time-steps
    dx = np.abs(np.diff(ee_pos_world[:, 0]))  # (T-1,)
    dy = np.abs(np.diff(ee_pos_world[:, 1]))  # (T-1,)
    dz = np.abs(np.diff(ee_pos_world[:, 2]))  # (T-1,)
    dyaw = np.abs(np.diff(ee_yaw_world))      # (T-1,)
    
    # Wrap yaw difference to handle angle wraparound
    dyaw = np.minimum(dyaw, 2 * np.pi - dyaw)
    
    # Mark time-steps with minimal change
    is_stationary = (dx < threshold) & (dy < threshold) & (dz < threshold) & (dyaw < threshold)
    
    # Find consecutive stationary periods of length >= window
    pause_count = 0
    pause_starts = []
    
    i = 0
    while i <= len(is_stationary) - window:
        # Check if next `window` time-steps are all stationary
        if np.all(is_stationary[i:i + window]):
            pause_count += 1
            pause_starts.append(i)
            
            # Skip to the end of this pause period
            j = i + window
            while j < len(is_stationary) and is_stationary[j]:
                j += 1
            i = j
        else:
            i += 1
    
    return pause_count, pause_starts


def process_hdf5_file(
    input_hdf5: str,
    output_csv: Optional[str] = None,
    add_to_hdf5: bool = False,
    demo_prefix: str = "demo_",
    ee_pos_key: str = "ee_pos_world",
    ee_yaw_key: str = "ee_yaw_world",
    pause_attr_name: str = "pause_count",
    threshold: float = 0.001,
    window: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Process an HDF5 file and calculate pause count for each episode.
    
    Args:
        input_hdf5: Path to input HDF5 file
        output_csv: Optional path to save results as CSV
        add_to_hdf5: If True, add pause_count as attribute to each episode
        demo_prefix: Prefix for demo episode names
        ee_pos_key: Key for end-effector position in obs group
        ee_yaw_key: Key for end-effector yaw in obs group
        pause_attr_name: Attribute name for storing pause count
        threshold: Maximum change threshold for stationary detection
        window: Number of consecutive stationary time-steps for a pause
        verbose: Print detailed output
    
    Returns:
        Dictionary with 'episodes' (mapping episode names to pause counts) and 'summary' (statistics)
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
            print(f"Pause detection: threshold={threshold}, window={window}")
            print("-" * 60)
        
        total_pauses = 0
        
        for demo_name in demo_names:
            demo = data_grp[demo_name]
            
            # Try to find ee_pos_world and ee_yaw_world in obs group
            if "obs" not in demo:
                print(f"[WARNING] {demo_name}: No 'obs' group found, skipping")
                continue
            
            obs = demo["obs"]
            
            if ee_pos_key not in obs:
                print(f"[WARNING] {demo_name}: Missing '{ee_pos_key}' in obs, skipping")
                continue
            
            if ee_yaw_key not in obs:
                print(f"[WARNING] {demo_name}: Missing '{ee_yaw_key}' in obs, skipping")
                continue
            
            ee_pos_world = obs[ee_pos_key][()]
            ee_yaw_world = obs[ee_yaw_key][()]
            
            try:
                pause_count, pause_starts = calculate_pause_count(
                    ee_pos_world, ee_yaw_world, threshold, window
                )
                results[demo_name] = {
                    "pause_count": pause_count,
                    "pause_starts": pause_starts,
                }
                total_pauses += pause_count
                
                # Optionally add as attribute to the episode
                if add_to_hdf5:
                    demo.attrs[pause_attr_name] = pause_count
                
                if verbose:
                    T = ee_pos_world.shape[0]
                    print(f"{demo_name}: {pause_attr_name} = {pause_count} (T={T})")
                    
            except Exception as e:
                print(f"[ERROR] {demo_name}: Failed to calculate pause count: {e}")
                continue
        
        # Calculate summary statistics
        pause_counts = [r["pause_count"] for r in results.values()]
        summary = {
            "total_episodes": len(results),
            "avg_pause_count": 0.0,
            "min_pause_count": 0,
            "max_pause_count": 0,
            "total_pauses": total_pauses,
        }
        
        if results:
            summary["avg_pause_count"] = total_pauses / len(results)
            summary["min_pause_count"] = min(pause_counts)
            summary["max_pause_count"] = max(pause_counts)
        
        if verbose:
            print("-" * 60)
            print(f"Total episodes processed: {summary['total_episodes']}")
            if results:
                print(f"Average pause count: {summary['avg_pause_count']:.2f}")
                print(f"Min pause count: {summary['min_pause_count']}")
                print(f"Max pause count: {summary['max_pause_count']}")
                print(f"Total pauses: {summary['total_pauses']}")
            
            if add_to_hdf5:
                print(f"\nPause counts saved as '{pause_attr_name}' attribute in each episode")
    
    # Optionally save results to CSV
    if output_csv:
        output_path = Path(output_csv)
        with open(output_path, "w") as f:
            f.write(f"episode,{pause_attr_name}\n")
            for demo_name, data in sorted(results.items()):
                f.write(f"{demo_name},{data['pause_count']}\n")
            # Add summary row
            f.write(f"\n# Summary Statistics\n")
            f.write(f"# Threshold: {threshold}, Window: {window}\n")
            f.write(f"# Total episodes: {summary['total_episodes']}\n")
            f.write(f"# Average pause count: {summary['avg_pause_count']:.2f}\n")
            f.write(f"# Min pause count: {summary['min_pause_count']}\n")
            f.write(f"# Max pause count: {summary['max_pause_count']}\n")
            f.write(f"# Total pauses: {summary['total_pauses']}\n")
        if verbose:
            print(f"\nResults saved to: {output_path}")
    
    return {
        "episodes": results,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate pause count for each episode in an HDF5 dataset"
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
        help="Add pause_count as attribute to each episode in the HDF5 file"
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
        default="ee_pos_world",
        help="Key for end-effector position in obs group (default: 'ee_pos_world')"
    )
    parser.add_argument(
        "--ee-yaw-key",
        type=str,
        default="ee_yaw_world",
        help="Key for end-effector yaw in obs group (default: 'ee_yaw_world')"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.001,
        help="Maximum change threshold for stationary detection (default: 0.001)"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Number of consecutive stationary time-steps for a pause (default: 5)"
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
        threshold=args.threshold,
        window=args.window,
        verbose=not args.quiet,
    )
    
    return results


if __name__ == "__main__":
    main()
