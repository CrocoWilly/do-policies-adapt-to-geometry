"""
Script to calculate the mean push speed for each episode in an HDF5 dataset.

Mean Push Speed Formula:
    mean_push_speed = abs(ee_pos_world_z[last] - ee_pos_world_z[first]) / completion_step

This measures the average vertical (z-axis) speed of the end-effector during the episode,
which can indicate pushing/insertion efficiency.

Usage:
    python BehaviorAnalysis_CalcMeanPushSpeed.py --input test_insert_dataset.hdf5
    python BehaviorAnalysis_CalcMeanPushSpeed.py --input test_insert_dataset.hdf5 --output mean_push_speed.csv
    python BehaviorAnalysis_CalcMeanPushSpeed.py --input test_insert_dataset.hdf5 --add-to-hdf5
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def calculate_mean_push_speed(ee_pos_world: np.ndarray) -> Tuple[float, float, int]:
    """
    Calculate the mean push speed from end-effector world position data.
    
    Args:
        ee_pos_world: (T, 3) array of end-effector positions [x, y, z]
    
    Returns:
        Tuple of (mean_push_speed, z_displacement, completion_step)
    
    Formula:
        mean_push_speed = abs(z_last - z_first) / T
    """
    if ee_pos_world.ndim != 2 or ee_pos_world.shape[1] < 3:
        raise ValueError(f"ee_pos_world expected (T, >=3) but got {ee_pos_world.shape}")
    
    T = ee_pos_world.shape[0]
    if T < 1:
        return 0.0, 0.0, 0
    
    # Get z position at first and last time-step
    z_first = ee_pos_world[0, 2]
    z_last = ee_pos_world[-1, 2]
    
    # Calculate z displacement
    z_displacement = abs(z_last - z_first)
    
    # Calculate mean push speed
    mean_push_speed = z_displacement / T
    
    return float(mean_push_speed), float(z_displacement), T


def process_hdf5_file(
    input_hdf5: str,
    output_csv: Optional[str] = None,
    add_to_hdf5: bool = False,
    demo_prefix: str = "demo_",
    ee_pos_key: str = "ee_pos_world",
    speed_attr_name: str = "mean_push_speed",
    verbose: bool = True,
) -> dict:
    """
    Process an HDF5 file and calculate mean push speed for each episode.
    
    Args:
        input_hdf5: Path to input HDF5 file
        output_csv: Optional path to save results as CSV
        add_to_hdf5: If True, add mean_push_speed as attribute to each episode
        demo_prefix: Prefix for demo episode names
        ee_pos_key: Key for end-effector position in obs group
        speed_attr_name: Attribute name for storing mean push speed
        verbose: Print detailed output
    
    Returns:
        Dictionary with 'episodes' (mapping episode names to speed data) and 'summary' (statistics)
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
            print("-" * 70)
        
        total_speed = 0.0
        
        for demo_name in demo_names:
            demo = data_grp[demo_name]
            
            # Try to find ee_pos_world in obs group
            if "obs" not in demo:
                print(f"[WARNING] {demo_name}: No 'obs' group found, skipping")
                continue
            
            obs = demo["obs"]
            
            if ee_pos_key not in obs:
                print(f"[WARNING] {demo_name}: Missing '{ee_pos_key}' in obs, skipping")
                continue
            
            ee_pos_world = obs[ee_pos_key][()]
            
            try:
                mean_push_speed, z_displacement, completion_step = calculate_mean_push_speed(ee_pos_world)
                results[demo_name] = {
                    "mean_push_speed": mean_push_speed,
                    "z_displacement": z_displacement,
                    "completion_step": completion_step,
                }
                total_speed += mean_push_speed
                
                # Optionally add as attribute to the episode
                if add_to_hdf5:
                    demo.attrs[speed_attr_name] = mean_push_speed
                
                if verbose:
                    print(f"{demo_name}: {speed_attr_name} = {mean_push_speed:.6f} "
                          f"(z_displacement={z_displacement:.4f}, T={completion_step})")
                    
            except Exception as e:
                print(f"[ERROR] {demo_name}: Failed to calculate mean push speed: {e}")
                continue
        
        # Calculate summary statistics
        speeds = [r["mean_push_speed"] for r in results.values()]
        summary = {
            "total_episodes": len(results),
            "avg_mean_push_speed": 0.0,
            "min_mean_push_speed": 0.0,
            "max_mean_push_speed": 0.0,
            "total_speed": total_speed,
        }
        
        if results:
            summary["avg_mean_push_speed"] = total_speed / len(results)
            summary["min_mean_push_speed"] = min(speeds)
            summary["max_mean_push_speed"] = max(speeds)
        
        if verbose:
            print("-" * 70)
            print(f"Total episodes processed: {summary['total_episodes']}")
            if results:
                print(f"Average mean push speed: {summary['avg_mean_push_speed']:.6f}")
                print(f"Min mean push speed: {summary['min_mean_push_speed']:.6f}")
                print(f"Max mean push speed: {summary['max_mean_push_speed']:.6f}")
            
            if add_to_hdf5:
                print(f"\nMean push speeds saved as '{speed_attr_name}' attribute in each episode")
    
    # Optionally save results to CSV
    if output_csv:
        output_path = Path(output_csv)
        with open(output_path, "w") as f:
            f.write(f"episode,{speed_attr_name},z_displacement,completion_step\n")
            for demo_name, data in sorted(results.items()):
                f.write(f"{demo_name},{data['mean_push_speed']:.6f},"
                        f"{data['z_displacement']:.6f},{data['completion_step']}\n")
            # Add summary row
            f.write(f"\n# Summary Statistics\n")
            f.write(f"# Total episodes: {summary['total_episodes']}\n")
            f.write(f"# Average mean push speed: {summary['avg_mean_push_speed']:.6f}\n")
            f.write(f"# Min mean push speed: {summary['min_mean_push_speed']:.6f}\n")
            f.write(f"# Max mean push speed: {summary['max_mean_push_speed']:.6f}\n")
        if verbose:
            print(f"\nResults saved to: {output_path}")
    
    return {
        "episodes": results,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate mean push speed for each episode in an HDF5 dataset"
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
        help="Add mean_push_speed as attribute to each episode in the HDF5 file"
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
        verbose=not args.quiet,
    )
    
    return results


if __name__ == "__main__":
    main()
