# Computed from ee_pos_rel by default; pass --ee-pos-key --ee_pos_world to use world coordinates instead

"""
Script to calculate the wiggle index for each episode in an HDF5 dataset.

Wiggle Index Formula:
    wiggle = sum_t sqrt((x_t - x_{t-1})^2 + (y_t - y_{t-1})^2)

This measures the total horizontal (x-y plane) path length traversed by the end-effector,
which can indicate trajectory smoothness or oscillation behavior.

Usage:
    python PostProcess_CalcWiggleIndex.py --input test_insert_dataset.hdf5
    python PostProcess_CalcWiggleIndex.py --input test_insert_dataset.hdf5 --output results.csv
    python PostProcess_CalcWiggleIndex.py --input test_insert_dataset.hdf5 --add-to-hdf5
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Optional
import json


def calculate_wiggle_index(ee_pos_rel: np.ndarray) -> float:
    """
    Calculate the wiggle index from end-effector relative position data.
    
    Args:
        ee_pos_rel: (T, 3) array of end-effector positions [x, y, z]
    
    Returns:
        Wiggle index: sum of Euclidean distances in x-y plane between consecutive timesteps
    
    Formula:
        wiggle = sum_t sqrt((x_t - x_{t-1})^2 + (y_t - y_{t-1})^2)
    """
    if ee_pos_rel.ndim != 2 or ee_pos_rel.shape[1] < 2:
        raise ValueError(f"ee_pos_rel expected (T, >=2) but got {ee_pos_rel.shape}")
    
    T = ee_pos_rel.shape[0]
    if T < 2:
        return 0.0
    
    # Extract x and y positions
    x = ee_pos_rel[:, 0]
    y = ee_pos_rel[:, 1]
    
    # Calculate differences between consecutive timesteps
    dx = np.diff(x)  # (T-1,)
    dy = np.diff(y)  # (T-1,)
    
    # Calculate Euclidean distance in x-y plane for each step
    distances = np.sqrt(dx**2 + dy**2)
    
    # Sum all distances to get wiggle index
    wiggle = np.sum(distances)
    
    return float(wiggle)


def process_hdf5_file(
    input_hdf5: str,
    output_csv: Optional[str] = None,
    add_to_hdf5: bool = False,
    demo_prefix: str = "demo_",
    ee_pos_key: str = "ee_pos_rel",
    wiggle_attr_name: str = "wiggle_index",
    verbose: bool = True,
) -> dict:
    """
    Process an HDF5 file and calculate wiggle index for each episode.
    
    Args:
        input_hdf5: Path to input HDF5 file
        output_csv: Optional path to save results as CSV
        add_to_hdf5: If True, add wiggle_index as attribute to each episode
        demo_prefix: Prefix for demo episode names
        ee_pos_key: Key for end-effector position in obs group
        wiggle_attr_name: Attribute name for storing wiggle index
        verbose: Print detailed output
    
    Returns:
        Dictionary with 'episodes' (mapping episode names to wiggle indices) and 'summary' (statistics)
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
            print("-" * 60)
        
        total_wiggle = 0.0
        
        for demo_name in demo_names:
            demo = data_grp[demo_name]
            
            # Try to find ee_pos_rel in obs group
            if "obs" not in demo:
                print(f"[WARNING] {demo_name}: No 'obs' group found, skipping")
                continue
            
            obs = demo["obs"]
            
            if ee_pos_key not in obs:
                print(f"[WARNING] {demo_name}: Missing '{ee_pos_key}' in obs, skipping")
                continue
            
            ee_pos_rel = obs[ee_pos_key][()]
            
            try:
                wiggle = calculate_wiggle_index(ee_pos_rel)
                results[demo_name] = wiggle
                total_wiggle += wiggle
                
                # Optionally add as attribute to the episode
                if add_to_hdf5:
                    demo.attrs[wiggle_attr_name] = wiggle
                
                if verbose:
                    T = ee_pos_rel.shape[0]
                    print(f"{demo_name}: wiggle_index = {wiggle:.6f} (T={T})")
                    
            except Exception as e:
                print(f"[ERROR] {demo_name}: Failed to calculate wiggle index: {e}")
                continue
        
        # Calculate summary statistics
        summary = {
            "total_episodes": len(results),
            "avg_wiggle": 0.0,
            "min_wiggle": 0.0,
            "max_wiggle": 0.0,
            "total_wiggle": total_wiggle,
        }
        
        if results:
            summary["avg_wiggle"] = total_wiggle / len(results)
            summary["min_wiggle"] = min(results.values())
            summary["max_wiggle"] = max(results.values())
        
        if verbose:
            print("-" * 60)
            print(f"Total episodes processed: {summary['total_episodes']}")
            if results:
                print(f"Average wiggle index: {summary['avg_wiggle']:.6f}")
                print(f"Min wiggle index: {summary['min_wiggle']:.6f}")
                print(f"Max wiggle index: {summary['max_wiggle']:.6f}")
            
            if add_to_hdf5:
                print(f"\nWiggle indices saved as '{wiggle_attr_name}' attribute in each episode")
    
    # Optionally save results to CSV
    if output_csv:
        output_path = Path(output_csv)
        with open(output_path, "w") as f:
            f.write("episode,wiggle_index\n")
            for demo_name, wiggle in sorted(results.items()):
                f.write(f"{demo_name},{wiggle:.6f}\n")
            # Add summary row
            f.write(f"\n# Summary Statistics\n")
            f.write(f"# Total episodes: {summary['total_episodes']}\n")
            f.write(f"# Average wiggle index: {summary['avg_wiggle']:.6f}\n")
            f.write(f"# Min wiggle index: {summary['min_wiggle']:.6f}\n")
            f.write(f"# Max wiggle index: {summary['max_wiggle']:.6f}\n")
        if verbose:
            print(f"\nResults saved to: {output_path}")
    
    return {
        "episodes": results,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate wiggle index for each episode in an HDF5 dataset"
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
        help="Add wiggle_index as attribute to each episode in the HDF5 file"
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
        help="Key for end-effector position in obs group (default: 'ee_pos_rel')"
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
