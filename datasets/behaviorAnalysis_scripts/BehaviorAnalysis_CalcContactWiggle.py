"""
Script to calculate the contact wiggle index for each episode in an HDF5 dataset.

Contact Wiggle Index Formula:
    contact_wiggle = sum_t[from contact_start to end] sqrt((x_t - x_{t-1})^2 + (y_t - y_{t-1})^2)

Contact starts when: peg_bottom_z - hole_top_z <= contact_threshold (default: -0.003)

Where:
    - peg_bottom_z = peg_center_z - 0.5 * peg_length
    - hole_top_z = hole_bottom_z + hole_height
    - peg_pos_world is at peg center
    - hole_pos is at hole bottom

Usage:
    python BehaviorAnalysis_CalcContactWiggle.py --input test_insert_dataset.hdf5
    python BehaviorAnalysis_CalcContactWiggle.py --input test_insert_dataset.hdf5 --output contact_wiggle.csv
    python BehaviorAnalysis_CalcContactWiggle.py --input test_insert_dataset.hdf5 --peg-length 0.12 --hole-height 0.08
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def calculate_contact_wiggle_index(
    ee_pos_rel: np.ndarray,
    peg_pos_world: np.ndarray,
    hole_pos: np.ndarray,
    peg_length: float = 0.12,
    hole_height: float = 0.08,
    contact_threshold: float = -0.003,
) -> Tuple[float, int, int]:
    """
    Calculate the contact wiggle index from end-effector position data.
    
    Only starts counting when peg bottom approaches hole top:
    peg_bottom_z - hole_top_z <= contact_threshold
    
    Args:
        ee_pos_rel: (T, 3) array of end-effector relative positions [x, y, z]
        peg_pos_world: (T, 3) array of peg world positions (peg center)
        hole_pos: (T, 3) or (3,) array of hole world positions (hole bottom)
        peg_length: Length of the peg (default: 0.12)
        hole_height: Height of the hole (default: 0.08)
        contact_threshold: Threshold for contact detection (default: -0.003)
    
    Returns:
        Tuple of (contact_wiggle_index, contact_start_step, total_steps)
    
    Formula:
        contact_wiggle = sum_t[contact_start:end] sqrt((x_t - x_{t-1})^2 + (y_t - y_{t-1})^2)
    """
    if ee_pos_rel.ndim != 2 or ee_pos_rel.shape[1] < 2:
        raise ValueError(f"ee_pos_rel expected (T, >=2) but got {ee_pos_rel.shape}")
    if peg_pos_world.ndim != 2 or peg_pos_world.shape[1] < 3:
        raise ValueError(f"peg_pos_world expected (T, >=3) but got {peg_pos_world.shape}")
    
    T = ee_pos_rel.shape[0]
    if T < 2:
        return 0.0, -1, T
    
    # Handle hole_pos: can be (T, 3) or (3,) - if static, broadcast
    if hole_pos.ndim == 1:
        hole_pos = np.tile(hole_pos[None, :], (T, 1))
    
    # Calculate peg_bottom_z and hole_top_z for each time step
    peg_center_z = peg_pos_world[:, 2]
    peg_bottom_z = peg_center_z - 0.5 * peg_length
    
    hole_bottom_z = hole_pos[:, 2]
    hole_top_z = hole_bottom_z + hole_height
    
    # Find the first time step where contact condition is met
    contact_dist = peg_bottom_z - hole_top_z
    contact_mask = contact_dist <= contact_threshold
    
    contact_indices = np.where(contact_mask)[0]
    if len(contact_indices) == 0:
        # No contact detected
        return 0.0, -1, T
    
    contact_start = contact_indices[0]
    
    if contact_start >= T - 1:
        # Contact at last step, no wiggle to calculate
        return 0.0, contact_start, T
    
    # Calculate wiggle from contact_start to end
    x = ee_pos_rel[contact_start:, 0]
    y = ee_pos_rel[contact_start:, 1]
    
    # Calculate differences between consecutive timesteps
    dx = np.diff(x)
    dy = np.diff(y)
    
    # Calculate Euclidean distance in x-y plane for each step
    distances = np.sqrt(dx**2 + dy**2)
    
    # Sum all distances to get contact wiggle index
    contact_wiggle = np.sum(distances)
    
    return float(contact_wiggle), int(contact_start), T


def process_hdf5_file(
    input_hdf5: str,
    output_csv: Optional[str] = None,
    add_to_hdf5: bool = False,
    demo_prefix: str = "demo_",
    ee_pos_key: str = "ee_pos_rel",
    peg_pos_key: str = "peg_pos_world",
    hole_pos_key: str = "hole_pos_world",
    wiggle_attr_name: str = "contact_wiggle_index",
    peg_length: float = 0.12,
    hole_height: float = 0.08,
    contact_threshold: float = -0.003,
    verbose: bool = True,
) -> dict:
    """
    Process an HDF5 file and calculate contact wiggle index for each episode.
    
    Args:
        input_hdf5: Path to input HDF5 file
        output_csv: Optional path to save results as CSV
        add_to_hdf5: If True, add contact_wiggle_index as attribute to each episode
        demo_prefix: Prefix for demo episode names
        ee_pos_key: Key for end-effector relative position in obs group
        peg_pos_key: Key for peg world position in obs group
        hole_pos_key: Key for hole position in obs group
        wiggle_attr_name: Attribute name for storing contact wiggle index
        peg_length: Length of the peg
        hole_height: Height of the hole
        contact_threshold: Threshold for contact detection
        verbose: Print detailed output
    
    Returns:
        Dictionary with 'episodes' (mapping episode names to wiggle data) and 'summary' (statistics)
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
            print(f"Peg length: {peg_length}, Hole height: {hole_height}, Contact threshold: {contact_threshold}")
            print("-" * 80)
        
        total_wiggle = 0.0
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
            if peg_pos_key not in obs:
                missing_keys.append(peg_pos_key)
            if hole_pos_key not in obs:
                missing_keys.append(hole_pos_key)
            
            if missing_keys:
                print(f"[WARNING] {demo_name}: Missing keys in obs: {missing_keys}, skipping")
                continue
            
            ee_pos_rel = obs[ee_pos_key][()]
            peg_pos_world = obs[peg_pos_key][()]
            hole_pos = obs[hole_pos_key][()]
            
            try:
                contact_wiggle, contact_start, total_steps = calculate_contact_wiggle_index(
                    ee_pos_rel, peg_pos_world, hole_pos,
                    peg_length, hole_height, contact_threshold
                )
                
                contact_steps = total_steps - contact_start if contact_start >= 0 else 0
                contact_wiggle_per_step = contact_wiggle / contact_steps if contact_steps > 0 else 0.0
                
                results[demo_name] = {
                    "contact_wiggle_index": contact_wiggle,
                    "contact_wiggle_per_step": contact_wiggle_per_step,
                    "contact_start_step": contact_start,
                    "total_steps": total_steps,
                    "contact_steps": contact_steps,
                }
                
                if contact_start >= 0:
                    total_wiggle += contact_wiggle
                    contact_count += 1
                
                # Optionally add as attribute to the episode
                if add_to_hdf5:
                    demo.attrs[wiggle_attr_name] = contact_wiggle
                    demo.attrs["contact_wiggle_per_step"] = contact_wiggle_per_step
                    demo.attrs["contact_start_step"] = contact_start
                
                if verbose:
                    if contact_start >= 0:
                        print(f"{demo_name}: {wiggle_attr_name} = {contact_wiggle:.6f}, per_step = {contact_wiggle_per_step:.6f} "
                              f"(contact_start={contact_start}, contact_steps={contact_steps}, T={total_steps})")
                    else:
                        print(f"{demo_name}: No contact detected (T={total_steps})")
                    
            except Exception as e:
                print(f"[ERROR] {demo_name}: Failed to calculate contact wiggle index: {e}")
                continue
        
        # Calculate summary statistics
        wiggle_values = [r["contact_wiggle_index"] for r in results.values() if r["contact_start_step"] >= 0]
        summary = {
            "total_episodes": len(results),
            "episodes_with_contact": contact_count,
            "avg_contact_wiggle": 0.0,
            "min_contact_wiggle": 0.0,
            "max_contact_wiggle": 0.0,
            "total_contact_wiggle": total_wiggle,
        }
        
        if wiggle_values:
            summary["avg_contact_wiggle"] = total_wiggle / len(wiggle_values)
            summary["min_contact_wiggle"] = min(wiggle_values)
            summary["max_contact_wiggle"] = max(wiggle_values)
        
        if verbose:
            print("-" * 80)
            print(f"Total episodes processed: {summary['total_episodes']}")
            print(f"Episodes with contact: {summary['episodes_with_contact']}")
            if wiggle_values:
                print(f"Average contact wiggle index: {summary['avg_contact_wiggle']:.6f}")
                print(f"Min contact wiggle index: {summary['min_contact_wiggle']:.6f}")
                print(f"Max contact wiggle index: {summary['max_contact_wiggle']:.6f}")
            
            if add_to_hdf5:
                print(f"\nContact wiggle indices saved as '{wiggle_attr_name}' attribute in each episode")
    
    # Optionally save results to CSV
    if output_csv:
        output_path = Path(output_csv)
        with open(output_path, "w") as f:
            f.write(f"episode,{wiggle_attr_name},contact_wiggle_per_step,contact_start_step,contact_steps,total_steps\n")
            for demo_name, data in sorted(results.items()):
                f.write(f"{demo_name},{data['contact_wiggle_index']:.6f},"
                        f"{data['contact_wiggle_per_step']:.6f},{data['contact_start_step']},{data['contact_steps']},{data['total_steps']}\n")
            # Add summary row
            f.write(f"\n# Summary Statistics\n")
            f.write(f"# Peg length: {peg_length}, Hole height: {hole_height}, Contact threshold: {contact_threshold}\n")
            f.write(f"# Total episodes: {summary['total_episodes']}\n")
            f.write(f"# Episodes with contact: {summary['episodes_with_contact']}\n")
            f.write(f"# Average contact wiggle index: {summary['avg_contact_wiggle']:.6f}\n")
            f.write(f"# Min contact wiggle index: {summary['min_contact_wiggle']:.6f}\n")
            f.write(f"# Max contact wiggle index: {summary['max_contact_wiggle']:.6f}\n")
        if verbose:
            print(f"\nResults saved to: {output_path}")
    
    return {
        "episodes": results,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate contact wiggle index for each episode in an HDF5 dataset"
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
        help="Add contact_wiggle_index as attribute to each episode in the HDF5 file"
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
        help="Threshold for contact detection: peg_bottom_z - hole_top_z <= threshold (default: -0.003)"
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
        peg_pos_key=args.peg_pos_key,
        hole_pos_key=args.hole_pos_key,
        peg_length=args.peg_length,
        hole_height=args.hole_height,
        contact_threshold=args.contact_threshold,
        verbose=not args.quiet,
    )
    
    return results


if __name__ == "__main__":
    main()
