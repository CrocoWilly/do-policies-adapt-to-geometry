"""
Script to calculate the contact push speed for each episode in an HDF5 dataset.

Contact Push Speed Formula:
    contact_push_speed = abs(ee_pos_world_z[last] - ee_pos_world_z[contact_start]) / (T_last - T_contact_start)

This measures the average vertical (z-axis) speed of the end-effector during the contact/insertion phase.

Start contact condition: peg_bottom_z - hole_top_z <= contact_threshold (default: -0.003)

Where:
    - peg_bottom_z = peg_center_z - 0.5 * peg_length
    - hole_top_z = hole_bottom_z + hole_height
    - peg_pos_world is at peg center
    - hole_pos is at hole bottom

Usage:
    python BehaviorAnalysis_CalcContactPushSpeed.py --input test_insert_dataset.hdf5
    python BehaviorAnalysis_CalcContactPushSpeed.py --input test_insert_dataset.hdf5 --output contact_push_speed.csv
    python BehaviorAnalysis_CalcContactPushSpeed.py --input test_insert_dataset.hdf5 --peg-length 0.12 --hole-height 0.08
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def calculate_contact_push_speed(
    ee_pos_world: np.ndarray,
    peg_pos_world: np.ndarray,
    hole_pos: np.ndarray,
    peg_length: float = 0.12,
    hole_height: float = 0.08,
    contact_threshold: float = -0.003,
) -> Tuple[float, float, int, int, int]:
    """
    Calculate the contact push speed from end-effector world position data.
    
    Only starts counting when peg bottom approaches hole top:
    peg_bottom_z - hole_top_z <= contact_threshold
    
    Args:
        ee_pos_world: (T, 3) array of end-effector world positions [x, y, z]
        peg_pos_world: (T, 3) array of peg world positions (peg center)
        hole_pos: (T, 3) or (3,) array of hole world positions (hole bottom)
        peg_length: Length of the peg (default: 0.12)
        hole_height: Height of the hole (default: 0.08)
        contact_threshold: Threshold for contact detection (default: -0.003)
    
    Returns:
        Tuple of (contact_push_speed, mean_dz_per_step, z_displacement, contact_start_step, contact_steps, total_steps)
    
    Formula:
        contact_push_speed = abs(z_last - z_contact_start) / contact_steps
        mean_dz_per_step = mean(abs(diff(z)))
    """
    if ee_pos_world.ndim != 2 or ee_pos_world.shape[1] < 3:
        raise ValueError(f"ee_pos_world expected (T, >=3) but got {ee_pos_world.shape}")
    if peg_pos_world.ndim != 2 or peg_pos_world.shape[1] < 3:
        raise ValueError(f"peg_pos_world expected (T, >=3) but got {peg_pos_world.shape}")
    
    T = ee_pos_world.shape[0]
    if T < 2:
        return 0.0, 0.0, 0.0, -1, 0, T
    
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
        return 0.0, 0.0, 0.0, -1, 0, T
    
    contact_start = contact_indices[0]
    
    if contact_start >= T - 1:
        # Contact at last step, no speed to calculate
        return 0.0, 0.0, 0.0, contact_start, 0, T
    
    # Get z position at contact start and last time-step
    z_contact_start = ee_pos_world[contact_start, 2]
    z_last = ee_pos_world[-1, 2]
    
    # Calculate z displacement during contact phase
    z_displacement = abs(z_last - z_contact_start)
    
    # Calculate contact steps (from contact_start to end)
    contact_steps = T - contact_start
    
    # Calculate contact push speed
    if contact_steps > 0:
        contact_push_speed = z_displacement / contact_steps
    else:
        contact_push_speed = 0.0
    
    # Calculate mean_dz_per_step = mean(abs(diff(z)))
    z_contact = ee_pos_world[contact_start:, 2]
    dz = np.abs(np.diff(z_contact))
    mean_dz_per_step = float(np.mean(dz)) if len(dz) > 0 else 0.0
    
    return float(contact_push_speed), float(mean_dz_per_step), float(z_displacement), int(contact_start), contact_steps, T


def process_hdf5_file(
    input_hdf5: str,
    output_csv: Optional[str] = None,
    add_to_hdf5: bool = False,
    demo_prefix: str = "demo_",
    ee_pos_key: str = "ee_pos_rel",
    peg_pos_key: str = "peg_pos_world",
    hole_pos_key: str = "hole_pos_world",
    speed_attr_name: str = "contact_push_speed",
    peg_length: float = 0.12,
    hole_height: float = 0.08,
    contact_threshold: float = -0.003,
    verbose: bool = True,
) -> dict:
    """
    Process an HDF5 file and calculate contact push speed for each episode.
    
    Args:
        input_hdf5: Path to input HDF5 file
        output_csv: Optional path to save results as CSV
        add_to_hdf5: If True, add contact_push_speed as attribute to each episode
        demo_prefix: Prefix for demo episode names
        ee_pos_key: Key for end-effector position in obs group
        peg_pos_key: Key for peg world position in obs group
        hole_pos_key: Key for hole position in obs group
        speed_attr_name: Attribute name for storing contact push speed
        peg_length: Length of the peg
        hole_height: Height of the hole
        contact_threshold: Threshold for contact detection
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
            print(f"Peg length: {peg_length}, Hole height: {hole_height}, Contact threshold: {contact_threshold}")
            print("-" * 90)
        
        total_speed = 0.0
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
            
            ee_pos_world = obs[ee_pos_key][()]
            peg_pos_world = obs[peg_pos_key][()]
            hole_pos = obs[hole_pos_key][()]
            
            try:
                contact_push_speed, mean_dz_per_step, z_displacement, contact_start, contact_steps, total_steps = calculate_contact_push_speed(
                    ee_pos_world, peg_pos_world, hole_pos,
                    peg_length, hole_height, contact_threshold
                )
                
                results[demo_name] = {
                    "contact_push_speed": contact_push_speed,
                    "mean_dz_per_step": mean_dz_per_step,
                    "z_displacement": z_displacement,
                    "contact_start_step": contact_start,
                    "contact_steps": contact_steps,
                    "total_steps": total_steps,
                }
                
                if contact_start >= 0:
                    total_speed += contact_push_speed
                    contact_count += 1
                
                # Optionally add as attribute to the episode
                if add_to_hdf5:
                    demo.attrs[speed_attr_name] = contact_push_speed
                    demo.attrs["contact_start_step"] = contact_start
                
                if verbose:
                    if contact_start >= 0:
                        print(f"{demo_name}: {speed_attr_name} = {contact_push_speed:.6f}, mean_dz_per_step = {mean_dz_per_step:.6f} "
                              f"(z_disp={z_displacement:.4f}, contact_start={contact_start}, contact_steps={contact_steps}, T={total_steps})")
                    else:
                        print(f"{demo_name}: No contact detected (T={total_steps})")
                    
            except Exception as e:
                print(f"[ERROR] {demo_name}: Failed to calculate contact push speed: {e}")
                continue
        
        # Calculate summary statistics
        speed_values = [r["contact_push_speed"] for r in results.values() if r["contact_start_step"] >= 0]
        summary = {
            "total_episodes": len(results),
            "episodes_with_contact": contact_count,
            "avg_contact_push_speed": 0.0,
            "min_contact_push_speed": 0.0,
            "max_contact_push_speed": 0.0,
            "total_contact_push_speed": total_speed,
        }
        
        if speed_values:
            summary["avg_contact_push_speed"] = total_speed / len(speed_values)
            summary["min_contact_push_speed"] = min(speed_values)
            summary["max_contact_push_speed"] = max(speed_values)
        
        if verbose:
            print("-" * 90)
            print(f"Total episodes processed: {summary['total_episodes']}")
            print(f"Episodes with contact: {summary['episodes_with_contact']}")
            if speed_values:
                print(f"Average contact push speed: {summary['avg_contact_push_speed']:.6f}")
                print(f"Min contact push speed: {summary['min_contact_push_speed']:.6f}")
                print(f"Max contact push speed: {summary['max_contact_push_speed']:.6f}")
            
            if add_to_hdf5:
                print(f"\nContact push speeds saved as '{speed_attr_name}' attribute in each episode")
    
    # Optionally save results to CSV
    if output_csv:
        output_path = Path(output_csv)
        with open(output_path, "w") as f:
            f.write(f"episode,{speed_attr_name},mean_dz_per_step,z_displacement,contact_start_step,contact_steps,total_steps\n")
            for demo_name, data in sorted(results.items()):
                f.write(f"{demo_name},{data['contact_push_speed']:.6f},"
                        f"{data['mean_dz_per_step']:.6f},{data['z_displacement']:.6f},{data['contact_start_step']},"
                        f"{data['contact_steps']},{data['total_steps']}\n")
            # Add summary row
            f.write(f"\n# Summary Statistics\n")
            f.write(f"# Peg length: {peg_length}, Hole height: {hole_height}, Contact threshold: {contact_threshold}\n")
            f.write(f"# Total episodes: {summary['total_episodes']}\n")
            f.write(f"# Episodes with contact: {summary['episodes_with_contact']}\n")
            f.write(f"# Average contact push speed: {summary['avg_contact_push_speed']:.6f}\n")
            f.write(f"# Min contact push speed: {summary['min_contact_push_speed']:.6f}\n")
            f.write(f"# Max contact push speed: {summary['max_contact_push_speed']:.6f}\n")
        if verbose:
            print(f"\nResults saved to: {output_path}")
    
    return {
        "episodes": results,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate contact push speed for each episode in an HDF5 dataset"
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
        help="Add contact_push_speed as attribute to each episode in the HDF5 file"
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
        help="Threshold for contact detection: peg_bottom_z - hole_top_z <= contact_threshold (default: -0.003)"
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
