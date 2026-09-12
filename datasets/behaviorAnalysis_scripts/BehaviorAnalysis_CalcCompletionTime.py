"""
Script to calculate the completion time steps for each episode in an HDF5 dataset.

This records the total number of time-steps from start to end for each episode,
which can indicate task completion efficiency.

Usage:
    python BehaviorAnalysis_CalcCompletionTime.py --input test_insert_dataset.hdf5
    python BehaviorAnalysis_CalcCompletionTime.py --input test_insert_dataset.hdf5 --output completion_time.csv
    python BehaviorAnalysis_CalcCompletionTime.py --input test_insert_dataset.hdf5 --add-to-hdf5
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Optional


def get_episode_length(demo: h5py.Group, length_key: str = "actions") -> int:
    """
    Get the number of time-steps in an episode.
    
    Args:
        demo: HDF5 group for the episode
        length_key: Key to use for determining episode length (default: 'actions')
    
    Returns:
        Number of time-steps in the episode
    """
    # Try to get length from the specified key
    if length_key in demo:
        return demo[length_key].shape[0]
    
    # Fallback: try obs group
    if "obs" in demo:
        obs = demo["obs"]
        # Get the first dataset in obs
        for key in obs.keys():
            if isinstance(obs[key], h5py.Dataset):
                return obs[key].shape[0]
    
    # Fallback: try any dataset at demo level
    for key in demo.keys():
        if isinstance(demo[key], h5py.Dataset):
            return demo[key].shape[0]
    
    raise ValueError("Could not determine episode length")


def process_hdf5_file(
    input_hdf5: str,
    output_csv: Optional[str] = None,
    add_to_hdf5: bool = False,
    demo_prefix: str = "demo_",
    length_key: str = "actions",
    completion_attr_name: str = "completion_time_step",
    verbose: bool = True,
) -> dict:
    """
    Process an HDF5 file and calculate completion time steps for each episode.
    
    Args:
        input_hdf5: Path to input HDF5 file
        output_csv: Optional path to save results as CSV
        add_to_hdf5: If True, add completion_time_step as attribute to each episode
        demo_prefix: Prefix for demo episode names
        length_key: Key to use for determining episode length
        completion_attr_name: Attribute name for storing completion time steps
        verbose: Print detailed output
    
    Returns:
        Dictionary with 'episodes' (mapping episode names to time steps) and 'summary' (statistics)
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
        
        total_steps = 0
        
        for demo_name in demo_names:
            demo = data_grp[demo_name]
            
            try:
                time_steps = get_episode_length(demo, length_key)
                results[demo_name] = time_steps
                total_steps += time_steps
                
                # Optionally add as attribute to the episode
                if add_to_hdf5:
                    demo.attrs[completion_attr_name] = time_steps
                
                if verbose:
                    print(f"{demo_name}: {completion_attr_name} = {time_steps}")
                    
            except Exception as e:
                print(f"[ERROR] {demo_name}: Failed to get episode length: {e}")
                continue
        
        # Calculate summary statistics
        summary = {
            "total_episodes": len(results),
            "avg_time_steps": 0.0,
            "min_time_steps": 0,
            "max_time_steps": 0,
            "total_time_steps": total_steps,
        }
        
        if results:
            summary["avg_time_steps"] = total_steps / len(results)
            summary["min_time_steps"] = min(results.values())
            summary["max_time_steps"] = max(results.values())
        
        if verbose:
            print("-" * 60)
            print(f"Total episodes processed: {summary['total_episodes']}")
            if results:
                print(f"Average time steps: {summary['avg_time_steps']:.2f}")
                print(f"Min time steps: {summary['min_time_steps']}")
                print(f"Max time steps: {summary['max_time_steps']}")
                print(f"Total time steps: {summary['total_time_steps']}")
            
            if add_to_hdf5:
                print(f"\nCompletion time steps saved as '{completion_attr_name}' attribute in each episode")
    
    # Optionally save results to CSV
    if output_csv:
        output_path = Path(output_csv)
        with open(output_path, "w") as f:
            f.write(f"episode,{completion_attr_name}\n")
            for demo_name, steps in sorted(results.items()):
                f.write(f"{demo_name},{steps}\n")
            # Add summary row
            f.write(f"\n# Summary Statistics\n")
            f.write(f"# Total episodes: {summary['total_episodes']}\n")
            f.write(f"# Average time steps: {summary['avg_time_steps']:.2f}\n")
            f.write(f"# Min time steps: {summary['min_time_steps']}\n")
            f.write(f"# Max time steps: {summary['max_time_steps']}\n")
            f.write(f"# Total time steps: {summary['total_time_steps']}\n")
        if verbose:
            print(f"\nResults saved to: {output_path}")
    
    return {
        "episodes": results,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate completion time steps for each episode in an HDF5 dataset"
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
        help="Add completion_time_step as attribute to each episode in the HDF5 file"
    )
    parser.add_argument(
        "--demo-prefix",
        type=str,
        default="demo_",
        help="Prefix for demo episode names (default: 'demo_')"
    )
    parser.add_argument(
        "--length-key",
        type=str,
        default="actions",
        help="Key to use for determining episode length (default: 'actions')"
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
        length_key=args.length_key,
        verbose=not args.quiet,
    )
    
    return results


if __name__ == "__main__":
    main()
