"""
Script to calculate success rate and average completion time for episodes in an HDF5 dataset.

This script analyzes rollout results to provide:
    - Success rate: percentage of successful episodes
    - Average completion time: mean number of time-steps across episodes

Episode success is determined by the 'success' attribute in each episode.
Completion time is determined by 'num_samples' attribute or actions shape.

Usage:
    python Diagnosis1_successRate_completionTime.py --input rollout_dataset.hdf5
    python Diagnosis1_successRate_completionTime.py --input rollout_dataset.hdf5 --output results.csv
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Optional


def get_episode_info(demo: h5py.Group) -> dict:
    """
    Get success status and completion time from an episode.
    
    Args:
        demo: HDF5 group for the episode
    
    Returns:
        Dictionary with 'success' and 'num_steps'
    """
    # Get success from attributes
    success = demo.attrs.get("success", False)
    if isinstance(success, np.bool_):
        success = bool(success)
    
    # Get num_steps from attributes or actions shape
    num_steps = demo.attrs.get("num_samples", None)
    if num_steps is None:
        if "actions" in demo:
            num_steps = demo["actions"].shape[0]
        elif "obs" in demo:
            # Get from first obs dataset
            for key in demo["obs"].keys():
                if isinstance(demo["obs"][key], h5py.Dataset):
                    num_steps = demo["obs"][key].shape[0]
                    break
    
    if num_steps is None:
        num_steps = 0
    else:
        num_steps = int(num_steps)
    
    return {
        "success": success,
        "num_steps": num_steps,
    }


def process_hdf5_file(
    input_hdf5: str,
    output_csv: Optional[str] = None,
    demo_prefix: str = "demo_",
    verbose: bool = True,
) -> dict:
    """
    Process an HDF5 file and calculate success rate and completion time statistics.
    
    Args:
        input_hdf5: Path to input HDF5 file
        output_csv: Optional path to save results as CSV
        demo_prefix: Prefix for demo episode names
        verbose: Print detailed output
    
    Returns:
        Dictionary with 'episodes' (per-episode data) and 'summary' (aggregated statistics)
    """
    input_path = Path(input_hdf5)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    results = {}
    
    with h5py.File(input_path, "r") as f:
        if "data" not in f:
            raise KeyError("HDF5 file missing 'data' group")
        
        data_grp = f["data"]
        demo_names = sorted([k for k in data_grp.keys() if k.startswith(demo_prefix)])
        
        if verbose:
            print(f"Analyzing: {input_path.name}")
            print(f"Found {len(demo_names)} episodes")
            print("=" * 70)
        
        total_episodes = 0
        success_count = 0
        total_steps = 0
        success_steps = 0
        failed_steps = 0
        
        step_list = []
        success_step_list = []
        failed_step_list = []
        
        for demo_name in demo_names:
            demo = data_grp[demo_name]
            
            try:
                info = get_episode_info(demo)
                success = info["success"]
                num_steps = info["num_steps"]
                
                results[demo_name] = {
                    "success": success,
                    "num_steps": num_steps,
                }
                
                total_episodes += 1
                total_steps += num_steps
                step_list.append(num_steps)
                
                if success:
                    success_count += 1
                    success_steps += num_steps
                    success_step_list.append(num_steps)
                else:
                    failed_steps += num_steps
                    failed_step_list.append(num_steps)
                
                if verbose:
                    status = "✓ SUCCESS" if success else "✗ FAILED"
                    print(f"{demo_name}: {status}, steps={num_steps}")
                    
            except Exception as e:
                print(f"[ERROR] {demo_name}: Failed to process: {e}")
                continue
        
        # Calculate summary statistics
        success_rate = (success_count / total_episodes * 100) if total_episodes > 0 else 0.0
        
        # Completion time statistics (only for successful episodes)
        median_completion_time = np.median(success_step_list) if success_step_list else 0.0
        std_completion_time = np.std(success_step_list) if success_step_list else 0.0
        min_completion_time = min(success_step_list) if success_step_list else 0
        max_completion_time = max(success_step_list) if success_step_list else 0
        
        # Failed episode statistics
        median_failed_time = np.median(failed_step_list) if failed_step_list else 0.0
        std_failed_time = np.std(failed_step_list) if failed_step_list else 0.0
        
        summary = {
            "total_episodes": total_episodes,
            "success_count": success_count,
            "failed_count": total_episodes - success_count,
            "success_rate_percent": success_rate,
            "median_completion_time": median_completion_time,
            "std_completion_time": std_completion_time,
            "min_completion_time": min_completion_time,
            "max_completion_time": max_completion_time,
            "median_failed_time": median_failed_time,
            "std_failed_time": std_failed_time,
        }
        
        if verbose:
            print("=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print(f"Total Episodes:      {summary['total_episodes']}")
            print(f"Successful:          {summary['success_count']}")
            print(f"Failed:              {summary['failed_count']}")
            print(f"Success Rate:        {summary['success_rate_percent']:.2f}%")
            print("-" * 70)
            print("Completion Time (successful episodes only):")
            if success_count > 0:
                print(f"  Median:            {summary['median_completion_time']:.2f} steps (std: {summary['std_completion_time']:.2f})")
                print(f"  Min:               {summary['min_completion_time']} steps")
                print(f"  Max:               {summary['max_completion_time']} steps")
            else:
                print("  No successful episodes")
            if summary['failed_count'] > 0:
                print(f"Failed Episodes:     median={summary['median_failed_time']:.2f} steps (std: {summary['std_failed_time']:.2f})")
    
    # Optionally save results to CSV
    if output_csv:
        output_path = Path(output_csv)
        with open(output_path, "w") as f:
            # Per-episode results
            f.write("episode,success,num_steps\n")
            for demo_name, data in sorted(results.items()):
                f.write(f"{demo_name},{data['success']},{data['num_steps']}\n")
            
            # Summary
            f.write(f"\n# Summary Statistics\n")
            f.write(f"# Input file: {input_path.name}\n")
            f.write(f"# Total episodes: {summary['total_episodes']}\n")
            f.write(f"# Successful: {summary['success_count']}\n")
            f.write(f"# Failed: {summary['failed_count']}\n")
            f.write(f"# Success rate: {summary['success_rate_percent']:.2f}%\n")
            f.write(f"# Completion time (successful episodes only):\n")
            if success_count > 0:
                f.write(f"# Median completion time: {summary['median_completion_time']:.2f} steps (std: {summary['std_completion_time']:.2f})\n")
                f.write(f"# Min completion time: {summary['min_completion_time']} steps\n")
                f.write(f"# Max completion time: {summary['max_completion_time']} steps\n")
            if summary['failed_count'] > 0:
                f.write(f"# Median failed time: {summary['median_failed_time']:.2f} steps (std: {summary['std_failed_time']:.2f})\n")
        
        if verbose:
            print(f"\nResults saved to: {output_path}")
    
    return {
        "episodes": results,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate success rate and completion time statistics for HDF5 rollout dataset"
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
        "--demo-prefix",
        type=str,
        default="demo_",
        help="Prefix for demo episode names (default: 'demo_')"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress detailed output"
    )
    
    args = parser.parse_args()
    
    # Without --output, reuse the input file name with a .csv extension
    output_csv = args.output
    if output_csv is None:
        input_path = Path(args.input)
        output_csv = str(input_path.with_suffix('.csv'))
    
    results = process_hdf5_file(
        input_hdf5=args.input,
        output_csv=output_csv,
        demo_prefix=args.demo_prefix,
        verbose=not args.quiet,
    )
    
    return results


if __name__ == "__main__":
    main()
