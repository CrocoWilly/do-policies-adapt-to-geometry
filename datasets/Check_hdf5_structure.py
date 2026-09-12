# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to check if multiple HDF5 dataset files have identical env_args and episode structures.

Checks performed:
    1. env_args comparison across all files
    2. Episode structure comparison across all files  
    3. Internal consistency (all episodes within each file have same structure)
    4. Data shape comparison (ignoring episode length dimension)

Usage:
    python Check_hdf5_structure.py --files T1.hdf5 T2.hdf5
    python Check_hdf5_structure.py --files T1.hdf5 T2.hdf5 -v  # verbose mode (show full structure)
"""

import argparse
import h5py
import json
import os
from typing import Optional
from collections import defaultdict


def get_env_args(filepath: str) -> Optional[dict]:
    """Read and parse env_args from an HDF5 file."""
    try:
        with h5py.File(filepath, "r") as f:
            if "data" not in f:
                return None
            if "env_args" not in f["data"].attrs:
                return None
            env_args_str = f["data"].attrs["env_args"]
            return json.loads(env_args_str)
    except Exception as e:
        print(f"[ERROR] Failed to read env_args from {filepath}: {e}")
        return None


def get_episode_structure(filepath: str) -> Optional[dict]:
    """
    Get the structure of episodes in an HDF5 file.
    Returns a dict with episode info including keys, dtypes, and shapes.
    """
    try:
        with h5py.File(filepath, "r") as f:
            if "data" not in f:
                return None
            
            structure = {
                "num_episodes": 0,
                "episodes": {}
            }
            
            for episode_name in f["data"].keys():
                episode = f["data"][episode_name]
                episode_info = extract_group_structure(episode)
                structure["episodes"][episode_name] = episode_info
                structure["num_episodes"] += 1
            
            return structure
    except Exception as e:
        print(f"[ERROR] Failed to read structure from {filepath}: {e}")
        return None


def extract_group_structure(group: h5py.Group, prefix: str = "") -> dict:
    """Recursively extract structure from an HDF5 group."""
    structure = {}
    
    for key in group.keys():
        item = group[key]
        full_key = f"{prefix}/{key}" if prefix else key
        
        if isinstance(item, h5py.Dataset):
            info = {
                "type": "dataset",
                "dtype": str(item.dtype),
                "shape": item.shape,  # Always include shape
            }
            structure[key] = info
        elif isinstance(item, h5py.Group):
            structure[key] = {
                "type": "group",
                "children": extract_group_structure(item, full_key)
            }
    
    # Also extract attributes
    if group.attrs:
        structure["__attrs__"] = {k: type(v).__name__ for k, v in group.attrs.items()}
    
    return structure


def normalize_structure(structure: dict) -> dict:
    """
    Normalize structure for comparison by removing episode-specific details.
    Returns a canonical structure that represents the 'template' of an episode.
    """
    if not structure or not structure.get("episodes"):
        return {}
    
    # Get structure of first episode as template
    first_episode = list(structure["episodes"].values())[0]
    return first_episode


def compare_structures(struct1: dict, struct2: dict, path: str = "") -> list[str]:
    """
    Compare two structures recursively.
    Returns a list of differences found.
    """
    differences = []
    
    keys1 = set(struct1.keys())
    keys2 = set(struct2.keys())
    
    # Check for missing/extra keys
    missing = keys1 - keys2
    extra = keys2 - keys1
    
    for key in missing:
        differences.append(f"Missing key: {path}/{key}" if path else f"Missing key: {key}")
    for key in extra:
        differences.append(f"Extra key: {path}/{key}" if path else f"Extra key: {key}")
    
    # Compare common keys
    common_keys = keys1 & keys2
    for key in common_keys:
        v1, v2 = struct1[key], struct2[key]
        current_path = f"{path}/{key}" if path else key
        
        if isinstance(v1, dict) and isinstance(v2, dict):
            # Check type field
            if v1.get("type") != v2.get("type"):
                differences.append(f"Type mismatch at {current_path}: {v1.get('type')} vs {v2.get('type')}")
                continue
            
            # Compare dtype for datasets
            if v1.get("type") == "dataset":
                if v1.get("dtype") != v2.get("dtype"):
                    differences.append(f"Dtype mismatch at {current_path}: {v1.get('dtype')} vs {v2.get('dtype')}")
                # Always compare shape (ignoring first dimension which is episode length)
                if "shape" in v1 and "shape" in v2:
                    shape1 = v1["shape"][1:] if len(v1["shape"]) > 1 else v1["shape"]
                    shape2 = v2["shape"][1:] if len(v2["shape"]) > 1 else v2["shape"]
                    if shape1 != shape2:
                        differences.append(f"Shape mismatch at {current_path}: {v1['shape']} vs {v2['shape']} (ignoring episode length)")
            
            # Recursively compare children
            if "children" in v1 and "children" in v2:
                differences.extend(compare_structures(v1["children"], v2["children"], current_path))
            elif "children" in v1 or "children" in v2:
                differences.append(f"Structure mismatch at {current_path}: one has children, other doesn't")
        elif v1 != v2:
            differences.append(f"Value mismatch at {current_path}: {v1} vs {v2}")
    
    return differences


def print_structure(structure: dict, indent: int = 0):
    """Pretty print a structure dict."""
    prefix = "  " * indent
    for key, value in sorted(structure.items()):
        if isinstance(value, dict):
            if value.get("type") == "dataset":
                shape = value.get('shape', ())
                # Show shape without episode length (first dim)
                data_shape = shape[1:] if len(shape) > 1 else shape
                print(f"{prefix}{key}: [Dataset] dtype={value.get('dtype')}, data_shape={data_shape}")
            elif value.get("type") == "group":
                print(f"{prefix}{key}/")
                if "children" in value:
                    print_structure(value["children"], indent + 1)
            else:
                print(f"{prefix}{key}:")
                print_structure(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Check if multiple HDF5 files have identical env_args and episode structures."
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        required=True,
        help="List of HDF5 files to compare.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed structure information.",
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"[ERROR] File not found: {filepath}")
            return
    
    if len(args.files) < 2:
        print("[ERROR] Need at least 2 files to compare.")
        return
    
    print(f"\n{'='*70}")
    print("HDF5 STRUCTURE COMPARISON")
    print(f"{'='*70}")
    print(f"Files to compare: {len(args.files)}")
    for f in args.files:
        print(f"  - {f}")
    
    # =========================================================================
    # Part 1: Compare env_args
    # =========================================================================
    print(f"\n{'#'*70}")
    print("PART 1: env_args Comparison")
    print(f"{'#'*70}")
    
    env_args_list = []
    for filepath in args.files:
        env_args = get_env_args(filepath)
        if env_args is None:
            print(f"[WARNING] No env_args found in {filepath}")
        else:
            env_args_list.append((filepath, env_args))
            if args.verbose:
                print(f"\n{filepath}:")
                for k, v in env_args.items():
                    print(f"  {k}: {v}")
    
    env_args_match = True
    if len(env_args_list) >= 2:
        base_file, base_args = env_args_list[0]
        for filepath, env_args in env_args_list[1:]:
            if env_args != base_args:
                env_args_match = False
                print(f"\n[DIFF] env_args differ between '{base_file}' and '{filepath}'")
                # Show specific differences
                for key in set(base_args.keys()) | set(env_args.keys()):
                    v1 = base_args.get(key, "<missing>")
                    v2 = env_args.get(key, "<missing>")
                    if v1 != v2:
                        print(f"  {key}: {v1} vs {v2}")
    
    if env_args_match:
        print("\n✅ env_args: IDENTICAL across all files")
    else:
        print("\n❌ env_args: DIFFERENCES FOUND")
    
    # =========================================================================
    # Part 2: Compare episode structures
    # =========================================================================
    print(f"\n{'#'*70}")
    print("PART 2: Episode Structure & Shape Comparison")
    print(f"{'#'*70}")
    
    structures = []
    for filepath in args.files:
        structure = get_episode_structure(filepath)
        if structure is None:
            print(f"[WARNING] Could not read structure from {filepath}")
        else:
            structures.append((filepath, structure))
            print(f"\n{filepath}: {structure['num_episodes']} episodes")
    
    if args.verbose and structures:
        print(f"\n--- Episode Structure Template (from {structures[0][0]}) ---")
        template = normalize_structure(structures[0][1])
        print_structure(template)
    
    # Compare structures
    structure_match = True
    all_differences = []
    
    if len(structures) >= 2:
        base_file, base_struct = structures[0]
        base_template = normalize_structure(base_struct)
        
        for filepath, struct in structures[1:]:
            template = normalize_structure(struct)
            differences = compare_structures(base_template, template)
            
            if differences:
                structure_match = False
                all_differences.append((filepath, differences))
                print(f"\n[DIFF] Structure differs between '{base_file}' and '{filepath}':")
                for diff in differences:
                    print(f"  - {diff}")
    
    # Also check consistency within each file
    print(f"\n--- Checking internal consistency within each file ---")
    for filepath, struct in structures:
        episodes = struct["episodes"]
        if len(episodes) < 2:
            print(f"{filepath}: Only 1 episode, skipping internal check")
            continue
        
        episode_names = list(episodes.keys())
        first_episode = episodes[episode_names[0]]
        internal_consistent = True
        
        for ep_name in episode_names[1:]:
            ep_struct = episodes[ep_name]
            diffs = compare_structures(first_episode, ep_struct)
            if diffs:
                internal_consistent = False
                structure_match = False
                print(f"\n[DIFF] {filepath}: Episode structure differs between '{episode_names[0]}' and '{ep_name}':")
                for diff in diffs[:5]:  # Limit output
                    print(f"  - {diff}")
                if len(diffs) > 5:
                    print(f"  ... and {len(diffs) - 5} more differences")
        
        if internal_consistent:
            print(f"{filepath}: ✅ All {len(episodes)} episodes have consistent structure")
    
    if structure_match:
        print("\n✅ Episode Structure: IDENTICAL across all files")
    else:
        print("\n❌ Episode Structure: DIFFERENCES FOUND")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Files compared:        {len(args.files)}")
    print(f"env_args:              {'✅ IDENTICAL' if env_args_match else '❌ DIFFERENT'}")
    print(f"Episode structure:     {'✅ IDENTICAL' if structure_match else '❌ DIFFERENT'}")
    print(f"Data shapes:           {'✅ IDENTICAL' if structure_match else '❌ DIFFERENT'}")
    print(f"Internal consistency:  {'✅ ALL CONSISTENT' if structure_match else '❌ INCONSISTENT'}")
    
    if env_args_match and structure_match:
        print(f"\n🎉 All files are compatible - Safe to merge!")
    else:
        print(f"\n⚠️  Files have differences - Review before merging!")


if __name__ == "__main__":
    main()
