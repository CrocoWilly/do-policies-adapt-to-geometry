"""
Copy action_world_4d into the actions key of an HDF5 dataset.
This lets robomimic train on the 4D action space.

Usage:
    python copy_action_world_4d_to_actions.py <input_file> [output_file]
    
Without output_file, a _4d_actions suffix is appended to the input name.
"""

import h5py
import sys
import os
from rich import print


def copy_action_world_4d_to_actions(input_path: str, output_path: str = None):
    """
    Copy action_world_4d into the actions key.
    
    Args:
        input_path: Path to the input HDF5 file
        output_path: Path to the output HDF5 file (optional; a new file is created by default)
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_4d_actions{ext}"
    
    # Copy the file
    if input_path != output_path:
        import shutil
        print(f"[yellow]Copying file: {input_path} -> {output_path}[/yellow]")
        shutil.copyfile(input_path, output_path)
    
    # Open the copy and modify it
    with h5py.File(output_path, "r+") as f:
        demos = [key for key in f["data"].keys() if key.startswith("demo_")]
        print(f"[blue]Found {len(demos)} demos[/blue]")
        
        for demo_name in sorted(demos):
            demo = f[f"data/{demo_name}"]
            
            # Check action_world_4d exists
            if "action_world_4d" not in demo:
                print(f"[red]Warning: {demo_name} has no action_world_4d, skipping[/red]")
                continue
            
            # Read action_world_4d
            action_world_4d = demo["action_world_4d"][:]
            
            # Delete the old actions if present
            if "actions" in demo:
                old_shape = demo["actions"].shape
                del demo["actions"]
                print(f"[cyan]{demo_name}[/cyan]: deleted old actions {old_shape}, ", end="")
            else:
                print(f"[cyan]{demo_name}[/cyan]: ", end="")
            
            # Copy action_world_4d into actions
            demo.create_dataset("actions", data=action_world_4d)
            print(f"new actions shape={action_world_4d.shape}")
    
    print(f"\n[green]Done. Output file: {output_path}[/green]")
    return output_path


if __name__ == "__main__":

    input_file = "FUCK_with4DAction.hdf5"
    output_file = "FUCK_with4DAction_forTraining.hdf5"
    
    copy_action_world_4d_to_actions(input_file, output_file)
