# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a trained 4D policy from robomimic and save rollouts to HDF5 dataset.

This script loads a robomimic policy, plays it in an Isaac Lab environment,
and saves the rollout trajectories to an HDF5 dataset file in robomimic format.

Args:
    task: Name of the environment.
    checkpoint: Path to the robomimic policy checkpoint.
    dataset_file: Path to output HDF5 dataset file.
    horizon: If provided, override the step horizon of each rollout.
    num_rollouts: If provided, override the number of rollouts.
    seed: If provided, override the default random seed.
    norm_factor_min: If provided, minimum value of the action space normalization factor.
    norm_factor_max: If provided, maximum value of the action space normalization factor.

Usage:
    isaaclab.sh -p scripts/imitation_learning/robomimic/play_4d_and_save.py \
        --task Isaac-Insert-Franka-v0 \
        --checkpoint /path/to/model.pth \
        --dataset_file ./datasets/exp1_inference_clearance120_yaw0_episodes100.hdf5 \
        --num_rollouts 100 \
        --horizon 200
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment and save to HDF5.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/policy_rollout.hdf5",
    help="File path to export rollout dataset (e.g., ./datasets/exp1_inference.hdf5)."
)
parser.add_argument("--horizon", type=int, default=300, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional: minimum value of the normalization factor."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional: maximum value of the normalization factor."
)
parser.add_argument("--enable_pinocchio", default=False, action="store_true", help="Enable Pinocchio.")
parser.add_argument(
    "--action_4d_to_7d", default=True, action="store_true",
    help="Convert 4D action [dx, dy, dz, dyaw] to 7D action [dx, dy, dz, droll=0, dpitch=0, dyaw, dgripper=0]."
)
parser.add_argument(
    "--scale_xyz", type=float, default=0.05,
    help="Scale factor for xyz position used during dataset preprocessing (default: 0.05 meters)."
)
parser.add_argument(
    "--scale_yaw", type=float, default=0.05,
    help="Scale factor for yaw rotation used during dataset preprocessing (default: 0.05 radians)."
)
parser.add_argument(
    "--c_min", type=float, default=1.03,
    help="Min clearance value for normalization (default: 1.03, matching PostProcess script)."
)
parser.add_argument(
    "--c_max", type=float, default=1.60,
    help="Max clearance value for normalization (default: 1.60, matching PostProcess script)."
)
parser.add_argument(
    "--save_successful_only", action="store_true", default=False,
    help="If set, only save successful rollouts to the dataset."
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import gymnasium as gym
import h5py
import json
import numpy as np
import os
import random
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg


class HDF5DatasetWriter:
    """Writer for saving rollout trajectories to HDF5 dataset file in robomimic format."""

    def __init__(self, filepath: str, env_name: str, env_args: dict = None):
        """Initialize HDF5 dataset writer.

        Args:
            filepath: Path to the HDF5 file to create.
            env_name: Name of the environment.
            env_args: Optional dictionary of environment arguments.
        """
        self.filepath = filepath
        self.env_name = env_name
        self.env_args = env_args or {}
        self.episode_count = 0

        # Create output directory if needed
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Create HDF5 file
        self.file = h5py.File(filepath, "w")
        self.data_group = self.file.create_group("data")

        # Store env_args as attribute
        env_args_data = {
            "env_name": env_name,
            "type": 1,  # robomimic convention
            "env_kwargs": self.env_args,
        }
        self.data_group.attrs["env_args"] = json.dumps(env_args_data)

    def add_episode(self, traj: dict, success: bool = False):
        """Add an episode trajectory to the dataset.

        Args:
            traj: Dictionary containing:
                - actions: List of action arrays (each shape (1, action_dim) or (action_dim,))
                - obs: List of observation dictionaries
                - next_obs: List of next observation dictionaries
            success: Whether the episode was successful.
        """
        demo_name = f"demo_{self.episode_count}"
        demo_group = self.data_group.create_group(demo_name)

        # Process actions - convert list of tensors/arrays to single array
        actions_list = []
        for a in traj["actions"]:
            if isinstance(a, list):
                a = np.array(a)
            if isinstance(a, torch.Tensor):
                a = a.cpu().numpy()
            if a.ndim == 2:
                a = a.squeeze(0)  # Remove batch dimension
            actions_list.append(a)

        if len(actions_list) > 0:
            actions_array = np.stack(actions_list, axis=0).astype(np.float32)
            demo_group.create_dataset("actions", data=actions_array, compression="gzip", compression_opts=4)

        # Process observations
        if len(traj["obs"]) > 0:
            obs_group = demo_group.create_group("obs")
            first_obs = traj["obs"][0]

            for key in first_obs.keys():
                obs_data_list = []
                for obs_dict in traj["obs"]:
                    value = obs_dict[key]
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()
                    obs_data_list.append(value)

                obs_array = np.stack(obs_data_list, axis=0)
                obs_group.create_dataset(key, data=obs_array, compression="gzip", compression_opts=4)

        # Process next_obs (optional, for compatibility)
        if len(traj.get("next_obs", [])) > 0:
            next_obs_group = demo_group.create_group("next_obs")
            first_next_obs = traj["next_obs"][0]

            # Handle the case where next_obs contains tensors
            for key in first_next_obs.keys():
                next_obs_data_list = []
                for obs_dict in traj["next_obs"]:
                    value = obs_dict[key]
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()
                    next_obs_data_list.append(value)

                next_obs_array = np.stack(next_obs_data_list, axis=0)
                next_obs_group.create_dataset(key, data=next_obs_array, compression="gzip", compression_opts=4)

        # Store episode metadata
        demo_group.attrs["num_samples"] = len(traj["actions"])
        demo_group.attrs["success"] = success

        self.episode_count += 1
        print(f"  Saved episode {demo_name}: {len(traj['actions'])} steps, success={success}")

    def close(self):
        """Close the HDF5 file."""
        # Update total episodes count
        self.data_group.attrs["total"] = self.episode_count
        self.file.close()
        print(f"\nDataset saved to: {self.filepath}")
        print(f"Total episodes: {self.episode_count}")


def rollout(policy, env, success_term, horizon, device):
    """Perform a single rollout of the policy in the environment.

    Args:
        policy: The robomimic policy to play.
        env: The environment to play in.
        success_term: The success termination term for checking success.
        horizon: The step horizon of each rollout.
        device: The device to run the policy on.

    Returns:
        success: Whether the rollout was successful.
        traj: The trajectory of the rollout containing actions, obs, and next_obs.
    """
    policy.start_episode()
    obs_dict, _ = env.reset()
    traj = dict(actions=[], obs=[], next_obs=[])

    for i in range(horizon):
        # Prepare observations
        obs = copy.deepcopy(obs_dict["policy"])

        # Only squeeze env dim, not feature dim
        for k, v in obs.items():
            if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == 1:
                obs[k] = v.squeeze(0)   # Only remove env dim

        # Check if environment image observations
        if hasattr(env.cfg, "image_obs_list"):
            # Process RGB image observations for robomimic inference
            for image_name in env.cfg.image_obs_list:
                if image_name in obs_dict["policy"].keys():
                    # Convert from HWC uint8 to CHW normalized float
                    image = torch.squeeze(obs_dict["policy"][image_name])
                    image = image.permute(2, 0, 1).clone().float()
                    image = image / 255.0
                    image = image.clip(0.0, 1.0)
                    obs[image_name] = image

        # Process clearance observation for robomimic inference
        # Training data clearance was scaled by PostProcess script:
        #   c_norm = 2 * (c - c_min) / (c_max - c_min) - 1  ->  maps [c_min, c_max] to [-1, 1]
        if "clearance" in obs and torch.is_tensor(obs["clearance"]):
            c_raw = obs["clearance"].float()
            obs["clearance"] = (2.0 * (c_raw - args_cli.c_min) / (args_cli.c_max - args_cli.c_min) - 1.0)

        # Process depth image observations for robomimic inference
        # Training data depth was normalized to [0, 1] by PostProcess script:
        #   inf/nan -> max_depth, clip to [0, max_depth], divide by max_depth
        # Then robomimic's DepthModality does HWC->CHW during training data loading.
        # During inference, _prepare_observation does NOT call process_obs_dict,
        # so we must do the full pipeline here: normalize [0,1] + HWC->CHW.
        for key in list(obs.keys()):
            if key.endswith("_depth") and torch.is_tensor(obs[key]):
                depth = obs[key].clone().float()
                # Replace inf/nan with the max finite value, then normalize to [0, 1]
                finite_mask = torch.isfinite(depth)
                if finite_mask.any():
                    max_depth = depth[finite_mask].max()
                else:
                    max_depth = torch.tensor(1.0)
                depth = torch.where(finite_mask, depth, max_depth)
                depth = depth.clamp(0.0, max_depth.item())
                if max_depth > 0:
                    depth = depth / max_depth
                # HWC (H, W, 1) -> CHW (1, H, W)
                if depth.ndim == 3 and depth.shape[-1] == 1:
                    depth = depth.permute(2, 0, 1)
                obs[key] = depth

        # Store original obs (before any policy processing) for dataset
        # Make a deep copy to preserve for saving
        obs_for_save = {}
        for k, v in obs.items():
            if torch.is_tensor(v):
                obs_for_save[k] = v.clone()
            elif isinstance(v, np.ndarray):
                obs_for_save[k] = v.copy()
            else:
                obs_for_save[k] = copy.deepcopy(v)

        traj["obs"].append(obs_for_save)

        # Compute actions
        actions = policy(obs)

        # Unnormalize actions
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = (
                (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min

        # Convert 4D action to 7D action if needed
        # 4D: [dx, dy, dz, dyaw] (scaled & clipped to [-1,1])
        # -> 7D: [dx, dy, dz, droll=0, dpitch=0, dyaw, dgripper=1.0] (original scale)
        if args_cli.action_4d_to_7d:
            if actions.shape[-1] == 4:
                # Reverse the scaling applied during dataset preprocessing
                dpos_scaled = actions[0:3]  # dx, dy, dz (scaled)
                dyaw_scaled = actions[3]    # dyaw (scaled)

                dpos_raw = dpos_scaled * args_cli.scale_xyz  # restore to meters
                dyaw_raw = dyaw_scaled * args_cli.scale_yaw  # restore to radians

                actions_7d = np.zeros(7, dtype=actions.dtype)
                actions_7d[0:3] = dpos_raw      # dx, dy, dz
                actions_7d[3] = 0.0             # droll = 0
                actions_7d[4] = 0.0             # dpitch = 0
                actions_7d[5] = dyaw_raw        # dyaw
                actions_7d[6] = 1.0             # dgripper = 1.0 (keep gripper closed)
                actions = actions_7d

                # Print converted 7D action (less verbose for saving mode)
                if i % 50 == 0:
                    print(f"[Step {i}] 4D->7D: scaled=[{dpos_scaled[0]:.4f}, {dpos_scaled[1]:.4f}, {dpos_scaled[2]:.4f}, {dyaw_scaled:.4f}] "
                          f"-> 7D=[{actions[0]:.6f}, {actions[1]:.6f}, {actions[2]:.6f}, {actions[3]:.6f}, {actions[4]:.6f}, {actions[5]:.6f}, {actions[6]:.6f}]")

        actions_tensor = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

        # Apply actions
        obs_dict, _, terminated, truncated, _ = env.step(actions_tensor)

        # Record trajectory - store the 7D actions that were actually sent to env
        traj["actions"].append(actions.copy())

        # Store next_obs
        next_obs = {}
        for k, v in obs_dict["policy"].items():
            if torch.is_tensor(v):
                if v.ndim >= 1 and v.shape[0] == 1:
                    next_obs[k] = v.squeeze(0).clone()
                else:
                    next_obs[k] = v.clone()
            elif isinstance(v, np.ndarray):
                next_obs[k] = v.copy()
            else:
                next_obs[k] = copy.deepcopy(v)
        traj["next_obs"].append(next_obs)

        # Check if rollout was successful
        if bool(success_term.func(env, **success_term.params)[0]):
            return True, traj
        elif terminated or truncated:
            return False, traj

    return False, traj


def main():
    """Run a trained policy from robomimic with Isaac Lab environment and save to HDF5."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    # Set observations to dictionary mode for Robomimic
    env_cfg.observations.policy.concatenate_terms = False

    # Set termination conditions
    env_cfg.terminations.time_out = None

    # Disable recorder (we use our own HDF5 writer)
    env_cfg.recorders = None

    # Extract success checking function
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Set seed
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)

    # Acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # Initialize HDF5 dataset writer
    env_args = {
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "horizon": args_cli.horizon,
        "seed": args_cli.seed,
        "scale_xyz": args_cli.scale_xyz,
        "scale_yaw": args_cli.scale_yaw,
        "action_4d_to_7d": args_cli.action_4d_to_7d,
    }
    dataset_writer = HDF5DatasetWriter(
        filepath=args_cli.dataset_file,
        env_name=args_cli.task,
        env_args=env_args
    )

    # Run policy
    results = []
    successful_count = 0

    for trial in range(args_cli.num_rollouts):
        print(f"\n[INFO] Starting trial {trial + 1}/{args_cli.num_rollouts}")
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device)

        success, traj = rollout(policy, env, success_term, args_cli.horizon, device)
        results.append(success)

        # Save episode to dataset
        if args_cli.save_successful_only:
            if success:
                dataset_writer.add_episode(traj, success=success)
                successful_count += 1
            else:
                print(f"  Trial {trial}: Failed, not saving (save_successful_only=True)")
        else:
            dataset_writer.add_episode(traj, success=success)
            if success:
                successful_count += 1

        print(f"[INFO] Trial {trial + 1}: {'SUCCESS' if success else 'FAILED'}")

    # Close dataset writer
    dataset_writer.close()

    # Print summary
    print(f"\n" + "=" * 60)
    print(f"SUMMARY")
    print(f"=" * 60)
    print(f"Successful trials: {results.count(True)} / {len(results)}")
    print(f"Success rate: {results.count(True) / len(results):.2%}")
    print(f"Trial Results: {results}")
    if args_cli.save_successful_only:
        print(f"Episodes saved: {successful_count} (successful only)")
    else:
        print(f"Episodes saved: {len(results)} (all)")
    print(f"Dataset file: {args_cli.dataset_file}")
    print(f"=" * 60)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
