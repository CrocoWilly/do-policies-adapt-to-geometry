# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy from robomimic.

This script loads a robomimic policy and plays it in an Isaac Lab environment.

Args:
    task: Name of the environment.
    checkpoint: Path to the robomimic policy checkpoint.
    horizon: If provided, override the step horizon of each rollout.
    num_rollouts: If provided, override the number of rollouts.
    seed: If provided, overeride the default random seed.
    norm_factor_min: If provided, minimum value of the action space normalization factor.
    norm_factor_max: If provided, maximum value of the action space normalization factor.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
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
import numpy as np
import random
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg


def rollout(policy, env, success_term, horizon, device):
    """Perform a single rollout of the policy in the environment.

    Args:
        policy: The robomimicpolicy to play.
        env: The environment to play in.
        horizon: The step horizon of each rollout.
        device: The device to run the policy on.

    Returns:
        terminated: Whether the rollout terminated.
        traj: The trajectory of the rollout.
    """
    policy.start_episode()
    obs_dict, _ = env.reset()
    traj = dict(actions=[], obs=[], next_obs=[])

    for i in range(horizon):
        # Prepare observations
        obs = copy.deepcopy(obs_dict["policy"])

        # Tm7s need to revise the following code: take env 0 only (or squeeze(0)); do not let torch.squeeze() collapse the feature dimension
        for k, v in obs.items():
            if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == 1:
                obs[k] = v.squeeze(0)   # remove only the env dim
        
        # # Original Franka used code
        # for ob in obs:
        #     obs[ob] = torch.squeeze(obs[ob])

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

        traj["obs"].append(obs)

        # ##### ----------------------------------------------------------------- #####
        # # Debug: Print observation info
        # if i == 0:  # Only print on first step to avoid spam
        #     print("\n" + "="*80)
        #     print("[DEBUG] Observation Structure:")
        #     print("="*80)
        #     for key, value in obs.items():
        #         if torch.is_tensor(value):
        #             print(f"  '{key}': Tensor shape={value.shape}, dtype={value.dtype}")
        #             if value.numel() <= 20:  # Print values if small enough
        #                 print(f"         values={value.cpu().numpy()}")
        #         elif isinstance(value, np.ndarray):
        #             print(f"  '{key}': ndarray shape={value.shape}, dtype={value.dtype}")
        #             if value.size <= 20:
        #                 print(f"         values={value}")
        #         else:
        #             print(f"  '{key}': type={type(value).__name__}, value={value}")
        #     print("="*80 + "\n")
        
        # # Debug: Print obs every 50 steps (less verbose)
        # if i % 50 == 0 and i > 0:
        #     print(f"\n[DEBUG Step {i}] Observation summary:")
        #     for key, value in obs.items():
        #         if torch.is_tensor(value) and value.numel() <= 10:
        #             print(f"  '{key}': {value.cpu().numpy()}")
        # ##### ----------------------------------------------------------------- #####


        # Compute actions
        actions = policy(obs)
        
        ##### ----------------------------------------------------------------- #####
        # Debug: Print which obs keys policy actually used (first step only)
        if i == 0:
            print("\n" + "="*80)
            print("[DEBUG] Policy Internal - Which obs keys does the policy actually use?")
            print("="*80)
            # Method 1: Check policy.nets.obs_key_shapes or similar
            if hasattr(policy, 'policy') and hasattr(policy.policy, 'nets'):
                nets = policy.policy.nets
                if hasattr(nets, 'encoder') and hasattr(nets.encoder, 'obs_shapes'):
                    print(f"  nets.encoder.obs_shapes: {nets.encoder.obs_shapes}")
            # Method 2: Check policy.obs_key_shapes
            if hasattr(policy, 'obs_key_shapes'):
                print(f"  policy.obs_key_shapes: {policy.obs_key_shapes}")
            # Method 3: Check policy.policy.obs_key_shapes
            if hasattr(policy, 'policy') and hasattr(policy.policy, 'obs_key_shapes'):
                print(f"  policy.policy.obs_key_shapes: {policy.policy.obs_key_shapes}")
            # Method 4: Check global_config from policy
            if hasattr(policy, 'global_config'):
                gc = policy.global_config
                if 'observation' in gc:
                    print(f"  global_config.observation: {gc['observation']}")
            # Method 5: List all attributes that contain 'obs'
            obs_attrs = [attr for attr in dir(policy) if 'obs' in attr.lower()]
            print(f"  policy attributes containing 'obs': {obs_attrs}")
            if hasattr(policy, 'policy'):
                obs_attrs_inner = [attr for attr in dir(policy.policy) if 'obs' in attr.lower()]
                print(f"  policy.policy attributes containing 'obs': {obs_attrs_inner}")
            print("="*80 + "\n")
        ##### ----------------------------------------------------------------- #####


        # Unnormalize actions
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = (
                (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min

        # Convert 4D action to 7D action if needed
        # 4D: [dx, dy, dz, dyaw] (scaled & clipped to [-1,1]) 
        # -> 7D: [dx, dy, dz, droll=0, dpitch=0, dyaw, dgripper=1.0] (original scale)
        # Reference: PostProcess_Add4DHoleRelAction.py
        if args_cli.action_4d_to_7d:
            if actions.shape[-1] == 4:
                # Reverse the scaling applied during dataset preprocessing
                # Dataset preprocessing: action_scaled = action_raw / scale, then clip to [-1,1]
                # Inference: action_raw = action_scaled * scale
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
                
                # Print converted 7D action
                print(f"[Step {i}] 4D->7D: scaled=[{dpos_scaled[0]:.4f}, {dpos_scaled[1]:.4f}, {dpos_scaled[2]:.4f}, {dyaw_scaled:.4f}] "
                      f"-> 7D=[{actions[0]:.6f}, {actions[1]:.6f}, {actions[2]:.6f}, {actions[3]:.6f}, {actions[4]:.6f}, {actions[5]:.6f}, {actions[6]:.6f}]")

        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

        # Apply actions
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]

        # Record trajectory
        traj["actions"].append(actions.tolist())
        traj["next_obs"].append(obs)

        # Check if rollout was successful
        if bool(success_term.func(env, **success_term.params)[0]):
            return True, traj
        elif terminated or truncated:
            return False, traj

    return False, traj


def main():
    """Run a trained policy from robomimic with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    # Set observations to dictionary mode for Robomimic
    env_cfg.observations.policy.concatenate_terms = False

    # Set termination conditions
    env_cfg.terminations.time_out = None

    # Disable recorder
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

    # Run policy
    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device)

        # ##### ----------------------------------------------------------------- #####
        # # Debug: Print policy expected observation keys
        # if trial == 0:
        #     print("\n" + "="*80)
        #     print("[DEBUG] Policy Expected Observation Keys (from checkpoint):")
        #     print("="*80)
        #     if hasattr(policy, 'obs_key_shapes') and policy.obs_key_shapes:
        #         for key, shape in policy.obs_key_shapes.items():
        #             print(f"  '{key}': shape={shape}")
        #     else:
        #         # Try to get from ckpt_dict safely
        #         try:
        #             if isinstance(ckpt_dict, dict):
        #                 config = ckpt_dict.get('config', {})
        #                 if isinstance(config, dict):
        #                     obs_config = config.get('observation', {})
        #                     if isinstance(obs_config, dict) and 'obs_key_shapes' in obs_config:
        #                         for key, shape in obs_config.get('obs_key_shapes', {}).items():
        #                             print(f"  '{key}': shape={shape}")
        #                     else:
        #                         print(f"  [INFO] ckpt_dict type: {type(ckpt_dict)}, keys: {ckpt_dict.keys() if isinstance(ckpt_dict, dict) else 'N/A'}")
        #                 else:
        #                     print(f"  [INFO] config is not a dict, type: {type(config)}")
        #             else:
        #                 print(f"  [INFO] ckpt_dict is not a dict, type: {type(ckpt_dict)}")
        #         except Exception as e:
        #             print(f"  [WARNING] Could not extract obs_key_shapes: {e}")
        #     print("="*80 + "\n")
        # ##### ----------------------------------------------------------------- #####
        
        terminated, traj = rollout(policy, env, success_term, args_cli.horizon, device)
        results.append(terminated)
        print(f"[INFO] Trial {trial}: {terminated}\n")

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")
    print(f"Trial Results: {results}\n")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
