# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run teleoperation for Insert task with 4D action space.

The insert task uses [Δx, Δy, Δz, Δyaw] in hole's local frame + gripper.
This script maps keyboard SE3 commands to the 4D + 1D action space.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleoperation for Insert task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default="Isaac-Peg-Insert-Franka-IK-Rel-v0", help="Task name.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import logging
import torch

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

logger = logging.getLogger(__name__)


def main() -> None:
    """Run teleoperation for insert task."""
    
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.terminations.time_out = None

    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # Flags
    should_reset = False

    def trigger_reset() -> None:
        nonlocal should_reset
        should_reset = True
        print("Reset triggered")

    # Create keyboard interface
    sensitivity = args_cli.sensitivity
    teleop_interface = Se3Keyboard(
        Se3KeyboardCfg(pos_sensitivity=0.5 * sensitivity, rot_sensitivity=0.5 * sensitivity)
    )
    teleop_interface.add_callback("R", trigger_reset)

    print("=" * 60)
    print("Insert Task Teleoperation")
    print("=" * 60)
    print("Keyboard controls:")
    print("  W/S: Move forward/backward (Δy in hole frame)")
    print("  A/D: Move left/right (Δx in hole frame)")
    print("  Q/E: Move up/down (Δz in hole frame)")
    print("  Z/X: Rotate yaw (Δyaw in hole frame)")
    print("  R: Reset environment")
    print("=" * 60)

    # Reset environment
    env.reset()
    teleop_interface.reset()

    print("Teleoperation started. Press 'R' to reset.")

    # Simulate
    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                # Get SE3 command: [x, y, z, roll, pitch, yaw, gripper]
                se3_action = teleop_interface.advance()  # (7,)

                # Map to insert action: [Δx, Δy, Δz, Δyaw, gripper]
                # SE3 keyboard: x=forward, y=left, z=up, yaw=rotate
                # Insert action in hole frame: x, y, z, yaw
                dx = se3_action[0]      # forward/backward -> x
                dy = se3_action[1]      # left/right -> y
                dz = se3_action[2]      # up/down -> z
                dyaw = se3_action[5]    # yaw rotation
                gripper = se3_action[6] # gripper

                # Create 5D action: [dx, dy, dz, dyaw, gripper]
                insert_action = torch.tensor([[dx, dy, dz, dyaw, gripper]], device=env.device)
                actions = insert_action.repeat(env.num_envs, 1)

                # Step
                env.step(actions)

                if should_reset:
                    env.reset()
                    teleop_interface.reset()
                    should_reset = False
                    print("Environment reset complete")

        except Exception as e:
            logger.error(f"Error during simulation step: {e}")
            break

    env.close()
    print("Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()
