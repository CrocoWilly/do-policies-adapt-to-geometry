# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import Se3RelRetargeterCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import stack_joint_pos_env_cfg

##
# Pre-defined configs

### New add ###
from isaaclab_assets.robots.tm7s import TM7S_ROBOTIQ85_HIGH_PD_CFG
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.stack import tm7s_mdp as mdp


### New add ###
@configclass
class Tm7sCubeStackEnvCfg(stack_joint_pos_env_cfg.Tm7sCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # --------------------------
        # Robot (TM7S): stiffer PD for better IK tracking
        # --------------------------
        self.scene.robot = TM7S_ROBOTIQ85_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot = TM7S_ROBOTIQ85_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot/tm7s_robotiq")

        # --------------------------
        # Actions (TM7S + DiffIK)
        # --------------------------
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-6]"],
            body_name="link_6",  # if this fails, change to "link_6"
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=0.5,
            # Franka used a +z offset into the hand.
            # For TM7S flange, start with zero; tune if your grasp frame is offset.
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )

        # --------------------------
        # Teleop devices (UNCHANGED)
        # --------------------------
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )

        # Override gripper_pos observation for Robotiq mimic gripper
        self.observations.policy.gripper_pos = ObsTerm(
            func=mdp.gripper_drive_joint_pos,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "drive_joint_name": "robotiq_85_left_knuckle_joint",
            },
        )

