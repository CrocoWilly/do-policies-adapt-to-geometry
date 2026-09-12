# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Techman TM7S + Robotiq 2F-85 (USD imported).

USD stage joints (per your screenshot):
- /World/tm7s_robotiq/joints/joint_1 ... joint_6
- /World/tm7s_robotiq/joints/robotiq_85_*_joint ...
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


# -----------------------------------------------------------------------------
# USD path (override-able)
# -----------------------------------------------------------------------------
# export TM7S_USD_PATH=~/IsaacSim_Project/tm7s_robotiq85.usd
TM7S_USD_PATH = os.path.expanduser(os.environ.get("TM7S_USD_PATH", "") + "") or \
    os.path.expanduser("~/IsaacSim_Project/tm7s_robotiq85_forIsaacLab.usd")


TM7S_ROBOTIQ85_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=TM7S_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # --- TM7S arm joints ---
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 1.57,
            "joint_4": 0.0,
            "joint_5": 1.57,
            "joint_6": 0.0,

            # --- Robotiq 2F-85 joints (revolute ones only) ---
            "robotiq_85_left_knuckle_joint": 0.0,
            "robotiq_85_right_knuckle_joint": 0.0,
            "robotiq_85_left_inner_knuckle_joint": 0.0,
            "robotiq_85_right_inner_knuckle_joint": 0.0,
            "robotiq_85_left_finger_tip_joint": 0.0,
            "robotiq_85_right_finger_tip_joint": 0.0,
        },
    ),
    actuators={
        # -----------------------
        # Arm (6-DoF)
        # -----------------------
        "tm7s_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-3]"],
            effort_limit_sim=170.0,
            velocity_limit_sim=3.6651914,
            stiffness=120.0,
            damping=6.0,
        ),
        "tm7s_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint_[4-6]"],
            effort_limit_sim=45.0,
            velocity_limit_sim=7.8539816,
            stiffness=80.0,
            damping=4.0,
        ),

        # -----------------------
        # Gripper (Robotiq 2F-85) - Franka style split
        # -----------------------
        # 1) drive joint: left knuckle is the primary commanded joint
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["robotiq_85_left_knuckle_joint"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=0.5,
            stiffness=50.0,
            damping=1.0,
        ),

        # 2) “finger” joints: (optional) give tiny PD so it doesn't feel completely floppy
        #    This USD has no inner_finger_joint (unlike the franka robotiq one), only inner_knuckle/tip.
        #    Use inner_knuckle as the "finger" segment, with a very small PD.
        "gripper_finger": ImplicitActuatorCfg(
            joint_names_expr=[
                "robotiq_85_left_inner_knuckle_joint",
                "robotiq_85_right_inner_knuckle_joint",
            ],
            effort_limit_sim=50.0,
            velocity_limit_sim=100.0,
            stiffness=0.2,
            damping=0.001,
        ),

        # 3) passive joints: mimic/closure joints set PD=0 to avoid fighting constraints
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=[
                "robotiq_85_right_knuckle_joint",
                "robotiq_85_left_finger_tip_joint",
                "robotiq_85_right_finger_tip_joint",
            ],
            effort_limit_sim=1.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of TM7S + Robotiq 2F-85."""


TM7S_ROBOTIQ85_HIGH_PD_CFG = TM7S_ROBOTIQ85_CFG.copy()
TM7S_ROBOTIQ85_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
TM7S_ROBOTIQ85_HIGH_PD_CFG.actuators["tm7s_shoulder"].stiffness = 400.0
TM7S_ROBOTIQ85_HIGH_PD_CFG.actuators["tm7s_shoulder"].damping = 80.0
TM7S_ROBOTIQ85_HIGH_PD_CFG.actuators["tm7s_wrist"].stiffness = 300.0
TM7S_ROBOTIQ85_HIGH_PD_CFG.actuators["tm7s_wrist"].damping = 60.0
"""Stiffer PD config for diff-IK / task-space control."""