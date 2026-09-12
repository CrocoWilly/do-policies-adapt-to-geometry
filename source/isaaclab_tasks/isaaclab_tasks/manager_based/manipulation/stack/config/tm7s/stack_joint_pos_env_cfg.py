# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack import tm7s_mdp as mdp
from isaaclab_tasks.manager_based.manipulation.stack.tm7s_stack_env_cfg import Tm7sStackEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

### New add ###
from isaaclab_assets.robots.tm7s import TM7S_ROBOTIQ85_CFG
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab_tasks.manager_based.manipulation.stack.tm7s_mdp import tm7s_stack_events

### New add ###
@configclass
class EventCfg:
    """Configuration for events (TM7S + Robotiq85)."""

    init_tm7s_arm_pose = EventTerm(
        func=tm7s_stack_events.set_default_tm7s_arm_pose,
        mode="reset",
        params={
            # only the 6 axes
            "default_arm_pose": [0.0, 0.0, 1.57, 0.0, 1.57, 0.0],
            "asset_cfg": SceneEntityCfg("robot"),
            "arm_joint_names": ["joint_[1-6]"],
        },
    )

    randomize_tm7s_joint_state = EventTerm(
        func=tm7s_stack_events.randomize_tm7s_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
            "arm_joint_names": ["joint_[1-6]"],
        },
    )

    randomize_cube_positions = EventTerm(
        func=tm7s_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )


### New add ###
@configclass
class Tm7sCubeStackEnvCfg(Tm7sStackEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        print("DEBUG gripper_pos func =", self.observations.policy.gripper_pos.func)

        # Set events
        self.events = EventCfg()

        # --------------------------
        # Robot: TM7S + Robotiq85
        # --------------------------
        self.scene.robot = TM7S_ROBOTIQ85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot = TM7S_ROBOTIQ85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot/tm7s_robotiq")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # --------------------------
        # Actions (TM7S)
        # --------------------------
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-6]"],
            scale=0.5,
            use_default_offset=True,
        )

        # Robotiq85: drive only the left knuckle joint (main actuated joint)
        # URDF limit upper ~= 0.80285
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["robotiq_85_left_knuckle_joint"],
            open_command_expr={"robotiq_85_left_knuckle_joint": 0.0},
            close_command_expr={"robotiq_85_left_knuckle_joint": 0.35},
        )

        # utilities for gripper status check
        self.gripper_joint_names = ["robotiq_85_left_knuckle_joint"]
        self.gripper_open_val = 0.0
        self.gripper_close_val = 0.35
        self.gripper_threshold = 0.01


        # --------------------------
        # Observations: override gripper_pos for Robotiq mimic gripper
        # --------------------------
        self.observations.policy.gripper_pos = ObsTerm(
            func=mdp.gripper_drive_joint_pos,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "drive_joint_name": "robotiq_85_left_knuckle_joint",
            },
        )

        # --------------------------
        # Rigid body properties of each cube (UNCHANGED)
        # --------------------------
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Set each stacking cube deterministically (UNCHANGED)
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_1")],
            ),
        )
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_2")],
            ),
        )
        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.1, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_3")],
            ),
        )

        # --------------------------
        # Frame Transformer (TM7S)
        # --------------------------
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            # use base link of the robot as reference
            prim_path="{ENV_REGEX_NS}/Robot/tm7s_robotiq/link_0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                # End-effector: flange (exists in your USD)
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/tm7s_robotiq/link_6",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/tm7s_robotiq/robotiq_85_base_link",
                    name="robotiq_grasp_frame",
                    offset=OffsetCfg(
                        pos=[0.130, 0.0, 0.0],   # <- put your grasp offset here
                        rot=[1.0, 0.0, 0.0, 0.0] # <- quaternion, identity for now
                    ),
                )
                # # Tool tip frames: finger tips
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/tm7s_robotiq/robotiq_85_right_finger_tip_link",
                #     name="tool_rightfinger",
                #     offset=OffsetCfg(
                #         pos=(0.0, 0.0, 0.0),
                #     ),
                # ),
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/tm7s_robotiq/robotiq_85_left_finger_tip_link",
                #     name="tool_leftfinger",
                #     offset=OffsetCfg(
                #         pos=(0.0, 0.0, 0.0),
                #     ),
                # ),
            ],
        )

