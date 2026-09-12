# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.insert import mdp
from isaaclab_tasks.manager_based.manipulation.insert.mdp import franka_insert_events

from . import insert_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class EventCfg(insert_joint_pos_env_cfg.EventCfg):
    """Configuration for events."""

    randomize_light = EventTerm(
        func=franka_insert_events.randomize_scene_lighting_domelight,
        mode="reset",
        params={
            "intensity_range": (1500.0, 10000.0),
            "color_variation": 0.4,
            "textures": [
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hotel_room_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/small_empty_house_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/surgery_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
            ],
            "default_intensity": 3000.0,
            "default_color": (0.75, 0.75, 0.75),
            "default_texture": "",
        },
    )

    # randomize_table_visual_material = EventTerm(
    #     func=franka_insert_events.randomize_visual_texture_material,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("table"),
    #         "textures": [
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash/Ash_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Birch/Birch_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Mahogany_Planks/Mahogany_Planks_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Plywood/Plywood_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber_Cladding/Timber_Cladding_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Stone/Marble/Marble_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
    #         ],
    #         "default_texture": (
    #             f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableBase_BaseColor.png"
    #         ),
    #     },
    # )

    # override texture 
    randomize_table_visual_material = EventTerm(
        func=franka_insert_events.fix_visual_texture_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "textures": [                                                               # all commented out: Texture 0: dark wood (original table texture)
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash/Ash_BaseColor.png",    # Texture 1: light wood (Ash)
                # f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
            ],
            "default_texture": (                                                        # all commented out: Texture 0: dark wood (original table texture)
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash/Ash_BaseColor.png",    # Texture 1: light wood (Ash)
                # f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
            ),
        },
    )

    randomize_robot_arm_visual_texture = EventTerm(
        func=franka_insert_events.randomize_visual_texture_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "textures": [
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Cast/Aluminum_Cast_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Polished/Aluminum_Polished_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Brass/Brass_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Bronze/Bronze_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Brushed_Antique_Copper/Brushed_Antique_Copper_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Cast_Metal_Silver_Vein/Cast_Metal_Silver_Vein_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Copper/Copper_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Gold/Gold_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Iron/Iron_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/RustedMetal/RustedMetal_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Silver/Silver_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Carbon/Steel_Carbon_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
            ],
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        # actions = ObsTerm(func=mdp.last_action)
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # Insert task specific observations
        ee_pos_rel = ObsTerm(func=mdp.ee_pos_rel_hole)
        ee_yaw_rel = ObsTerm(func=mdp.ee_yaw_rel_hole)


        # ## Diagnosis3 ##
        # ee_yaw_rel_slot2 = ObsTerm(func=mdp.ee_yaw_rel_hole_slot2, params={"yaw_offset": 1.5708})
        # peg_pos_rel = ObsTerm(func=mdp.peg_pos_rel_hole)
        # peg_yaw_rel = ObsTerm(func=mdp.peg_yaw_rel_hole)
        # ###############


        # ## Diagnosis4 ##
        # # Shape encoding (circle=0, square=1, rectangle=2)
        # # peg_shape_no: (N, 1) - shape of the peg derived from peg.spawn.usd_path
        # # hole_shape_codes: (N, 3) - shapes at hole slots [+0.6cm, 0cm, -0.6cm]
        # #                   derived from hole.spawn.usd_path (ThreeShapeHoleBlock_*)
        # peg_shape_no = ObsTerm(func=mdp.peg_shape_no)
        # hole_shape_codes = ObsTerm(func=mdp.hole_shape_codes)
        # # hole_slot_positions_world: (N, 9) - real (env-relative) positions of
        # # the 3 hole slot centers on the box top surface, ordered to match
        # # hole_shape_codes: [slot_at_+0.6cm, slot_at_center, slot_at_-0.6cm].
        # # Combined with peg_shape_no, the policy can look up which slot has
        # # the matching shape and target its real-world (x, y, z) position.
        # hole_slot_positions_world = ObsTerm(func=mdp.hole_slot_positions_world)
        # ###############


        # ## Diagnosis5 ##
        # # 4 corners of peg bottom and hole top in world frame (env-relative), flattened (N, 12) each
        # peg_bottom_corners_world = ObsTerm(func=mdp.peg_bottom_corners_world)
        # peg_center_corners_world = ObsTerm(func=mdp.peg_center_corners_world)
        # peg_top_corners_world = ObsTerm(func=mdp.peg_top_corners_world)
        # hole_top_corners_world = ObsTerm(func=mdp.hole_top_corners_world)

        ee_pos_world = ObsTerm(func=mdp.ee_pos_in_env)
        ee_yaw_world = ObsTerm(func=mdp.ee_yaw_in_env)
        ee_quat_world = ObsTerm(func=mdp.ee_quat_in_env)

        hole_pos_world = ObsTerm(func=mdp.hole_pos_in_world)
        hole_quat_world = ObsTerm(func=mdp.hole_quat_in_world)
        hole_yaw_world = ObsTerm(func=mdp.hole_yaw_in_world)
        peg_pos_world = ObsTerm(func=mdp.peg_pos_in_world)
        peg_quat_world = ObsTerm(func=mdp.peg_quat_in_world)
        
        clearance = ObsTerm(func=mdp.clearance)
        

        # Appearance variations
        # color_code = ObsTerm(func=mdp.color_code)  # (N, 3)
        box_color_no = ObsTerm(func=mdp.box_color_no)  # (N, 1) red=0, green=1, blue=2
        peg_texture_no = ObsTerm(func=mdp.peg_texture_no)  # (N, 1) metal=0, wood=1
        table_texture_no = ObsTerm(func=mdp.table_texture_no)  # (N, 1) dark_wood=0, ash=1
        

        table_side_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("table_side_cam"), "data_type": "rgb", "normalize": False}
        )
        table_front_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("table_front_cam"), "data_type": "rgb", "normalize": False}
        )
        wrist_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False}
        )

        table_side_cam_depth = ObsTerm(
            func=mdp.image, 
            params={
                "sensor_cfg": SceneEntityCfg("table_side_cam"), 
                "data_type": "distance_to_image_plane", 
                "normalize": False
            }
        )

        table_front_cam_depth = ObsTerm(
            func=mdp.image, 
            params={
                "sensor_cfg": SceneEntityCfg("table_front_cam"), 
                "data_type": "distance_to_image_plane", 
                "normalize": False
            }
        )

        wrist_cam_depth = ObsTerm(
            func=mdp.image, 
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"), 
                "data_type": "distance_to_image_plane", 
                "normalize": False
            }
        )

        # 6. Nullspace posture control
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaPegInsertVisuomotorEnvCfg(insert_joint_pos_env_cfg.FrankaPegInsertEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()

    # Evaluation settings
    eval_mode = False
    eval_type = None

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
                ik_params={"lambda_val": 0.01},
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1234]),
        )

        # Set wrist camera
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
            ),
            offset=CameraCfg.OffsetCfg(
                # pos=(0.13, 0.0, -0.15), 
                # pos=(0.13, 0.0, -0.08),
                pos=(0.13, 0.0, -0.07),

                # rot=(-0.70614, 0.03701, 0.03701, -0.70614),  # roll = -6° pitch = 0° yaw = +90°
                # rot=(0.70106, -0.09230, -0.09230, 0.70106),  # roll = -15° pitch = 0° yaw = +90°
                rot=(0.69636, -0.12279, -0.12279, 0.69636),  # roll = -20° pitch = 0° yaw = +90°

                # rot=(0.69591, -0.12887, -0.12887, 0.69591),  # roll = -21° pitch = 0° yaw = +90°
                # rot=(0.69545, -0.13495, -0.13495, 0.69545),  # roll = -22° pitch = 0° yaw = +90°
                convention="ros"
            ),
        )

        # Set table front view camera
        # camera placement: looking down on the peg-in-hole workspace from the front
        # pos: (x = front/back, y = left/right, z = height) relative to the env origin
        # rot: quaternion (w, x, y, z), or (x, y, z, w) under the ROS convention
        self.scene.table_front_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_front_cam",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
            ),
            offset=CameraCfg.OffsetCfg(
                # pos=(1.0, 0.0, 0.5), 
                pos=(0.85, -0.02, 0.3),                     # slightly more level view
                rot=(0.35355, -0.61237, -0.61237, 0.35355), # roll = -120° pitch = 0° yaw = +90°
                convention="ros"
            ),
        )

        self.scene.table_side_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_side_cam",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
            ),
            offset=CameraCfg.OffsetCfg(
                # ## Setting 1: distant three-quarter view ##
                # pos=(1.0, 0.3, 0.6),
                # rot=(0.25, -0.43301, -0.75, 0.43301),     # roll = -120° pitch = 0° yaw = +120°
                # rot=(0.23087, -0.44351, -0.76818, 0.43301), # roll = -125° pitch = 0° yaw = +120°
                
                # ## Setting 2: close three-quarter view ##
                # pos=(0.8, 0.2, 0.4),
                # rot=(0.19134, -0.46194, -0.80010, 0.33141),   # roll = -135° pitch = 0° yaw = +120°
                # convention="ros"

                # ## Setting 3: closer three-quarter view ##
                # pos=(0.70, 0.115, 0.35),
                # rot=(0.19134, -0.46194, -0.80010, 0.33141),   # roll = -135° pitch = 0° yaw = +120°
                # convention="ros"

                ## Setting 4: very close three-quarter view ##
                pos=(0.65, 0.09, 0.3),
                rot=(0.19134, -0.46194, -0.80010, 0.33141),   # roll = -135° pitch = 0° yaw = +120°
                convention="ros"
            ),
        )


        # Set settings for camera rendering
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"  # Use DLAA for higher quality rendering

        # List of image observations in policy observations
        self.image_obs_list = ["table_front_cam", "wrist_cam", "table_side_cam"]
        # self.image_obs_list = ["table_front_cam"]
