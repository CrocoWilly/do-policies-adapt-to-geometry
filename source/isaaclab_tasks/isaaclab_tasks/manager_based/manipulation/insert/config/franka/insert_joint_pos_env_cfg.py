# isaaclab_tasks/manager_based/manipulation/insert/config/franka/insert_joint_pos_env_cfg.py

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.insert import mdp
from isaaclab_tasks.manager_based.manipulation.insert.mdp import franka_insert_events
from isaaclab_tasks.manager_based.manipulation.insert.assets import insert_asset
from isaaclab_tasks.manager_based.manipulation.insert.insert_env_cfg import InsertEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from isaaclab.sim.spawners.materials import MdlFileCfg


@configclass
class EventCfg:
    init_franka_arm_pose = EventTerm(
        func=franka_insert_events.set_default_joint_pose,
        mode="reset",
        params={
            # Can reuse the stack task's default pose (puts the hand above the table); roll, pitch = 90
            # "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.00, 0.00],  # original stack task 
            # "default_pose": [0.0444, -0.1894, -0.1107, -2.3, 0.0044, 2.2, 0.6952, 0.00, 0.00],        # hand sits a little higher
            # "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.75, 0.00, 0.00],

            # # Setting 1: with appoarch phase: roll, pitch = 90
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, -0.0044, 2.32, 0.75, 0.00, 0.00],     
            
            # Setting 2: without appoarch phase: joint_4 sets arm height, joint_5 sets pitch=90, joint_6 sets roll=90, joint_7 sets yaw=0
            # -> setting 2 currently has an odd bug: while the gripper holds the peg the peg drifts to one side, which badly affects policy training and inference
            # "default_pose": [0.0444, -0.1894, -0.1107, -2.62, -0.016, 2.43, 0.75, 0.00, 0.00],
            # "default_pose": [0.0444, -0.1894, -0.1107, -2.62, -0.016, 2.4740, 0.75, 0.00, 0.00],
        },
    )

    # The arm pose is never randomized; it stays fixed at default_pose
    randomize_franka_joint_state = EventTerm(
        func=franka_insert_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": 0.0, "asset_cfg": SceneEntityCfg("robot")},
    )

    place_hole_under_ee = EventTerm(
        func=franka_insert_events.place_hole_under_ee_randomized,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "hole_cfg": SceneEntityCfg("hole"),
            "ee_body_name": "panda_hand",
            
            # # Setting 1: with approach phase
            "dx": 0.0015,
            "dy": 0.0035,
            "dz": 0.34796,   # hole sits 0.4 m directly below the gripper

            # Setting 2: without approach phase
            # "dx": 0.00275,
            # "dy": 0.00155,
            # "dz": 0.31818,   # hole directly below the gripper

            
            # "xy_range": (-0.01, 0.01),   # x and y each randomized by +/-1 cm
            "xy_range": (0.00, 0.00),   # x and y each randomized by +/-1 cm


            ## Use this for Diagnosis 0 ##
            # "yaw_range": (-0.2618, 0.2618),  # yaw randomized +/-15 deg


            ## Use this for Diagnosis 1 ##
            "yaw_range": (-0.2618, -0.2617),    # yaw -15°
            # "yaw_range": (0.2617, 0.2618),    # yaw +15°
            # "yaw_range": (-0.2618, -0.0873),  # yaw randomized over [-15, -5] deg
            # "yaw_range": (-0.0873, 0.0873),  # yaw randomized over [-5, 5] deg
            # "yaw_range": (0.0873, 0.2618),  # yaw randomized over [+5, +15] deg

            ## Use this for Diagnosis 2 ##
            # "yaw_range": (-0.7854, -0.7853),    # yaw -45° -> 


            ## Use this for Diagnosis 3 data collection ##
            # "yaw_range": (-1.5708, 1.4835), # yaw -90° ~ +85°
            # "yaw_range": (-1.5708, -1.0472), # yaw -90° ~ -60°
            # "yaw_range": (-1.0472, -0.5234), # yaw -60° ~ -30°
            # "yaw_range": (-0.5234, 0.00),      # yaw -30° ~ 0°
            # "yaw_range": (0.00, 0.5234),      # yaw 0° ~ 30°
            # "yaw_range": (0.5234, 1.0472),    # yaw 30° ~ 60°
            # "yaw_range": (1.0472, 1.4835),    # yaw 60° ~ 85°


            ## Use this for Diagnosis 4 data collection ##
            # "yaw_range": (0.00, 0.00),  # yaw 0°
        },
    )

    attach_peg_to_ee = EventTerm(
        func=franka_insert_events.teleport_peg_to_ee_grasped,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "peg_cfg": SceneEntityCfg("peg"),
            "ee_body_name": "panda_hand",
            # "peg_offset_pos": (0.005, 0.0, -0.1534),  # peg below panda_hand (fingertip position)
            "peg_offset_pos": (0.0, 0.0, -0.1534),
            "peg_offset_yaw": 0.0,
        },
    )

    # # Rigidly attach the peg to the EE every step to emulate a firm grasp
    # rigid_attach_peg = EventTerm(
    #     func=franka_insert_events.attach_peg_to_ee_rigid,
    #     mode="interval",
    #     interval_range_s=(0.0, 0.0),  # run every step
    #     params={
    #         "robot_cfg": SceneEntityCfg("robot"),
    #         "peg_cfg": SceneEntityCfg("peg"),
    #         "ee_body_name": "panda_hand",
    #         "peg_offset_pos": (0.0, 0.0, 0.1534),  # same offset as at reset
    #         "peg_offset_yaw": 0.0,
    #     },
    # )



@configclass
class FrankaPegInsertEnvCfg(InsertEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events = EventCfg()

        # robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # actions (joint-pos version kept here; the IK version lives in another cfg)
        # self.actions.arm_action = mdp.Se3Delta4ActionCfg(
        #     asset_name="robot",
        #     body_name="panda_hand",
        #     # controller left unset; the IK cfg overrides it
        #     scale_xyz=0.01,          # about 1 cm per step (tune by feel)
        #     scale_yaw=0.10,          # yaw 0.1 rad per step
        #     fixed_roll_pitch=True,
        # )
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

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            # open_command_expr={"panda_finger_.*": 0.4},
            open_command_expr={"panda_finger_.*": 0.00},   # always closed
            close_command_expr={"panda_finger_.*": 0.00},  # always closed
        )

        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        # # rigid props (contact is sensitive during insertion; iteration counts can be retuned)
        # obj_props = RigidBodyPropertiesCfg(
        #     solver_position_iteration_count=16,
        #     solver_velocity_iteration_count=1,
        #     max_angular_velocity=1000.0,
        #     max_linear_velocity=1000.0,
        #     max_depenetration_velocity=2.0,
        #     disable_gravity=False,
        # )

        # --- peg & hole ---
        # Prepare two USD files: peg.usd / hole_block.usd (the hole is hollowed out inside)
        # Set the peg mass to half the original
        peg_mass_props = MassPropertiesCfg(density=500.0)  # density chosen to halve the mass
        # If the original mass is known, MassPropertiesCfg(mass=original_mass/2) also works
        
        peg_rigid_props = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=2.0,
            disable_gravity=True,
        )

        self.scene.peg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Peg",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.2], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=insert_asset("SquareCube.usd"),

                ## Use this for Diagnosis 4
                # usd_path=insert_asset("CirclePeg.usd"),
                # usd_path=insert_asset("RealSquarePeg.usd"),
                # usd_path=insert_asset("RectanglePeg.usd"),
                scale=(1.0, 1.0, 1.0),
                rigid_props=peg_rigid_props,
                mass_props=peg_mass_props,
                semantic_tags=[("class", "peg")],
                # override texture                                                       # all commented out: metal texture
                visual_material=MdlFileCfg(
                    mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak.mdl",    # wood texture
                    # mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",  # stone texture
                    project_uvw=True,
                    texture_scale=(1.0, 1.0),
                ),
            ),
        )

        # The hole is fixed to the table: kinematic, unaffected by physics forces
        hole_rigid_props = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=2.0,
            disable_gravity=False,
            kinematic_enabled=True,  # kinematic body, stays put
        )

        # Clearance / hole scale factor (change this to adjust the peg-hole gap)
        self.clearance = 1.12
        self.scene.hole = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Hole",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.0, 0.02], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(

                ## Use this for Diagnosis 0 and 1 ##
                usd_path=insert_asset("HoleBlock.usd"), 
                scale=(self.clearance, self.clearance, 0.8),
                # usd_path=insert_asset("HoleBlock_codeClearance100_usdBase112_hole160.usd"), 
                # scale=(1.0, 1.0, 0.8), 


                ## Use this for Diagnosis 1 - generalized ##
                # usd_path=insert_asset("HoleBlock_codeClearance100_usdBase112_hole124.usd"), 
                # usd_path=insert_asset("HoleBlock_codeClearance100_usdBase112_hole136.usd"), 
                # usd_path=insert_asset("HoleBlock_codeClearance100_usdBase112_hole148.usd"),
                # usd_path=insert_asset("HoleBlock_codeClearance100_usdBase112_hole160.usd"), 
                # usd_path=insert_asset("HoleBlock_codeClearance100_usdBase112_hole103.usd"),
                # scale=(1.0, 1.0, 0.8), 


                ## Use this for Diagnosis 2 ##
                # usd_path=insert_asset("XHoleBlock.usd"),


                ## Use this for Diagnosis 3 ##
                # usd_path=insert_asset("BigSmallXHoleBlock_case1.usd"),
                # usd_path=insert_asset("BigSmallXHoleBlock_case2.usd"),
                # usd_path=insert_asset("BigSmallXHoleBlock_case1_easyVersion.usd"),


                ## Use this for Diagnosis 4 ##
                # usd_path=insert_asset("ThreeShapeHoleBlock_1.usd"), 
                # usd_path=insert_asset("ThreeShapeHoleBlock_2.usd"),
                # usd_path=insert_asset("ThreeShapeHoleBlock_3.usd"),
                # usd_path=insert_asset("ThreeShapeHoleBlock_4.usd"),
                # usd_path=insert_asset("ThreeShapeHoleBlock_5.usd"),
                # usd_path=insert_asset("ThreeShapeHoleBlock_6.usd"),    
                

                # rigid_props=obj_props,
                rigid_props=hole_rigid_props,
                semantic_tags=[("class", "hole")],
                # override colour
                visual_material=sim_utils.PreviewSurfaceCfg(
                    # diffuse_color=(0.7, 0.2, 0.2),  # Red color
                    diffuse_color=(0.2, 0.7, 0.2),  # Green color
                    # diffuse_color=(0.2, 0.2, 0.7),  # Blue color
                    # diffuse_color=(0.8, 0.7, 0.1),  # Yellow color
                    # roughness=0.5,
                ),
            ),
        )

        # ee frame transformer (same approach as in stack)
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
            ],
        )
