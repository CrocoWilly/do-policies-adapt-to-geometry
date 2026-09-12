# isaaclab_tasks/manager_based/manipulation/insert/insert_env_cfg.py

from dataclasses import MISSING
import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from . import mdp

from isaaclab.sim.spawners.materials import MdlFileCfg


@configclass
class InsertSceneCfg(InteractiveSceneCfg):
    # robot / ee_frame are filled in by the franka cfg
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # --- insertion objects ---
    peg: RigidObjectCfg = MISSING
    hole: RigidObjectCfg = MISSING


# @configclass
# class ActionsCfg:
#     # target: policy action = [dx, dy, dz, dyaw]
#     # arm_action: mdp.Se3Delta4ActionCfg = MISSING
#     arm_action: DifferentialInverseKinematicsActionCfg = MISSING
#     gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
@configclass
class ActionsCfg:
    arm_action: DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING



@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):

        ##           Train policy used observation data            ##
        # EE position & yaw relative to hole: [ee_hole_x, ee_hole_y, ee_hole_z, ee_hole_yaw]
        ee_pos_rel = ObsTerm(
            func=mdp.ee_pos_rel_hole,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "hole_cfg": SceneEntityCfg("hole"),
            },
        )  # (N, 3)
        ee_yaw_rel = ObsTerm(
            func=mdp.ee_yaw_rel_hole,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "hole_cfg": SceneEntityCfg("hole"),
            },
        )  # (N, 1)

        # ## SmallExp4 ##
        # # 4 corners of peg bottom and hole top in world frame (env-relative), flattened (N, 12) each
        # peg_bottom_corners_world = ObsTerm(func=mdp.peg_bottom_corners_world)
        # peg_center_corners_world = ObsTerm(func=mdp.peg_center_corners_world)
        # peg_top_corners_world = ObsTerm(func=mdp.peg_top_corners_world)
        # hole_top_corners_world = ObsTerm(func=mdp.hole_top_corners_world)

        # ## SmallExp7 ##
        # ee_yaw_rel_slot2 = ObsTerm(func=mdp.ee_yaw_rel_hole_slot2, params={"yaw_offset": 1.5708})
        # peg_pos_rel = ObsTerm(func=mdp.peg_pos_rel_hole)
        # peg_yaw_rel = ObsTerm(func=mdp.peg_yaw_rel_hole)
        # ###############

        ## SmallExp8 ##
        peg_shape_no = ObsTerm(func=mdp.peg_shape_no)
        hole_shape_codes = ObsTerm(func=mdp.hole_shape_codes)
        hole_slot_positions_world = ObsTerm(func=mdp.hole_slot_positions_world)
        ###############


        # EE position in world frame (relative to env origin): [ee_x, ee_y, ee_z]
        ee_pos_world = ObsTerm(
            func=mdp.ee_pos_in_env,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )  # (N, 3)

        # EE yaw in world frame: [ee_yaw]
        ee_yaw_world = ObsTerm(
            func=mdp.ee_yaw_in_env,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )  # (N, 1)

        # EE orientation (quaternion) in world frame: [ee_quat_w, ee_quat_x, ee_quat_y, ee_quat_z]
        ee_quat_world = ObsTerm(
            func=mdp.ee_quat_in_env,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )  # (N, 4)

        # Hole scale / clearance (from env.cfg.clearance)
        clearance = ObsTerm(func=mdp.clearance)  # (N, 1)

        # # One-hot color code derived from hole's diffuse_color: R=[1,0,0], G=[0,1,0], B=[0,0,1]
        # color_code = ObsTerm(func=mdp.color_code)  # (N, 3)


        ##             Auxiliary data           ##
        # hole position in world frame (relative to env origin)
        hole_pos_world = ObsTerm(
            func=mdp.hole_pos_in_world,
            params={"hole_cfg": SceneEntityCfg("hole")},
        )  # (N, 3)

        # hole orientation (quaternion) in world frame
        hole_quat_world = ObsTerm(
            func=mdp.hole_quat_in_world,
            params={"hole_cfg": SceneEntityCfg("hole")},
        )  # (N, 4)

        hole_yaw_world = ObsTerm(
            func=mdp.hole_yaw_in_world, 
            params={"hole_cfg": SceneEntityCfg("hole")}
        )  # (N, 1)

        # peg position in world frame (relative to env origin)
        peg_pos_world = ObsTerm(
            func=mdp.peg_pos_in_world,
            params={"peg_cfg": SceneEntityCfg("peg")},
        )  # (N, 3)

        # peg orientation (quaternion) in world frame
        peg_quat_world = ObsTerm(
            func=mdp.peg_quat_in_world,
            params={"peg_cfg": SceneEntityCfg("peg")},
        )  # (N, 4)

        

        # actions = ObsTerm(func=mdp.last_action)
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)


        # ##            Self-defined data             ##
        # # 1.End-effector wiggle index: cumulative x-y displacement
        # # Fomula: wiggle = sum_t sqrt((x_t - x_{t-1})^2 + (y_t - y_{t-1})^2)
        # wiggle_index = ObsTerm(func=mdp.ee_wiggle_index)  # (N, 1)

        # # 2.After-contact end-effector wiggle index: cumulative x-y displacement after contact
        # # Contact condition: peg_bottom_z - hole_top_z <= 0.01
        # # Formula: after_contact_wiggle = sum_t[from contact start to end] sqrt((x_t - x_{t-1})^2 + (y_t - y_{t-1})^2)
        # after_contact_wiggle_index = ObsTerm(func=mdp.ee_after_contact_wiggle_index)  # (N, 1)

        # # # Whether the peg has contacted the hole (0.0 or 1.0)
        # # has_contacted = ObsTerm(func=mdp.has_contacted)  # (N, 1)


        # # 3.Completion time-step: episode_step_count when success was first detected (-1 if not yet)
        # completion_step = ObsTerm(func=mdp.completion_step)  # (N, 1)

        # # # Whether the task has succeeded (0.0 or 1.0)
        # # has_succeeded = ObsTerm(func=mdp.has_succeeded)  # (N, 1)

        # # 4.Trajectory pause count: number of times EE was stationary for 5 consecutive steps
        # # A pause is when ee_pos and ee_quat have no change for 5 consecutive time-steps
        # pause_count = ObsTerm(func=mdp.pause_count)  # (N, 1)

        # # 5.Mean push speed: average vertical speed during successful insertion
        # # Formula: mean_push_speed = (success_ee_pos_z - start_ee_pos_z) / completion_step
        # # Returns 0 if not yet succeeded
        # mean_push_speed = ObsTerm(func=mdp.mean_push_speed)  # (N, 1)

        # 6. Nullspace posture control
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)


        # to hand gripper open/close to the policy as well (not needed here)
        # gripper_pos = ObsTerm(func=mdp.gripper_pos)


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        # left empty: add an mdp.rgb_image term once a camera sensor is attached
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # success (the three conditions from the spec)
    success = DoneTerm(
        func=mdp.peg_inserted_success,
        params={
            "peg_cfg": SceneEntityCfg("peg"),
            "hole_cfg": SceneEntityCfg("hole"),
        },
    )


@configclass
class InsertEnvCfg(ManagerBasedRLEnvCfg):
    scene: InsertSceneCfg = InsertSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # managers not used by this task
    commands = None
    rewards = None
    events = None
    curriculum = None

    # for a 20-30 Hz controller: sim at 100 Hz, decimation 4-5
    def __post_init__(self):

        ### Setting 1: 25 Hz control
        # self.sim.dt = 0.01             # 100 Hz physics
        # # self.decimation = 4          # 100Hz / 4 = 25Hz
        # self.decimation = 2          # 100Hz / 2 = 50Hz
        # self.sim.render_interval = self.decimation
        # self.episode_length_s = 10.0 # insertion does not need long episodes; keep it short while debugging
        

        # ### Setting 2: 50 Hz control (recorded with record_demos.py --step_hz 50)
        # self.sim.dt = 0.005             # 200 Hz physics
        # self.decimation = 4             # 200 / 4 = 50 Hz control
        # self.sim.render_interval = self.decimation  # camera observation = 50 Hz
        # self.episode_length_s = 10.0    # at most 50 * 10 time-steps; insertion usually needs 5-15 s


        ### Setting 3: 25 Hz control (matches record_demos.py at 25 Hz)
        self.sim.dt = 0.005                 # 1 / 0.005 = 200 Hz physics
        self.decimation = 8                 # control/action: 25 Hz  => decimation = 200 / 25 = 8
                                            # env.step advances 8 physics steps => 25 Hz control
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0

        # physx stability settings (as in the stack task)
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
