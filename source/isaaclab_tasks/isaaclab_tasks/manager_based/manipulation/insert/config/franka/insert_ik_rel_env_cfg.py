# isaaclab_tasks/manager_based/manipulation/insert/config/franka/insert_ik_rel_env_cfg.py

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import insert_joint_pos_env_cfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaPegInsertEnvCfg(insert_joint_pos_env_cfg.FrankaPegInsertEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # DiffIK, but the policy action stays 4D; the wrapper turns it into an SE(3) delta
        self.actions.arm_action.controller = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=True, ik_method="dls"
        )
        self.actions.arm_action.ik_action_cfg = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=self.actions.arm_action.controller,
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
