# isaaclab_tasks/manager_based/manipulation/insert/mdp/actions.py

from __future__ import annotations
from dataclasses import dataclass
import torch
import isaaclab.utils.math as math_utils

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.actions import ActionTerm
from isaaclab.envs.mdp.actions.actions_cfg import ActionTermCfg
from isaaclab.utils import configclass


# =============================================================================
# NEW: Action in hole's local frame
# =============================================================================


# Define policy action: [Δee_hole_x, Δee_hole_y, Δee_hole_z, Δee_hole_yaw] in hole's local frame
@configclass
class Se3Delta4InHoleFrameActionCfg(ActionTermCfg):
    """Policy action: [Δee_hole_x, Δee_hole_y, Δee_hole_z, Δee_hole_yaw] in hole's local frame.
    
    The action represents the desired change of EE position/yaw relative to the hole.
    This delta is then transformed to world frame for IK execution.
    
    Note: This action term wraps DifferentialInverseKinematicsAction and expects 
    the controller to use `command_type="pose"` with `use_relative_mode=True`.
    The DiffIK controller expects (N, 6) input: [dx, dy, dz, axis_angle_x, axis_angle_y, axis_angle_z].
    """
    class_type: type = None

    asset_name: str = "robot"
    body_name: str = "panda_hand"
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole")

    # scaling
    scale_xy  = 0.0035   # meters
    scale_z   = 0.0035   # meters
    scale_yaw = 0.01     # radians

    # injected by the IK env cfg
    controller = None
    ik_action_cfg = None

    def __post_init__(self):
        self.class_type = Se3Delta4InHoleFrameActionTerm


class Se3Delta4InHoleFrameActionTerm(ActionTerm):
    """Action term where [dx, dy, dz, dyaw] are defined in hole's local coordinate frame.
    
    - dx, dy, dz: position delta in hole's local frame
    - dyaw: yaw rotation delta relative to hole's yaw
    
    These are transformed to world frame before being sent to the IK controller.
    
    The IK controller (DifferentialInverseKinematicsAction with pose_rel mode) expects:
    - (N, 6): [dx, dy, dz, axis_angle_x, axis_angle_y, axis_angle_z] in controller/task frame (typically world/base frame)
    
    Since we assume the EE roll/pitch are fixed (pointing down), dyaw corresponds to rotation around the hole’s local z-axis, 
    expressed in the world frame. (axis_angle = [0, 0, dyaw]).
    """
    
    def __init__(self, cfg: Se3Delta4InHoleFrameActionCfg, env):
        super().__init__(cfg, env)
        self._cfg = cfg
        self._env = env

        # delegates internally to the real DiffIK action term
        if cfg.ik_action_cfg is None:
            raise RuntimeError("Se3Delta4InHoleFrameActionCfg.ik_action_cfg is None.")
        self._ik_term = cfg.ik_action_cfg.class_type(cfg.ik_action_cfg, env)

        # cache
        self._raw_actions = torch.zeros((env.num_envs, 4), device=env.device)
        self._processed_actions = torch.zeros((env.num_envs, 4), device=env.device)

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        """The raw actions received from the policy."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The processed actions (after clipping/scaling)."""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        assert actions.shape[-1] == 4
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, -1.0, 1.0)

    def apply_actions(self):
        hole: RigidObject = self._env.scene[self._cfg.hole_cfg.name]
        hole_quat_w = hole.data.root_quat_w  # (N,4)

        # (N,)
        dx_h = self._processed_actions[:, 0] * self._cfg.scale_xy
        dy_h = self._processed_actions[:, 1] * self._cfg.scale_xy
        dz_h = self._processed_actions[:, 2] * self._cfg.scale_z
        dyaw = self._processed_actions[:, 3] * self._cfg.scale_yaw

        # (N,3) hole-frame delta
        delta_pos_h = torch.stack([dx_h, dy_h, dz_h], dim=-1)

        # world-frame delta position
        # delta_pos_w = math_utils.quat_rotate(hole_quat_w, delta_pos_h)  # (N,3)  # deprecated
        delta_pos_w = math_utils.quat_apply(hole_quat_w, delta_pos_h)  # (N,3)

        # world-frame rotation vector: axis (hole z in world) * angle
        z_axis_h = delta_pos_h.new_zeros(delta_pos_h.shape)
        z_axis_h[:, 2] = 1.0
        # z_axis_w = math_utils.quat_rotate(hole_quat_w, z_axis_h)        # (N,3)  # deprecated
        z_axis_w = math_utils.quat_apply(hole_quat_w, z_axis_h)        # (N,3)

        delta_rot_w = z_axis_w * dyaw.unsqueeze(-1)                     # (N,3)

        ik_cmd = torch.cat([delta_pos_w, delta_rot_w], dim=-1)          # (N,6)

        self._ik_term.process_actions(ik_cmd)
        self._ik_term.apply_actions()


    def reset(self, env_ids=None):
        """Reset the action term."""
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._processed_actions[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
        self._ik_term.reset(env_ids)



# @configclass
# class Se3Delta4ActionCfg(ActionTermCfg):
#     """Policy action: [dx, dy, dz, dyaw] -> relative SE(3) delta pose for IK."""
#     class_type: type = None  # filled in by __post_init__ below

#     asset_name: str = "robot"
#     body_name: str = "panda_hand"

#     # scaling
#     scale_xyz: float = 0.01
#     scale_yaw: float = 0.10

#     fixed_roll_pitch: bool = True

#     # injected by the IK env cfg (see insert_ik_rel_env_cfg.py)
#     controller = None
#     ik_action_cfg = None

#     def __post_init__(self):
#         self.class_type = Se3Delta4ActionTerm


# class Se3Delta4ActionTerm(ActionTerm):
#     def __init__(self, cfg: Se3Delta4ActionCfg, env):
#         super().__init__(cfg, env)
#         self._cfg = cfg

#         # delegates internally to the real DiffIK action term
#         # cfg.ik_action_cfg is a DifferentialInverseKinematicsActionCfg
#         if cfg.ik_action_cfg is None:
#             raise RuntimeError("Se3Delta4ActionCfg.ik_action_cfg is None. Use the IK env cfg to set it.")
#         self._ik_term = cfg.ik_action_cfg.class_type(cfg.ik_action_cfg, env)

#         # cache
#         self._action = torch.zeros((env.num_envs, 4), device=env.device)

#     @property
#     def action_dim(self) -> int:
#         return 4

#     def process_actions(self, actions: torch.Tensor):
#         self._action[:] = actions

#     def apply_actions(self):
#         # actions: (N,4)
#         dx = self._action[:, 0:1] * self._cfg.scale_xyz
#         dy = self._action[:, 1:2] * self._cfg.scale_xyz
#         dz = self._action[:, 2:3] * self._cfg.scale_xyz
#         dyaw = self._action[:, 3:4] * self._cfg.scale_yaw

#         # DiffIK relative pose command typically expects (N,7): [pos(3), quat(4)] delta
#         # roll/pitch fixed => dyaw about the z axis only
#         # delta rotation quaternion from yaw
#         zeros = torch.zeros_like(dyaw)
#         delta_quat = math_utils.quat_from_euler_xyz(zeros, zeros, dyaw.squeeze(-1)).to(dyaw.device)
#         delta_pos = torch.cat([dx, dy, dz], dim=-1)

#         ik_cmd = torch.cat([delta_pos, delta_quat], dim=-1)  # (N,7)

#         self._ik_term.process_actions(ik_cmd)
#         self._ik_term.apply_actions()
