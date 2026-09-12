"""Differential IK action terms with nullspace posture control.

Reference implementation, not used by the published experiments — those run the
action term in ``actions.py``. It is kept because the problem it solves recurs
whenever a redundant arm is driven by a low-dimensional action.

The Franka has seven joints; the insertion task commands four degrees of
freedom. The remaining three form a nullspace the controller is otherwise free
to wander in, and over a long alignment the arm drifts into awkward postures
even though the end-effector tracks its target correctly.

``DiffIKWithNullspaceAction`` adds a projection that steers those redundant
joints towards a target configuration without disturbing end-effector tracking.
``Se3Delta4ActionTerm`` is the matching 4D wrapper.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Sequence
import torch
import isaaclab.utils.math as math_utils

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.actions import ActionTerm
from isaaclab.envs.mdp.actions.actions_cfg import ActionTermCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.assets import Articulation
from isaaclab.utils import configclass


# =============================================================================
# DifferentialIK with Nullspace Posture Control
# =============================================================================

@configclass
class DiffIKWithNullspaceActionCfg(ActionTermCfg):
    """DifferentialIK action with nullspace posture control.
    
    This action adds a nullspace projection on top of DiffIK, steering the
    redundant degrees of freedom towards a target joint configuration (the
    joint centres, say) without disturbing end-effector tracking.
    """
    class_type: type = None  # set in __post_init__

    asset_name: str = "robot"
    joint_names: list[str] = ["panda_joint.*"]
    body_name: str = "panda_hand"
    
    controller: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
        ik_params={"lambda_val": 0.01},
    )
    
    scale: float = 1.0
    
    # Body offset (for EE frame)
    @configclass
    class OffsetCfg:
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    
    body_offset: OffsetCfg | None = None
    
    # Nullspace control settings
    nullspace_control: Literal["none", "position"] = "position"
    """Nullspace control mode: 'none' or 'position'"""
    
    nullspace_stiffness: float = 0.5
    """Nullspace posture stiffness (small value to avoid interfering with task)"""
    
    nullspace_target: Literal["center", "default", "zero"] = "center"
    """Target for nullspace: 'center' (joint limits center), 'default' (initial pose), 'zero'"""

    def __post_init__(self):
        self.class_type = DiffIKWithNullspaceAction


class DiffIKWithNullspaceAction(ActionTerm):
    """DifferentialIK action term with nullspace posture control.
    
    Formulation:
    Δq = J⁺ @ Δx + (I - J⁺J) @ (k * (q_target - q_current))
    
    where:
    - J+ @ dx: the primary task, end-effector pose tracking
    - (I - J⁺J)：nullspace projector
    - k * (q_target - q_current): the nullspace objective, driving the joints
    """

    cfg: DiffIKWithNullspaceActionCfg

    def __init__(self, cfg: DiffIKWithNullspaceActionCfg, env):
        super().__init__(cfg, env)
        
        self._asset: Articulation = env.scene[cfg.asset_name]
        
        # Resolve joint indices
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        
        # Resolve body index
        body_ids, _ = self._asset.find_bodies(cfg.body_name)
        self._body_idx = body_ids[0]
        
        # For Jacobian extraction: handle fixed-base vs floating-base
        # For fixed-base robots, jacobian doesn't include base, so index is body_idx - 1
        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]
        
        # Create IK controller
        self._ik_controller = DifferentialIKController(cfg.controller, env.num_envs, env.device)
        
        # Action buffers
        self._raw_actions = torch.zeros(env.num_envs, self.action_dim, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, self._num_joints, device=env.device)
        
        # EE state buffers
        self._ee_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        self._ee_quat_w = torch.zeros(env.num_envs, 4, device=env.device)
        self._ee_pos_b = torch.zeros(env.num_envs, 3, device=env.device)
        self._ee_quat_b = torch.zeros(env.num_envs, 4, device=env.device)
        
        # Offset handling
        if cfg.body_offset is not None:
            self._offset_pos = torch.tensor(cfg.body_offset.pos, device=env.device).unsqueeze(0)
            self._offset_rot = torch.tensor(cfg.body_offset.rot, device=env.device).unsqueeze(0)
        else:
            self._offset_pos = None
            self._offset_rot = None
        
        # Nullspace target
        self._nullspace_target = None
        self._setup_nullspace_target()

    def _setup_nullspace_target(self):
        """Setup nullspace target joint positions."""
        if self.cfg.nullspace_control == "none":
            return
            
        if self.cfg.nullspace_target == "center":
            # Center of joint limits
            limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]
            self._nullspace_target = (limits[:, :, 0] + limits[:, :, 1]) / 2.0
        elif self.cfg.nullspace_target == "default":
            # Will be set in reset() based on default_joint_pos
            self._nullspace_target = None
        elif self.cfg.nullspace_target == "zero":
            self._nullspace_target = torch.zeros(
                self._env.num_envs, self._num_joints, device=self.device
            )

    @property
    def action_dim(self) -> int:
        return self._ik_controller.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        # Update nullspace target for "default" mode
        if self.cfg.nullspace_target == "default" and self.cfg.nullspace_control == "position":
            self._nullspace_target = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions * self.cfg.scale
        
        # Update EE pose
        self._compute_ee_pose()
        
        # Set command to IK controller
        self._ik_controller.set_command(
            command=self._raw_actions,
            ee_pos=self._ee_pos_b,
            ee_quat=self._ee_quat_b,
        )

    def apply_actions(self):
        # Get current joint positions
        current_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        
        # Get Jacobian
        jacobian = self._get_jacobian_b()
        
        # Compute IK solution (primary task)
        target_joint_pos = self._ik_controller.compute(
            ee_pos=self._ee_pos_b,
            ee_quat=self._ee_quat_b,
            jacobian=jacobian,
            joint_pos=current_joint_pos,
        )
        
        # Add nullspace component
        if self.cfg.nullspace_control == "position" and self._nullspace_target is not None:
            # Compute nullspace projector: N = I - J⁺J
            jacobian_pinv = torch.linalg.pinv(jacobian)
            identity = torch.eye(self._num_joints, device=self.device).unsqueeze(0).expand(
                self._env.num_envs, -1, -1
            )
            nullspace_projector = identity - jacobian_pinv @ jacobian
            
            # Nullspace velocity: k * (q_target - q_current)
            nullspace_vel = self.cfg.nullspace_stiffness * (self._nullspace_target - current_joint_pos)
            
            # Project into nullspace and add to target
            nullspace_delta = (nullspace_projector @ nullspace_vel.unsqueeze(-1)).squeeze(-1)
            target_joint_pos = target_joint_pos + nullspace_delta
        
        # Store processed actions
        self._processed_actions[:] = target_joint_pos
        
        # Apply to robot
        self._asset.set_joint_position_target(target_joint_pos, joint_ids=self._joint_ids)

    def _compute_ee_pose(self):
        """Compute EE pose in base frame."""
        # Get body pose in world frame
        self._ee_pos_w[:] = self._asset.data.body_pos_w[:, self._body_idx]
        self._ee_quat_w[:] = self._asset.data.body_quat_w[:, self._body_idx]
        
        # Transform to base frame
        self._ee_pos_b[:], self._ee_quat_b[:] = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w,
            self._asset.data.root_quat_w,
            self._ee_pos_w,
            self._ee_quat_w,
        )
        
        # Apply offset if configured
        if self._offset_pos is not None:
            self._ee_pos_b[:], self._ee_quat_b[:] = math_utils.combine_frame_transforms(
                self._ee_pos_b, self._ee_quat_b,
                self._offset_pos.expand(self._env.num_envs, -1),
                self._offset_rot.expand(self._env.num_envs, -1),
            )

    def _get_jacobian_b(self) -> torch.Tensor:
        """Get Jacobian in base frame."""
        jacobian_w = self._asset.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._jacobi_joint_ids
        ]
        # Transform to base frame
        base_rot_matrix = math_utils.matrix_from_quat(
            math_utils.quat_inv(self._asset.data.root_quat_w)
        )
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])
        return jacobian_b


# =============================================================================
# Original Se3Delta4 Action (kept for compatibility)
# =============================================================================

@configclass
class Se3Delta4ActionCfg(ActionTermCfg):
    """Policy action: [dx, dy, dz, dyaw] -> relative SE(3) delta pose for IK."""
    class_type: type = None  # filled in by __post_init__ below

    asset_name: str = "robot"
    body_name: str = "panda_hand"

    # scaling
    scale_xyz: float = 0.01
    scale_yaw: float = 0.10

    fixed_roll_pitch: bool = True

    # injected by the IK env cfg (see insert_ik_rel_env_cfg.py)
    controller = None
    ik_action_cfg = None

    def __post_init__(self):
        self.class_type = Se3Delta4ActionTerm


class Se3Delta4ActionTerm(ActionTerm):
    def __init__(self, cfg: Se3Delta4ActionCfg, env):
        super().__init__(cfg, env)
        self._cfg = cfg

        # delegates internally to the real DiffIK action term
        # cfg.ik_action_cfg is a DifferentialInverseKinematicsActionCfg
        if cfg.ik_action_cfg is None:
            raise RuntimeError("Se3Delta4ActionCfg.ik_action_cfg is None. Use the IK env cfg to set it.")
        self._ik_term = cfg.ik_action_cfg.class_type(cfg.ik_action_cfg, env)

        # cache
        self._action = torch.zeros((env.num_envs, 4), device=env.device)

    @property
    def action_dim(self) -> int:
        return 4

    def process_actions(self, actions: torch.Tensor):
        self._action[:] = actions

    def apply_actions(self):
        # actions: (N,4)
        dx = self._action[:, 0:1] * self._cfg.scale_xyz
        dy = self._action[:, 1:2] * self._cfg.scale_xyz
        dz = self._action[:, 2:3] * self._cfg.scale_xyz
        dyaw = self._action[:, 3:4] * self._cfg.scale_yaw

        # DiffIK relative pose command typically expects (N,7): [pos(3), quat(4)] delta
        # roll/pitch fixed => dyaw about the z axis only
        # delta rotation quaternion from yaw
        zeros = torch.zeros_like(dyaw)
        delta_quat = math_utils.quat_from_euler_xyz(zeros, zeros, dyaw.squeeze(-1)).to(dyaw.device)
        delta_pos = torch.cat([dx, dy, dz], dim=-1)

        ik_cmd = torch.cat([delta_pos, delta_quat], dim=-1)  # (N,7)

        self._ik_term.process_actions(ik_cmd)
        self._ik_term.apply_actions()
