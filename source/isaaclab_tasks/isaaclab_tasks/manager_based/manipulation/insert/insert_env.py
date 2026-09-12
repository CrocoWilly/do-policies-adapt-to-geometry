# insert_env.py (or place it in insert/__init__.py)

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from . import mdp


class PegInsertEnv(ManagerBasedRLEnv):
    """Peg insertion environment with custom state tracking."""

    # # Peg length for computing peg bottom position (should match your asset)
    # PEG_LENGTH: float = 0.12
    # # Hole height for computing hole top position (should match your asset)
    # HOLE_HEIGHT: float = 0.08
    # # Contact threshold: peg_bottom_z - hole_top_z <= this value means contact
    # CONTACT_THRESHOLD: float = 0.01
    # # Pause detection: number of consecutive steps with no change
    # PAUSE_WINDOW: int = 5
    # # Threshold for considering position/orientation as "no change"
    # PAUSE_POS_THRESHOLD: float = 1e-4  # meters
    # PAUSE_QUAT_THRESHOLD: float = 1e-4  # quaternion difference

    # def __init__(self, cfg, render_mode=None, **kwargs):
    #     # Initialize tracking buffers BEFORE super().__init__() 
    #     # because parent's __init__ may trigger observation computation
    #     self._prev_ee_pos_xy: torch.Tensor | None = None
    #     self._wiggle_accumulator: torch.Tensor | None = None
    #     # After-contact wiggle tracking
    #     self._after_contact_wiggle_accumulator: torch.Tensor | None = None
    #     self._has_contacted: torch.Tensor | None = None  # bool tensor to track if contact started
    #     # Episode time-step counter
    #     self._episode_step_count: torch.Tensor | None = None
    #     # Completion tracking: record step count when success is first detected
    #     self._completion_step: torch.Tensor | None = None  # step count at success
    #     self._has_succeeded: torch.Tensor | None = None  # bool tensor to track if success occurred
    #     # Pause detection tracking
    #     self._pause_count: torch.Tensor | None = None  # count of detected pauses
    #     self._ee_pos_history: torch.Tensor | None = None  # (N, PAUSE_WINDOW, 3) history buffer
    #     self._ee_quat_history: torch.Tensor | None = None  # (N, PAUSE_WINDOW, 4) history buffer
    #     self._history_idx: int = 0  # circular buffer index
    #     self._history_filled: torch.Tensor | None = None  # whether history buffer is filled
    #     # Mean push speed tracking
    #     self._start_ee_pos_z: torch.Tensor | None = None  # ee_pos_z at episode start
    #     self._success_ee_pos_z: torch.Tensor | None = None  # ee_pos_z at success time
        
    #     # Now call parent's __init__
    #     super().__init__(cfg, render_mode, **kwargs)

    # def _reset_idx(self, env_ids: torch.Tensor):
    #     """Reset environments and clear wiggle state for reset envs."""
    #     super()._reset_idx(env_ids)
    #     # Reset wiggle accumulator for the reset environments
    #     if self._wiggle_accumulator is not None:
    #         self._wiggle_accumulator[env_ids] = 0.0
    #     if self._after_contact_wiggle_accumulator is not None:
    #         self._after_contact_wiggle_accumulator[env_ids] = 0.0
    #     if self._has_contacted is not None:
    #         self._has_contacted[env_ids] = False
    #     # Reset episode step counter
    #     if self._episode_step_count is not None:
    #         self._episode_step_count[env_ids] = 0
    #     # Reset completion tracking
    #     if self._completion_step is not None:
    #         self._completion_step[env_ids] = -1  # -1 means not yet completed
    #     if self._has_succeeded is not None:
    #         self._has_succeeded[env_ids] = False
    #     # Reset pause tracking
    #     if self._pause_count is not None:
    #         self._pause_count[env_ids] = 0
    #     if self._ee_pos_history is not None:
    #         self._ee_pos_history[env_ids] = 0.0
    #     if self._ee_quat_history is not None:
    #         self._ee_quat_history[env_ids] = 0.0
    #     if self._history_filled is not None:
    #         self._history_filled[env_ids] = False
    #     # Reset mean push speed tracking and record start position
    #     if self._start_ee_pos_z is not None:
    #         ee_frame = self.scene["ee_frame"]
    #         ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]  # (N, 3)
    #         self._start_ee_pos_z[env_ids] = ee_pos_w[env_ids, 2].clone()
    #     if self._success_ee_pos_z is not None:
    #         self._success_ee_pos_z[env_ids] = 0.0  # will be set when success detected
    #     if self._prev_ee_pos_xy is not None:
    #         # Get current EE position for reset envs
    #         ee_frame = self.scene["ee_frame"]
    #         ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]  # (N, 3)
    #         self._prev_ee_pos_xy[env_ids] = ee_pos_w[env_ids, :2].clone()

    def post_physics_step(self):
        super().post_physics_step()

        # peg follows the ee every step
        mdp.follow_peg_to_ee(
            self,
            peg_cfg=SceneEntityCfg("peg"),
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
            peg_offset_pos=(0.0, 0.0, 0.0),
            peg_offset_yaw=0.0,
        )

        # # Update wiggle index
        # self._update_wiggle_index()

    # def _check_success(self) -> torch.Tensor:
    #     """Check if peg insertion is successful (same logic as termination).
        
    #     Success conditions:
    #     1. peg center z < hole top z
    #     2. inserted depth >= 0.5 * peg_length
    #     3. xy distance <= peg_width
        
    #     Returns:
    #         torch.Tensor: (N,) bool tensor, True if success condition is met.
    #     """
    #     peg = self.scene["peg"]
    #     hole = self.scene["hole"]
        
    #     peg_pos = peg.data.root_pos_w       # (N,3)
    #     hole_pos = hole.data.root_pos_w     # (N,3)
        
    #     # Geometry parameters (matching termination function)
    #     peg_length = self.PEG_LENGTH
    #     peg_width = 0.04  # peg radius
    #     z_eps = 1e-4
        
    #     hole_center_z = hole_pos[:, 2] + (0.5 * self.HOLE_HEIGHT)
    #     hole_top_z = hole_center_z + (0.5 * self.HOLE_HEIGHT)
    #     peg_center_z = peg_pos[:, 2]
    #     peg_bottom_z = peg_center_z - (0.5 * peg_length)
        
    #     # Success conditions
    #     inserted_depth = hole_top_z - peg_bottom_z
    #     success_center_below_top = peg_center_z < (hole_top_z - z_eps)
    #     success_depth = inserted_depth >= (0.5 * peg_length)
        
    #     xy_dist = torch.linalg.norm(peg_pos[:, :2] - hole_pos[:, :2], dim=1)
    #     success_xy = xy_dist <= peg_width
        
    #     return success_center_below_top & success_depth & success_xy

    # def _check_and_record_success(self):
    #     """Check for success and record completion step for newly successful envs."""
    #     current_success = self._check_success()  # (N,) bool
        
    #     # Find envs that just succeeded (success now but not before)
    #     newly_succeeded = current_success & (~self._has_succeeded)
        
    #     # Record completion step and ee_pos_z for newly succeeded envs
    #     if newly_succeeded.any():
    #         self._completion_step[newly_succeeded] = self._episode_step_count[newly_succeeded]
    #         # Record ee_pos_z at success time
    #         ee_frame = self.scene["ee_frame"]
    #         ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]  # (N, 3)
    #         self._success_ee_pos_z[newly_succeeded] = ee_pos_w[newly_succeeded, 2].clone()
        
    #     # Update has_succeeded (once succeeded, always succeeded)
    #     self._has_succeeded = self._has_succeeded | current_success

    # def _check_contact(self) -> torch.Tensor:
    #     """Check if peg has contacted hole (peg_bottom_z - hole_top_z <= threshold).
        
    #     Returns:
    #         torch.Tensor: (N,) bool tensor, True if contact condition is met.
    #     """
    #     peg = self.scene["peg"]
    #     hole = self.scene["hole"]
        
    #     peg_pos = peg.data.root_pos_w       # (N,3)
    #     hole_pos = hole.data.root_pos_w     # (N,3)

    #     hole_center_z = hole_pos[:, 2] + (0.5 * self.HOLE_HEIGHT)        # hole pos sits on the hole bottom face
    #     hole_top_z = hole_center_z + (0.5 * self.HOLE_HEIGHT)
    #     peg_center_z = peg_pos[:, 2]                                # peg pos sits at the peg centre
    #     peg_bottom_z = peg_center_z - (0.5 * self.PEG_LENGTH)

    #     # Contact condition: peg_bottom_z - hole_top_z <= threshold
    #     contact = (peg_bottom_z - hole_top_z) <= self.CONTACT_THRESHOLD
    #     return contact

    # def _update_wiggle_index(self):
    #     """Update both total wiggle index and after-contact wiggle index."""
    #     ee_frame = self.scene["ee_frame"]
    #     ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]  # (N, 3)
    #     ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]  # (N, 4)
    #     ee_pos_xy = ee_pos_w[:, :2]  # (N, 2) - only x, y

    #     # Initialize buffers on first call
    #     if self._prev_ee_pos_xy is None:
    #         self._prev_ee_pos_xy = ee_pos_xy.clone()
    #         self._wiggle_accumulator = torch.zeros(self.num_envs, device=self.device)
    #         self._after_contact_wiggle_accumulator = torch.zeros(self.num_envs, device=self.device)
    #         self._has_contacted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    #         self._episode_step_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
    #         self._completion_step = torch.full((self.num_envs,), -1, dtype=torch.int32, device=self.device)
    #         self._has_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    #         # Initialize pause tracking buffers
    #         self._pause_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
    #         self._ee_pos_history = torch.zeros(self.num_envs, self.PAUSE_WINDOW, 3, device=self.device)
    #         self._ee_quat_history = torch.zeros(self.num_envs, self.PAUSE_WINDOW, 4, device=self.device)
    #         self._history_filled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    #         # Initialize mean push speed tracking
    #         self._start_ee_pos_z = ee_pos_w[:, 2].clone()
    #         self._success_ee_pos_z = torch.zeros(self.num_envs, device=self.device)
    #         return

    #     # Increment step counter for all envs
    #     self._episode_step_count += 1

    #     # Check for success and record completion step (only first time)
    #     self._check_and_record_success()

    #     # Update pause detection
    #     self._update_pause_count(ee_pos_w, ee_quat_w)

    #     # Compute displacement in x-y plane
    #     delta_xy = ee_pos_xy - self._prev_ee_pos_xy  # (N, 2)
    #     displacement = torch.sqrt((delta_xy**2).sum(dim=-1))  # (N,)

    #     # Accumulate total wiggle
    #     self._wiggle_accumulator += displacement

    #     # Check contact and update after-contact wiggle
    #     current_contact = self._check_contact()  # (N,) bool
    #     # Once contacted, always considered "has_contacted"
    #     self._has_contacted = self._has_contacted | current_contact
        
    #     # Only accumulate after-contact wiggle for envs that have contacted
    #     self._after_contact_wiggle_accumulator += displacement * self._has_contacted.float()

    #     # Update previous position
    #     self._prev_ee_pos_xy = ee_pos_xy.clone()

    # def get_wiggle_index(self) -> torch.Tensor:
    #     """Get the current wiggle index for all environments.
        
    #     Returns:
    #         torch.Tensor: (N, 1) - accumulated wiggle index.
    #     """
    #     if self._wiggle_accumulator is None:
    #         return torch.zeros(self.num_envs, 1, device=self.device)
    #     return self._wiggle_accumulator.unsqueeze(-1)

    # def get_after_contact_wiggle_index(self) -> torch.Tensor:
    #     """Get the after-contact wiggle index for all environments.
        
    #     This only accumulates wiggle after the peg has first contacted the hole
    #     (peg_bottom_z - hole_top_z <= 0.01).
        
    #     Returns:
    #         torch.Tensor: (N, 1) - accumulated after-contact wiggle index.
    #     """
    #     if self._after_contact_wiggle_accumulator is None:
    #         return torch.zeros(self.num_envs, 1, device=self.device)
    #     return self._after_contact_wiggle_accumulator.unsqueeze(-1)

    # def get_has_contacted(self) -> torch.Tensor:
    #     """Get whether each environment has contacted (for debugging).
        
    #     Returns:
    #         torch.Tensor: (N, 1) - bool tensor indicating if contact has occurred.
    #     """
    #     if self._has_contacted is None:
    #         return torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
    #     return self._has_contacted.unsqueeze(-1)

    # def get_time_stamp(self) -> torch.Tensor:
    #     """Get the current time stamp (step count) for all environments.
        
    #     This counts the number of time-steps elapsed since the episode started.
        
    #     Returns:
    #         torch.Tensor: (N, 1) - time stamp for each environment.
    #     """
    #     if self._episode_step_count is None:
    #         return torch.zeros(self.num_envs, 1, dtype=torch.int32, device=self.device)
    #     return self._episode_step_count.unsqueeze(-1)

    # def get_completion_step(self) -> torch.Tensor:
    #     """Get the completion time-step for all environments.
        
    #     This is the episode_step_count at which success was first detected.
    #     Returns -1 for environments that haven't succeeded yet.
        
    #     Returns:
    #         torch.Tensor: (N, 1) - completion step for each environment.
    #     """
    #     if self._completion_step is None:
    #         return torch.full((self.num_envs, 1), -1, dtype=torch.int32, device=self.device)
    #     return self._completion_step.unsqueeze(-1)

    # def get_has_succeeded(self) -> torch.Tensor:
    #     """Get whether each environment has succeeded.
        
    #     Returns:
    #         torch.Tensor: (N, 1) - bool tensor indicating if success has occurred.
    #     """
    #     if self._has_succeeded is None:
    #         return torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
    #     return self._has_succeeded.unsqueeze(-1)

    # def _update_pause_count(self, ee_pos_w: torch.Tensor, ee_quat_w: torch.Tensor):
    #     """Update pause count by checking if EE has been stationary for PAUSE_WINDOW steps.
        
    #     A pause is detected when ee_pos and ee_quat have no change for PAUSE_WINDOW consecutive steps.
        
    #     Args:
    #         ee_pos_w: (N, 3) current EE position in world frame
    #         ee_quat_w: (N, 4) current EE quaternion in world frame
    #     """
    #     # Store current pose in circular buffer
    #     self._ee_pos_history[:, self._history_idx, :] = ee_pos_w
    #     self._ee_quat_history[:, self._history_idx, :] = ee_quat_w
        
    #     # Update circular buffer index
    #     next_idx = (self._history_idx + 1) % self.PAUSE_WINDOW
        
    #     # Mark history as filled once we've gone through all slots
    #     if next_idx == 0:
    #         self._history_filled[:] = True
        
    #     # Only check for pause if history buffer is filled
    #     if self._history_filled.any():
    #         # Get envs with filled history
    #         filled_envs = self._history_filled
            
    #         # Check if all positions in history are the same (max - min < threshold)
    #         pos_history = self._ee_pos_history[filled_envs]  # (M, PAUSE_WINDOW, 3)
    #         pos_max = pos_history.max(dim=1).values  # (M, 3)
    #         pos_min = pos_history.min(dim=1).values  # (M, 3)
    #         pos_range = (pos_max - pos_min).abs().max(dim=1).values  # (M,)
    #         pos_stationary = pos_range < self.PAUSE_POS_THRESHOLD
            
    #         # Check if all quaternions in history are the same
    #         quat_history = self._ee_quat_history[filled_envs]  # (M, PAUSE_WINDOW, 4)
    #         quat_max = quat_history.max(dim=1).values  # (M, 4)
    #         quat_min = quat_history.min(dim=1).values  # (M, 4)
    #         quat_range = (quat_max - quat_min).abs().max(dim=1).values  # (M,)
    #         quat_stationary = quat_range < self.PAUSE_QUAT_THRESHOLD
            
    #         # Pause detected if both position and orientation are stationary
    #         is_paused = pos_stationary & quat_stationary  # (M,)
            
    #         # Increment pause count for paused envs (only count once per pause detection)
    #         # To avoid counting the same pause multiple times, we only increment when
    #         # transitioning from non-paused to paused state at the end of PAUSE_WINDOW
    #         if next_idx == 0:  # Only count at the end of each window
    #             pause_increment = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
    #             pause_increment[filled_envs] = is_paused.int()
    #             self._pause_count += pause_increment
        
    #     self._history_idx = next_idx

    # def get_pause_count(self) -> torch.Tensor:
    #     """Get the pause count for all environments.
        
    #     A pause is when ee_pos and ee_quat have no change for PAUSE_WINDOW consecutive steps.
        
    #     Returns:
    #         torch.Tensor: (N, 1) - pause count for each environment.
    #     """
    #     if self._pause_count is None:
    #         return torch.zeros(self.num_envs, 1, dtype=torch.int32, device=self.device)
    #     return self._pause_count.unsqueeze(-1)

    # def get_mean_push_speed(self) -> torch.Tensor:
    #     """Get the mean push speed for all environments.
        
    #     mean_push_speed = (success_ee_pos_z - start_ee_pos_z) / completion_step
        
    #     This measures the average vertical speed during the insertion task.
    #     Returns 0 for environments that haven't succeeded yet.
        
    #     Returns:
    #         torch.Tensor: (N, 1) - mean push speed for each environment.
    #     """
    #     if self._start_ee_pos_z is None or self._success_ee_pos_z is None or self._completion_step is None:
    #         return torch.zeros(self.num_envs, 1, device=self.device)
        
    #     # Calculate delta_z (start - success = positive for downward push)
    #     # Since insertion moves downward (z decreases), start_z > success_z
    #     delta_z = self._start_ee_pos_z - self._success_ee_pos_z  # (N,), positive for downward motion
        
    #     # Avoid division by zero: use completion_step, but set to 1 where not succeeded
    #     completion_step_safe = self._completion_step.float().clone()
    #     not_succeeded = ~self._has_succeeded
    #     completion_step_safe[not_succeeded] = 1.0  # avoid div by zero
        
    #     # Calculate mean push speed (positive value)
    #     mean_speed = delta_z / completion_step_safe  # (N,)
        
    #     # Set to 0 for envs that haven't succeeded
    #     mean_speed[not_succeeded] = 0.0
        
    #     return mean_speed.unsqueeze(-1)
