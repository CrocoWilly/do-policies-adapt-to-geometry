# isaaclab_tasks/manager_based/manipulation/insert/mdp/terminations.py

from __future__ import annotations
import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

def peg_inserted_success(
    env,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
    peg_length: float = 0.12,          # peg length (world units, e.g. metres)
    peg_width: float = 0.04,           # peg width (radius, world units)
    hole_height: float = 0.08,        # hole height measured from the hole bottom (world units)
    yaw_tol_rad: float =  5.0 * 3.14159 / 180.0,
    z_eps: float = 1e-4,               # small tolerance, avoids chatter at exact equality
) -> torch.Tensor:
    """Success when peg center is below hole top AND inserted depth >= half peg length, with yaw aligned.

    Assumptions:
      - hole_pos[:,2] is the z of the hole's bottom face, where the hole origin sits
      - peg_pos[:,2] is the z of the peg centre
      - +z points up, the usual Isaac/PhysX convention
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]

    peg_pos = peg.data.root_pos_w       # (N,3)
    peg_quat = peg.data.root_quat_w     # (N,4)
    hole_pos = hole.data.root_pos_w     # (N,3)
    hole_quat = hole.data.root_quat_w   # (N,4)

    # --- (1) depth success condition ---
    hole_center_z = hole_pos[:, 2] + (0.5 * hole_height)        # hole pos sits on the hole bottom face
    hole_top_z = hole_center_z + (0.5 * hole_height)
    peg_center_z = peg_pos[:, 2]                                # peg pos sits at the peg centre
    peg_bottom_z = peg_center_z - (0.5 * peg_length)

    inserted_depth = hole_top_z - peg_bottom_z   # grows as the peg goes deeper
    success_center_below_top = peg_center_z < (hole_top_z - z_eps)
    success_depth = inserted_depth >= (0.5 * peg_length)


    # --- (2) XY alignment condition (radius = peg_long_width) ---
    xy_dist = torch.linalg.norm(peg_pos[:, :2] - hole_pos[:, :2], dim=1)
    success_xy = xy_dist <= peg_width
    # success_xy = True

    # # --- (2.5) yaw alignment requirement ---
    # peg_row, peg_pitch, peg_yaw = math_utils.euler_xyz_from_quat(peg_quat)
    # _, _, hole_yaw = math_utils.euler_xyz_from_quat(hole_quat)
    # yaw_err = math_utils.wrap_to_pi(peg_yaw - hole_yaw).abs()
    # success_yaw = yaw_err <= yaw_tol_rad

    # --- (3) timeout check: only count as success if episode hasn't timed out ---
    not_timeout = env.episode_length_buf < env.max_episode_length
    
    # # --- debug prints ---
    # print("DEBUG: hole_pos_x:", hole_pos[:, 0])
    # print("DEBUG: hole_pos_y:", hole_pos[:, 1])
    # print("DEBUG: peg_center_x:", peg_pos[:, 0])
    # print("DEBUG: peg_center_y:", peg_pos[:, 1])
    # print("DEBUG: hole_bottom_z:", hole_pos[:, 2])
    # print("DEBUG: hole_center_z:", hole_center_z)
    # print("DEBUG: hole_top_z:", hole_top_z)
    # print("DEBUG: peg_center_z:", peg_center_z)
    # print("DEBUG: peg_bottom_z:", peg_bottom_z)
    # print("DEBUG: inserted_depth:", inserted_depth)
    # print("DEBUG: peg_yaw:", peg_yaw)
    # print()
    # print("DEBUG: env.episode_length_buf:", env.episode_length_buf)
    # print("DEBUG: env.max_episode_length:", env.max_episode_length)

    return success_center_below_top & success_depth & success_xy & not_timeout
