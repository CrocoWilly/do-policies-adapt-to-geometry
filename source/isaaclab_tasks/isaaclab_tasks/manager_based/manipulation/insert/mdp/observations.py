# isaaclab_tasks/manager_based/manipulation/insert/mdp/observations.py

from __future__ import annotations
import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


# ee x, y, z in world coordinates
def ee_pos_in_env(env, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]
    return pos

# ee yaw in world coordinates
def ee_yaw_in_env(env, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    quat = ee_frame.data.target_quat_w[:, 0, :]  # world quat
    # convert to euler and take yaw (z)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat)
    return yaw.unsqueeze(-1)


# ee quaternion in world coordinates
def ee_quat_in_env(env, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """EE orientation (quaternion) in world frame.
    
    Returns:
        torch.Tensor: (N, 4) - ee quaternion in world frame (w, x, y, z).
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    quat = ee_frame.data.target_quat_w[:, 0, :]  # world quat (N, 4)
    return quat


# color code: one-hot encoding derived from hole's diffuse_color
def color_code(env) -> torch.Tensor:
    """One-hot color code derived from hole's diffuse_color.

    Based on which RGB channel of diffuse_color is largest:
    - Red   (1st dim largest): [1, 0, 0]
    - Green (2nd dim largest): [0, 1, 0]
    - Blue  (3rd dim largest): [0, 0, 1]

    Returns:
        torch.Tensor: (N, 3) - one-hot color code.
    """
    diffuse_color = env.cfg.scene.hole.spawn.visual_material.diffuse_color
    max_idx = max(range(3), key=lambda i: diffuse_color[i])
    one_hot = [0.0, 0.0, 0.0]
    one_hot[max_idx] = 1.0
    return torch.tensor([one_hot], device=env.device).expand(env.num_envs, -1)


# color number: red=0, green=1, blue=2 derived from hole's diffuse_color
def box_color_no(env) -> torch.Tensor:
    """Color number derived from hole's diffuse_color.

    Based on which RGB channel of diffuse_color is largest:
    - Red   (1st dim largest): 0
    - Green (2nd dim largest): 1
    - Blue  (3rd dim largest): 2

    Returns:
        torch.Tensor: (N, 1) - color number.
    """
    diffuse_color = env.cfg.scene.hole.spawn.visual_material.diffuse_color
    max_idx = max(range(3), key=lambda i: diffuse_color[i])
    return torch.full((env.num_envs, 1), float(max_idx), device=env.device)


# peg texture number: no texture (metal)=0, wood=1
def peg_texture_no(env) -> torch.Tensor:
    """Peg texture number derived from peg's visual_material config.

    - No visual_material (None / metal default): 0
    - Wood/Oak mdl_path: 1

    Returns:
        torch.Tensor: (N, 1) - texture number.
    """
    vis_mat = getattr(env.cfg.scene.peg.spawn, "visual_material", None)
    if vis_mat is None:
        tex_no = 0
    else:
        mdl_path = getattr(vis_mat, "mdl_path", "")
        if "Wood/Oak" in mdl_path:
            tex_no = 1
        else:
            tex_no = 0
    return torch.full((env.num_envs, 1), float(tex_no), device=env.device)


# table texture number: original dark wood=0, Ash light wood=1
def table_texture_no(env) -> torch.Tensor:
    """Table texture number derived from EventCfg.randomize_table_visual_material.

    Checks the 'textures' list in the event config:
    - Empty list (original dark wood table): 0
    - Contains Ash BaseColor path: 1

    Returns:
        torch.Tensor: (N, 1) - texture number.
    """
    tex_list = env.cfg.events.randomize_table_visual_material.params.get("textures", [])
    if not tex_list:
        tex_no = 0
    elif any("Wood/Ash" in t for t in tex_list):
        tex_no = 1
    else:
        tex_no = 0
    return torch.full((env.num_envs, 1), float(tex_no), device=env.device)


# peg shape number derived from peg's usd_path
# Encoding: CirclePeg=0, RealSquarePeg=1, RectanglePeg=2
def peg_shape_no(env) -> torch.Tensor:
    """Peg shape number derived from peg's usd_path.

    - CirclePeg     -> 0
    - RealSquarePeg -> 1
    - RectanglePeg  -> 2

    Returns:
        torch.Tensor: (N, 1) - peg shape number.
    """
    usd_path = env.cfg.scene.peg.spawn.usd_path
    if "CirclePeg" in usd_path:
        shape_no = 0
    elif "RealSquarePeg" in usd_path:
        shape_no = 1
    elif "RectanglePeg" in usd_path:
        shape_no = 2
    else:
        shape_no = -1  # unknown
    return torch.full((env.num_envs, 1), float(shape_no), device=env.device)


# hole shape codes for the 3 slots of ThreeShapeHoleBlock_*.usd
# Slot order: [+0.6cm slot, 0cm (center) slot, -0.6cm slot]
# Shape encoding: circle=0, square=1, rectangle=2
_THREE_SHAPE_HOLE_TABLE = {
    "ThreeShapeHoleBlock_1": (0, 2, 1),  # +0.6: circle, 0: rectangle, -0.6: square
    "ThreeShapeHoleBlock_2": (0, 1, 2),  # +0.6: circle, 0: square, -0.6: rectangle
    "ThreeShapeHoleBlock_3": (2, 0, 1),  # +0.6: rectangle, 0: circle, -0.6: square
    "ThreeShapeHoleBlock_4": (1, 0, 2),  # +0.6: square, 0: circle, -0.6: rectangle
    "ThreeShapeHoleBlock_5": (1, 2, 0),  # +0.6: square, 0: rectangle, -0.6: circle
    "ThreeShapeHoleBlock_6": (2, 1, 0),  # +0.6: rectangle, 0: square, -0.6: circle
}


def hole_shape_codes(env) -> torch.Tensor:
    """Hole shape codes for ThreeShapeHoleBlock_*.usd.

    Returns a (N, 3) tensor encoding the shape at each of the 3 slots
    along the box, ordered as [slot_at_+0.6cm, slot_at_center, slot_at_-0.6cm].

    Shape encoding (matches peg_shape_no):
        circle=0, square=1, rectangle=2

    Returns:
        torch.Tensor: (N, 3) - shape codes for the three slots.
    """
    usd_path = env.cfg.scene.hole.spawn.usd_path
    codes = (-1, -1, -1)
    for key, val in _THREE_SHAPE_HOLE_TABLE.items():
        if key in usd_path:
            codes = val
            break
    codes_tensor = torch.tensor(codes, device=env.device, dtype=torch.float32)
    return codes_tensor.unsqueeze(0).expand(env.num_envs, -1).contiguous()


# Real-world positions of the 3 slot centers on ThreeShapeHoleBlock_*.usd
# Slots are placed along the hole's local +x axis at offsets +0.006 m, 0 m, -0.006 m
# (i.e. +0.6cm, 0cm, -0.6cm from the box center). The slot z-coordinate uses the
# hole top surface (hole_pos_z + hole_height).
def hole_slot_positions_world(
    env,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
    slot_offset: float = 0.06,
    hole_height: float = 0.08,
    slot_axis: str = "y",
) -> torch.Tensor:
    """Real-world (env-relative) center positions of the 3 hole slots.

    Slot order matches ``hole_shape_codes``:
        slot 0: +slot_offset along hole local axis
        slot 1: 0  (center)
        slot 2: -slot_offset along hole local axis

    Each slot center is computed as the box center transformed to world frame,
    using the hole's root pose. The z is lifted to the hole top surface
    (``hole_pos_z + hole_height``).

    Args:
        slot_offset: distance from box center to the outer slots (default 0.006 m).
        hole_height: height of the hole block (default 0.08 m).
        slot_axis: axis in hole local frame along which slots are arranged ("x" or "y").

    Returns:
        torch.Tensor: (N, 9) - 3 slots x (x, y, z) flattened, env-relative.
    """
    hole: RigidObject = env.scene[hole_cfg.name]
    hole_pos_w = hole.data.root_pos_w        # (N, 3) at hole bottom
    hole_quat_w = hole.data.root_quat_w      # (N, 4)

    # local offsets for the 3 slots: top surface (z = +hole_height)
    axis_idx = 0 if slot_axis.lower() == "x" else 1
    offsets_local = torch.zeros((3, 3), device=env.device, dtype=hole_pos_w.dtype)
    offsets_local[0, axis_idx] = +slot_offset
    offsets_local[1, axis_idx] = 0.0
    offsets_local[2, axis_idx] = -slot_offset
    offsets_local[:, 2] = hole_height  # lift to hole top

    # rotate offsets by hole orientation: (N, 3, 3)
    n = hole_pos_w.shape[0]
    quat_rep = hole_quat_w.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 4)
    offsets_flat = offsets_local.unsqueeze(0).expand(n, -1, -1).reshape(-1, 3)
    rotated = math_utils.quat_apply(quat_rep, offsets_flat).reshape(n, 3, 3)

    # world positions of the 3 slot centers
    slot_pos_w = hole_pos_w.unsqueeze(1) + rotated  # (N, 3, 3)
    # convert to env-relative
    slot_pos_env = slot_pos_w - env.scene.env_origins[:, 0:3].unsqueeze(1)
    return slot_pos_env.reshape(env.num_envs, -1)


# hole scale / clearance (read from env.cfg.clearance)
def clearance(env) -> torch.Tensor:
    """Hole scale / clearance value from env config.
    
    This value represents the scale factor of the hole, which determines
    the clearance between the peg and hole. Larger value = more clearance.
    
    Returns:
        torch.Tensor: (N, 1) - clearance value for each environment.
    """
    clearance_val = getattr(env.cfg, "clearance", 1.0)  # default to 1.0 if not set
    return torch.full((env.num_envs, 1), clearance_val, device=env.device)


# ee x, y, z relative to the hole top
def ee_pos_rel_hole(
    env,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
    hole_height: float = 0.08,
) -> torch.Tensor:
    """EE position relative to hole top: [ee_hole_x, ee_hole_y, ee_hole_z].
    
    The reference point is the hole top center (not hole bottom).
    hole_pos_w is at the bottom, so we offset by hole_height to get hole top.
    
    Returns:
        torch.Tensor: (N, 3) - ee position relative to hole top in hole's local frame.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]  # (N, 3)
    hole_pos_w = hole.data.root_pos_w               # (N, 3) - at hole bottom
    hole_quat_w = hole.data.root_quat_w             # (N, 4)
    
    # compute hole top position in world frame
    # hole_pos_w[:, 2] is hole bottom z, hole top z = hole bottom z + hole_height
    hole_top_pos_w = hole_pos_w.clone()
    hole_top_pos_w[:, 2] = hole_pos_w[:, 2] + hole_height

    # # Consider row, pitch rotate: 
    # top_offset_h = ee_pos_w.new_zeros(hole_pos_w.shape)
    # top_offset_h[:, 2] = hole_height
    # hole_top_pos_w = hole_pos_w + math_utils.quat_rotate(hole_quat_w, top_offset_h)
    
    # ee position relative to hole top in world frame
    ee_rel_pos_w = ee_pos_w - hole_top_pos_w  # (N, 3)

    # --- express in hole local frame ---
    # ee_rel_pos_hole = math_utils.quat_rotate_inverse(hole_quat_w, ee_rel_pos_w)  # deprecated
    ee_rel_pos_hole = math_utils.quat_apply_inverse(hole_quat_w, ee_rel_pos_w)

    return ee_rel_pos_hole



# ee yaw relative to the hole top
def ee_yaw_rel_hole(
    env,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
) -> torch.Tensor:
    """EE yaw relative to hole: [ee_hole_yaw].
    
    Returns:
        torch.Tensor: (N, 1) - ee yaw angle relative to hole's yaw.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]
    
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]  # (N, 4)
    hole_quat_w = hole.data.root_quat_w               # (N, 4)
    
    # get yaw from both
    _, _, ee_yaw = math_utils.euler_xyz_from_quat(ee_quat_w)
    _, _, hole_yaw = math_utils.euler_xyz_from_quat(hole_quat_w)
    
    # relative yaw, wrapped to [-pi, pi]
    rel_yaw = math_utils.wrap_to_pi(ee_yaw - hole_yaw)
    
    return rel_yaw.unsqueeze(-1)


# ee yaw relative to the hole's second slot (slot2 = slot1 + yaw_offset)
def ee_yaw_rel_hole_slot2(
    env,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
    yaw_offset: float = 1.5708,  # default 90° = π/2
) -> torch.Tensor:
    """EE yaw relative to hole's second slot (rotated by yaw_offset from primary slot).

    For a cross-shaped hole, slot1 is at hole_yaw + 0°, slot2 is at hole_yaw + 90°.

    Returns:
        torch.Tensor: (N, 1) - ee yaw angle relative to hole's second slot.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]

    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]  # (N, 4)
    hole_quat_w = hole.data.root_quat_w               # (N, 4)

    _, _, ee_yaw = math_utils.euler_xyz_from_quat(ee_quat_w)
    _, _, hole_yaw = math_utils.euler_xyz_from_quat(hole_quat_w)

    # slot2 yaw = hole_yaw + yaw_offset
    rel_yaw = math_utils.wrap_to_pi(ee_yaw - (hole_yaw + yaw_offset))

    return rel_yaw.unsqueeze(-1)


# peg center position relative to hole top (in hole local frame)
def peg_pos_rel_hole(
    env,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
    hole_height: float = 0.08,
) -> torch.Tensor:
    """Peg center position relative to hole top in hole's local frame.

    Returns:
        torch.Tensor: (N, 3) - peg position relative to hole top.
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]

    peg_pos_w = peg.data.root_pos_w        # (N, 3)
    hole_pos_w = hole.data.root_pos_w      # (N, 3) - at hole bottom
    hole_quat_w = hole.data.root_quat_w    # (N, 4)

    hole_top_pos_w = hole_pos_w.clone()
    hole_top_pos_w[:, 2] = hole_pos_w[:, 2] + hole_height

    rel_pos_w = peg_pos_w - hole_top_pos_w
    rel_pos_hole = math_utils.quat_apply_inverse(hole_quat_w, rel_pos_w)
    return rel_pos_hole


# peg yaw relative to hole yaw
def peg_yaw_rel_hole(
    env,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
) -> torch.Tensor:
    """Peg yaw relative to hole yaw, wrapped to [-pi, pi].

    Returns:
        torch.Tensor: (N, 1) - relative yaw angle.
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]

    _, _, peg_yaw = math_utils.euler_xyz_from_quat(peg.data.root_quat_w)
    _, _, hole_yaw = math_utils.euler_xyz_from_quat(hole.data.root_quat_w)

    rel_yaw = math_utils.wrap_to_pi(peg_yaw - hole_yaw)
    return rel_yaw.unsqueeze(-1)


# ============== Auxiliary Data ==============

# hole position in world frame (relative to env origin)
def hole_pos_in_world(
    env,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
) -> torch.Tensor:
    """Hole position in world frame (relative to env origin).
    
    Returns:
        torch.Tensor: (N, 3) - hole position in world frame.
    """
    hole: RigidObject = env.scene[hole_cfg.name]
    hole_pos_w = hole.data.root_pos_w  # (N, 3)
    # subtract env origin to get position relative to env origin
    pos_rel_env = hole_pos_w - env.scene.env_origins[:, 0:3]
    return pos_rel_env


# hole orientation (quaternion) in world frame
def hole_quat_in_world(
    env,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
) -> torch.Tensor:
    """Hole orientation (quaternion) in world frame.
    
    Returns:
        torch.Tensor: (N, 4) - hole quaternion in world frame (w, x, y, z).
    """
    hole: RigidObject = env.scene[hole_cfg.name]
    hole_quat_w = hole.data.root_quat_w  # (N, 4)
    return hole_quat_w


# peg position in world frame (relative to env origin)
def peg_pos_in_world(
    env,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """Peg position in world frame (relative to env origin).
    
    Returns:
        torch.Tensor: (N, 3) - peg position in world frame.
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    peg_pos_w = peg.data.root_pos_w  # (N, 3)
    # subtract env origin to get position relative to env origin
    pos_rel_env = peg_pos_w - env.scene.env_origins[:, 0:3]
    return pos_rel_env


# peg orientation (quaternion) in world frame
def peg_quat_in_world(
    env,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """Peg orientation (quaternion) in world frame.
    
    Returns:
        torch.Tensor: (N, 4) - peg quaternion in world frame (w, x, y, z).
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    peg_quat_w = peg.data.root_quat_w  # (N, 4)
    return peg_quat_w


# hole yaw in world frame (for demo recording)
def hole_yaw_in_world(
    env,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
) -> torch.Tensor:
    """Hole yaw angle in world frame.
    
    This observation captures the yaw rotation of the hole, which is useful
    for recording the randomized yaw in each demo/episode.
    
    Returns:
        torch.Tensor: (N, 1) - hole yaw angle in radians.
    """
    hole: RigidObject = env.scene[hole_cfg.name]
    hole_quat_w = hole.data.root_quat_w  # (N, 4)
    
    # extract yaw from quaternion
    _, _, yaw = math_utils.euler_xyz_from_quat(hole_quat_w)
    
    return yaw.unsqueeze(-1)


# ============== Box-corner observations ==============

def _box_corners_world(
    root_pos_w: torch.Tensor,   # (N, 3) box origin in world
    root_quat_w: torch.Tensor,  # (N, 4) wxyz
    half_x: float,
    half_y: float,
    z_offset: float,            # local z of the face (e.g. -half_z for bottom, +height for hole top)
) -> torch.Tensor:
    """Compute 4 corners of an axis-aligned face on a box, in world frame.

    Corner order (local x, y signs):
        0: (+hx, +hy)
        1: (+hx, -hy)
        2: (-hx, +hy)
        3: (-hx, -hy)

    Returns:
        (N, 4, 3) world-frame corner positions.
    """
    n = root_pos_w.shape[0]
    device = root_pos_w.device

    signs = torch.tensor(
        [[+1.0, +1.0],
         [+1.0, -1.0],
         [-1.0, +1.0],
         [-1.0, -1.0]],
        device=device,
        dtype=root_pos_w.dtype,
    )  # (4, 2)

    corners_local = torch.zeros((4, 3), device=device, dtype=root_pos_w.dtype)
    corners_local[:, 0] = signs[:, 0] * half_x
    corners_local[:, 1] = signs[:, 1] * half_y
    corners_local[:, 2] = z_offset
    corners_local = corners_local.unsqueeze(0).expand(n, -1, -1)  # (N, 4, 3)

    # rotate each corner by root_quat_w then translate
    quat_rep = root_quat_w.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)   # (N*4, 4)
    corners_flat = corners_local.reshape(-1, 3)                            # (N*4, 3)
    rotated = math_utils.quat_apply(quat_rep, corners_flat).reshape(n, 4, 3)

    return root_pos_w.unsqueeze(1) + rotated


def peg_bottom_corners_world(
    env,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    peg_half_x: float = 0.02,
    peg_half_y: float = 0.01,
    peg_half_z: float = 0.06,
) -> torch.Tensor:
    """Peg bottom 4 corners in world frame (relative to env origin).

    Peg root is at the peg center; bottom face is at local z = -peg_half_z.
    Defaults match the current peg USD: full size 0.04 x 0.02 x 0.12.
    Override the half-extents if your USD differs.

    Returns:
        torch.Tensor: (N, 12) - 4 corners x (x, y, z) flattened, env-relative.
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    corners_w = _box_corners_world(
        peg.data.root_pos_w, peg.data.root_quat_w,
        peg_half_x, peg_half_y, -peg_half_z,
    )  # (N, 4, 3)
    corners_env = corners_w - env.scene.env_origins[:, 0:3].unsqueeze(1)
    return corners_env.reshape(env.num_envs, -1)


def peg_center_corners_world(
    env,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    peg_half_x: float = 0.02,
    peg_half_y: float = 0.01,
) -> torch.Tensor:
    """Peg mid-height (center cross-section) 4 corners in world frame (env-relative).

    Cross-section at local z = 0 (peg root is at the peg center).

    Returns:
        torch.Tensor: (N, 12) - 4 corners x (x, y, z) flattened, env-relative.
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    corners_w = _box_corners_world(
        peg.data.root_pos_w, peg.data.root_quat_w,
        peg_half_x, peg_half_y, 0.0,
    )  # (N, 4, 3)
    corners_env = corners_w - env.scene.env_origins[:, 0:3].unsqueeze(1)
    return corners_env.reshape(env.num_envs, -1)


def peg_top_corners_world(
    env,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    peg_half_x: float = 0.02,
    peg_half_y: float = 0.01,
    peg_half_z: float = 0.06,
) -> torch.Tensor:
    """Peg top 4 corners in world frame (env-relative).

    Top face is at local z = +peg_half_z (peg root is at the peg center).

    Returns:
        torch.Tensor: (N, 12) - 4 corners x (x, y, z) flattened, env-relative.
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    corners_w = _box_corners_world(
        peg.data.root_pos_w, peg.data.root_quat_w,
        peg_half_x, peg_half_y, +peg_half_z,
    )  # (N, 4, 3)
    corners_env = corners_w - env.scene.env_origins[:, 0:3].unsqueeze(1)
    return corners_env.reshape(env.num_envs, -1)


def hole_top_corners_world(
    env,
    hole_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
    hole_half_x: float = 0.02,
    hole_half_y: float = 0.01,
    hole_height: float = 0.08,
    apply_clearance_scale: bool = True,
) -> torch.Tensor:
    """Hole *inner-opening* top 4 corners in world frame (relative to env origin).

    Hole root is at the bottom of the block; the inner opening top face is at
    local z = +hole_height. The hole's xy is scaled by ``env.cfg.clearance``
    in the env config, so we multiply the half-extents by clearance when
    ``apply_clearance_scale=True``.

    Defaults match the current hole USD inner opening at clearance=1.0:
    full size 0.04 x 0.02 (half = 0.02 x 0.01).

    Returns:
        torch.Tensor: (N, 12) - 4 corners x (x, y, z) flattened, env-relative.
    """
    hole: RigidObject = env.scene[hole_cfg.name]

    if apply_clearance_scale:
        clearance = float(getattr(env.cfg, "clearance", 1.0))
        hx = hole_half_x * clearance
        hy = hole_half_y * clearance
    else:
        hx, hy = hole_half_x, hole_half_y

    corners_w = _box_corners_world(
        hole.data.root_pos_w, hole.data.root_quat_w,
        hx, hy, hole_height,
    )  # (N, 4, 3)
    corners_env = corners_w - env.scene.env_origins[:, 0:3].unsqueeze(1)
    return corners_env.reshape(env.num_envs, -1)


# # ============== Self-defined Metrics ==============

# def ee_wiggle_index(env) -> torch.Tensor:
#     """End-effector wiggle index: cumulative x-y displacement.
    
#     wiggle = sum_t sqrt((x_t - x_{t-1})^2 + (y_t - y_{t-1})^2)
    
#     This metric measures how much the end-effector "wiggles" in the x-y plane
#     during the episode. Lower values indicate smoother, more direct trajectories.
    
#     Note: Requires PegInsertEnv which tracks this state internally.
    
#     Returns:
#         torch.Tensor: (N, 1) - accumulated wiggle index for each environment.
#     """
#     return env.get_wiggle_index()


# def ee_after_contact_wiggle_index(env) -> torch.Tensor:
#     """After-contact end-effector wiggle index: cumulative x-y displacement after contact.
    
#     after_contact_wiggle = sum_t[from contact start to end] sqrt((x_t - x_{t-1})^2 + (y_t - y_{t-1})^2)
    
#     Contact condition: peg_bottom_z - hole_top_z <= 0.01
    
#     This metric measures how much the end-effector "wiggles" in the x-y plane
#     after the peg has first made contact with the hole. This is useful for
#     evaluating the quality of the insertion phase specifically.
    
#     Note: Requires PegInsertEnv which tracks this state internally.
    
#     Returns:
#         torch.Tensor: (N, 1) - accumulated after-contact wiggle index for each environment.
#     """
#     return env.get_after_contact_wiggle_index()


# def has_contacted(env) -> torch.Tensor:
#     """Whether the peg has contacted the hole.
    
#     Contact condition: peg_bottom_z - hole_top_z <= 0.01
    
#     Returns:
#         torch.Tensor: (N, 1) - bool tensor (as float: 0.0 or 1.0) indicating contact.
#     """
#     return env.get_has_contacted().float()


# def time_stamp(env) -> torch.Tensor:
#     """Time stamp: number of time-steps elapsed since episode start.
    
#     This metric tracks how many time-steps have passed in the current episode.
#     Useful for analyzing episode length and efficiency.
    
#     Note: Requires PegInsertEnv which tracks this state internally.
    
#     Returns:
#         torch.Tensor: (N, 1) - time stamp (step count) for each environment (as float).
#     """
#     return env.get_time_stamp().float()


# def completion_step(env) -> torch.Tensor:
#     """Completion time-step: the episode_step_count when success was first detected.
    
#     Returns -1 for environments that haven't succeeded yet.
    
#     Note: Requires PegInsertEnv which tracks this state internally.
    
#     Returns:
#         torch.Tensor: (N, 1) - completion step for each environment (as float).
#     """
#     return env.get_completion_step().float()


# def has_succeeded(env) -> torch.Tensor:
#     """Whether the task has succeeded.
    
#     Returns:
#         torch.Tensor: (N, 1) - bool tensor (as float: 0.0 or 1.0) indicating success.
#     """
#     return env.get_has_succeeded().float()


# def pause_count(env) -> torch.Tensor:
#     """Trajectory pause count: number of times EE was stationary for 5 consecutive steps.
    
#     A pause is detected when both ee_pos and ee_quat have no change for
#     PAUSE_WINDOW (default 5) consecutive time-steps.
    
#     Note: Requires PegInsertEnv which tracks this state internally.
    
#     Returns:
#         torch.Tensor: (N, 1) - pause count for each environment (as float).
#     """
#     return env.get_pause_count().float()


# def mean_push_speed(env) -> torch.Tensor:
#     """Mean push speed: average vertical speed during successful insertion.
    
#     mean_push_speed = (success_ee_pos_z - start_ee_pos_z) / completion_step
    
#     This measures the average vertical displacement rate during the insertion.
#     A negative value indicates downward motion (pushing down).
#     Returns 0 for environments that haven't succeeded yet.
    
#     Note: Requires PegInsertEnv which tracks this state internally.
    
#     Returns:
#         torch.Tensor: (N, 1) - mean push speed for each environment.
#     """
#     return env.get_mean_push_speed()
