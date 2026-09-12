# Action comes from: keyboard command delta
import h5py
import numpy as np
import shutil
from pathlib import Path


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (w,x,y,z) to rotation matrix R (3x3).
    Assumes quaternion is normalized or close to normalized.
    """
    w, x, y, z = q_wxyz
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)],
    ], dtype=np.float32)
    return R


def rotate_world_vec_to_hole(vec_w: np.ndarray, hole_quat_wxyz: np.ndarray) -> np.ndarray:
    """
    vec_w: (3,) vector expressed in WORLD frame
    hole_quat_wxyz: (4,) quaternion of hole orientation in WORLD frame (wxyz)
    Returns vec expressed in HOLE frame.

    If R_wh maps hole->world, then world->hole is R_hw = R_wh^T.
    """
    R_wh = quat_wxyz_to_rotmat(hole_quat_wxyz)   # hole -> world
    R_hw = R_wh.T                                # world -> hole
    return (R_hw @ vec_w.astype(np.float32)).astype(np.float32)


def process_depth(d, max_depth=None):
    """
    Normalize depth data to [0, 1] range.
    Handles inf/nan values by replacing them with max_depth.
    """
    d = d.astype(np.float32)
    finite = np.isfinite(d)
    if max_depth is None:
        # Use the largest finite value present in the data as max_depth (simplest option)
        max_depth = float(d[finite].max())
    # inf / nan -> max_depth
    d = np.where(finite, d, max_depth).astype(np.float32)
    d = np.clip(d, 0.0, max_depth).astype(np.float32)
    d = (d / max_depth).astype(np.float32)  # [0, 1]
    return d, max_depth


def add_action_hole_and_world_4d_from_actions(
    input_hdf5: str = "dataset.hdf5",
    output_hdf5: str = "dataset_with4DAction.hdf5",
    demo_prefix: str = "demo_",

    # where to read commands
    actions_key: str = "actions",          # demo["actions"] : (T,7) = [dx,dy,dz,droll,dpitch,dyaw,grip]

    # hole quat key (wxyz) stored in obs
    hole_quat_key: str = "hole_quat_world",      # demo["obs"]["hole_quat_world"] : (T,4) or (4,)

    # output dataset names
    out_key_world: str = "action_world_4d",
    out_key_hole: str = "action_hole_4d",

    # scaling
    scale_xyz: float = 0.05,   # meters
    # scale_yaw: float = 0.05,   # radians: smallExp2-5, MainExp1 used
    scale_yaw: float = 0.2,    # smallExp6, 7 used

    clip_to_unit: bool = True,

    # clearance scaling parameters
    clearance_key: str = "clearance",  # obs key for clearance
    c_min: float = 1.03,
    c_max: float = 1.60,

    # depth normalization parameters
    depth_keys: list = None,  # depth obs keys to normalize
    depth_max: float = None,  # max depth for normalization (None = auto-detect per key)
):
    """
    Recompute 4D actions from 7D *command* actions (NOT from state deltas):

      action_world_4d[t] = [actions[t,0], actions[t,1], actions[t,2], actions[t,5]]
      action_hole_4d[t]  = [R_hw(t) @ dpos_world(t), dyaw_world(t)]
        where R_hw(t) = (R_wh(t))^T from hole_quat(t) (wxyz)

    Then:
      - wrap yaw to [-pi,pi] (mainly for safety)
      - scale (/scale_xyz, /scale_yaw)
      - optional clip
      - last step set to zero (to keep length T)

    Also applies clearance scaling:
      c_norm = 2 * (c - c_min) / (c_max - c_min) - 1
      This maps clearance from [c_min, c_max] to [-1, 1].

    Also applies depth normalization:
      Normalizes depth images to [0, 1] range, handling inf/nan values.
    """

    # Default depth keys
    if depth_keys is None:
        depth_keys = ["table_side_cam_depth", "table_front_cam_depth", "wrist_cam_depth"]

    input_hdf5 = Path(input_hdf5)
    output_hdf5 = Path(output_hdf5)
    shutil.copy(input_hdf5, output_hdf5)

    if scale_xyz <= 0 or scale_yaw <= 0:
        raise ValueError(f"scale_xyz and scale_yaw must be > 0, got {scale_xyz}, {scale_yaw}")

    with h5py.File(output_hdf5, "r+") as f:
        data_grp = f["data"]

        for demo_name in data_grp:
            if not demo_name.startswith(demo_prefix):
                continue

            demo = data_grp[demo_name]

            # ---- read actions (commands) ----
            if actions_key not in demo:
                raise KeyError(f"{demo_name} has no '{actions_key}' dataset at demo level.")
            actions = demo[actions_key][()]   # expected (T,7) float

            if actions.ndim != 2 or actions.shape[1] < 6:
                raise ValueError(f"{demo_name}: {actions_key} expected (T,7-ish) but got {actions.shape}")

            T = actions.shape[0]

            # ---- read hole_quat from obs ----
            if "obs" not in demo:
                raise KeyError(f"{demo_name} has no 'obs' group.")
            obs = demo["obs"]
            if hole_quat_key not in obs:
                raise KeyError(f"{demo_name} missing obs key: '{hole_quat_key}'")

            hole_quat = obs[hole_quat_key][()]

            # normalize hole_quat to (T,4)
            if hole_quat.ndim == 1:
                if hole_quat.shape[0] != 4:
                    raise ValueError(f"{demo_name}: {hole_quat_key} expected (4,) but got {hole_quat.shape}")
                hole_quat = np.tile(hole_quat[None, :], (T, 1))
            elif hole_quat.ndim == 2:
                if hole_quat.shape != (T, 4):
                    raise ValueError(f"{demo_name}: {hole_quat_key} expected (T,4) but got {hole_quat.shape}")
            else:
                raise ValueError(f"{demo_name}: {hole_quat_key} expected (4,) or (T,4) but got {hole_quat.shape}")

            # ---- allocate outputs ----
            action_world_4d = np.zeros((T, 4), dtype=np.float32)
            action_hole_4d = np.zeros((T, 4), dtype=np.float32)

            # ---- build world 4d from commands ----
            dpos_w = actions[:, 0:3].astype(np.float32)     # (T,3)
            dyaw_w = actions[:, 5].astype(np.float32)       # (T,)
            dyaw_w = wrap_to_pi(dyaw_w)

            # last action usually should be 0 to match "state[t] -> action[t] -> state[t+1]" convention
            dpos_w_last0 = dpos_w.copy()
            dyaw_w_last0 = dyaw_w.copy()
            dpos_w_last0[-1, :] = 0.0
            dyaw_w_last0[-1] = 0.0

            # ---- rotate to hole frame ----
            dpos_h = np.zeros_like(dpos_w_last0, dtype=np.float32)
            for t in range(T):
                if t == T - 1:
                    dpos_h[t] = 0.0
                else:
                    dpos_h[t] = rotate_world_vec_to_hole(dpos_w_last0[t], hole_quat[t])

            dyaw_h = dyaw_w_last0.copy()  # hole is static -> delta relative yaw == delta world yaw (same scalar)

            # ---- scaling ----
            dpos_w_s = dpos_w_last0 / float(scale_xyz)
            dyaw_w_s = dyaw_w_last0 / float(scale_yaw)

            dpos_h_s = dpos_h / float(scale_xyz)
            dyaw_h_s = dyaw_h / float(scale_yaw)

            # ---- clip ----
            if clip_to_unit:
                dpos_w_s = np.clip(dpos_w_s, -1.0, 1.0)
                dyaw_w_s = np.clip(dyaw_w_s, -1.0, 1.0)
                dpos_h_s = np.clip(dpos_h_s, -1.0, 1.0)
                dyaw_h_s = np.clip(dyaw_h_s, -1.0, 1.0)

            # ---- write arrays ----
            action_world_4d[:, :3] = dpos_w_s.astype(np.float32)
            action_world_4d[:, 3] = dyaw_w_s.astype(np.float32)

            action_hole_4d[:, :3] = dpos_h_s.astype(np.float32)
            action_hole_4d[:, 3] = dyaw_h_s.astype(np.float32)

            # ---- save datasets ----
            if out_key_world in demo:
                del demo[out_key_world]
            demo.create_dataset(out_key_world, data=action_world_4d, compression="gzip", compression_opts=4)

            if out_key_hole in demo:
                del demo[out_key_hole]
            demo.create_dataset(out_key_hole, data=action_hole_4d, compression="gzip", compression_opts=4)

            # ---- clearance scaling ----
            clearance_scaled = False
            if clearance_key in obs:
                clearance_raw = obs[clearance_key][()]
                # Apply scaling: c_norm = 2 * (c - c_min) / (c_max - c_min) - 1
                clearance_norm = 2.0 * (clearance_raw - c_min) / (c_max - c_min) - 1.0
                clearance_norm = clearance_norm.astype(np.float32)

                # Overwrite the original clearance dataset
                del obs[clearance_key]
                obs.create_dataset(clearance_key, data=clearance_norm, compression="gzip", compression_opts=4)
                clearance_scaled = True

            # ---- depth normalization ----
            depth_normalized_keys = []
            for depth_key in depth_keys:
                if depth_key in obs:
                    depth_raw = obs[depth_key][()]
                    depth_norm, used_max = process_depth(depth_raw, max_depth=depth_max)

                    # Overwrite the original depth dataset
                    del obs[depth_key]
                    obs.create_dataset(depth_key, data=depth_norm, compression="gzip", compression_opts=4)
                    depth_normalized_keys.append(f"{depth_key}(max={used_max:.3f})")

            print(
                f"[OK] {demo_name}: wrote {out_key_world} & {out_key_hole}, "
                f"T={T}, scale_xyz={scale_xyz}, scale_yaw={scale_yaw}, clip={clip_to_unit}, "
                f"clearance_scaled={clearance_scaled} (c_min={c_min}, c_max={c_max}), "
                f"depth_normalized={depth_normalized_keys}"
            )

    print(f"\nSaved new file with {out_key_world}, {out_key_hole}, scaled {clearance_key}, and normalized depth → {output_hdf5}")


if __name__ == "__main__":
    add_action_hole_and_world_4d_from_actions()
