# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# MIT License
#
# Copyright (c) 2021 Stanford Vision and Learning Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Entry point for training a Diffusion Policy with the geometric relational loss.

This script mirrors ``train_v2.py`` (the standard diffusion-policy training entry
point used with ``diffusion_policy_image_200.json``) and additionally injects an
auxiliary *geometric relational hinge loss* computed from an anchor pool, in the
same spirit as ``train_bc_with_geoRelLoss_mainExp2.py`` does for BC.

Example:

    ./isaaclab.sh -p scripts/imitation_learning/robomimic/train_bc_with_geoRelLoss_mainExp2_v2.py \
        --task Isaac-Peg-Insert-Franka-IK-Rel-Visuomotor-v0 \
        --algo diffusion_policy \
        --dataset ./datasets/test.hdf5 \
        --anchor_pool ./datasets/MainExp_dataset_anchorPool_forTraining.hdf5
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Standard library imports
import argparse

# Third-party imports
import gymnasium as gym
import h5py
import importlib
import json
import numpy as np
import os
import shutil
import sys
import time
import torch
import torch.nn.functional as F
import traceback
from collections import OrderedDict
from torch.utils.data import DataLoader

import psutil

# Robomimic imports
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.algo import algo_factory
from robomimic.config import Config, config_factory
from robomimic.utils.log_utils import DataLogger, PrintLogger

# Isaac Lab imports (needed so that environment is registered)
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401


def normalize_hdf5_actions(config: Config, log_dir: str) -> str:
    """Normalizes actions in hdf5 dataset to [-1, 1] range."""
    base, ext = os.path.splitext(config.train.data)
    normalized_path = base + "_normalized" + ext

    print(f"Creating normalized dataset at {normalized_path}")
    shutil.copyfile(config.train.data, normalized_path)

    with h5py.File(normalized_path, "r+") as f:
        dataset_paths = [f"/data/demo_{str(i)}/actions" for i in range(len(f["data"].keys()))]

        dataset = np.array(f[dataset_paths[0]]).flatten()
        for i, path in enumerate(dataset_paths):
            if i != 0:
                data = np.array(f[path]).flatten()
                dataset = np.append(dataset, data)

        max = np.max(dataset)
        min = np.min(dataset)

        for i, path in enumerate(dataset_paths):
            data = np.array(f[path])
            normalized_data = 2 * ((data - min) / (max - min)) - 1
            del f[path]
            f[path] = normalized_data

        with open(os.path.join(log_dir, "normalization_params.txt"), "w") as f:
            f.write(f"min: {min}\n")
            f.write(f"max: {max}\n")

    return normalized_path


# =============================================================================
# Anchor pool (identical to the BC version)
# =============================================================================
class AnchorPool:
    """Loads and manages the anchor pool for geometric relational loss."""

    def __init__(self, hdf5_path, device, eps_z=0.008, eps_yaw=5.0):
        self.device = device
        self.eps_z = eps_z
        self.eps_yaw = eps_yaw

        with h5py.File(hdf5_path, "r") as f:
            anchors_grp = f["anchors"]
            anchor_keys = sorted(anchors_grp.keys(), key=lambda x: int(x.split("_")[1]))
            n = len(anchor_keys)

            self.colors = []
            self.peg_textures = []
            self.table_textures = []
            self.sizes = np.zeros(n, dtype=np.float32)
            self.delta_z = np.zeros(n, dtype=np.float32)
            self.yaw_error = np.zeros(n, dtype=np.float32)

            self.obs_ee_pos = np.zeros((n, 3), dtype=np.float32)
            self.obs_ee_yaw = np.zeros((n, 1), dtype=np.float32)
            self.obs_images = np.zeros((n, 200, 200, 3), dtype=np.uint8)

            for idx, key in enumerate(anchor_keys):
                a = anchors_grp[key]
                self.colors.append(a.attrs["color"])
                self.peg_textures.append(a.attrs.get("peg_texture", ""))
                self.table_textures.append(a.attrs.get("table_texture", ""))
                self.sizes[idx] = a.attrs["size"]
                self.delta_z[idx] = a.attrs["delta_z"]
                self.yaw_error[idx] = a.attrs["yaw_error_deg"]
                self.obs_ee_pos[idx] = a["obs"]["ee_pos_world"][:]
                self.obs_ee_yaw[idx] = a["obs"]["ee_yaw_world"][:]
                self.obs_images[idx] = a["obs"]["table_side_cam"][:]

            self.colors = np.array(self.colors)
            self.peg_textures = np.array(self.peg_textures)
            self.table_textures = np.array(self.table_textures)

        self._precompute_pairs()
        print(f"[AnchorPool] Loaded {n} anchors from {hdf5_path}")

    def _precompute_pairs(self):
        n = len(self.colors)
        self.app_pairs = {}
        self.geo_pairs = {}

        for i in range(n):
            app_mask = (
                (self.sizes == self.sizes[i])
                & (
                    (self.colors != self.colors[i])
                    | (self.peg_textures != self.peg_textures[i])
                    | (self.table_textures != self.table_textures[i])
                )
                & (np.abs(self.delta_z - self.delta_z[i]) < self.eps_z)
                & (np.abs(self.yaw_error - self.yaw_error[i]) < self.eps_yaw)
            )
            app_indices = np.where(app_mask)[0]

            geo_mask = (
                (self.sizes != self.sizes[i])
                & (self.colors == self.colors[i])
                & (self.peg_textures == self.peg_textures[i])
                & (self.table_textures == self.table_textures[i])
                & (np.abs(self.delta_z - self.delta_z[i]) < self.eps_z)
                & (np.abs(self.yaw_error - self.yaw_error[i]) < self.eps_yaw)
            )
            geo_indices = np.where(geo_mask)[0]

            if len(app_indices) > 0 and len(geo_indices) > 0:
                self.app_pairs[i] = app_indices
                self.geo_pairs[i] = geo_indices

        self.valid_anchors = np.array(sorted(self.app_pairs.keys()))
        print(f"[AnchorPool] {len(self.valid_anchors)}/{n} anchors have valid pairs")

    # def _find_nearest(self, base_idx, candidates):
    #     dz_diff = np.abs(self.delta_z[candidates] - self.delta_z[base_idx])
    #     yaw_diff = np.abs(self.yaw_error[candidates] - self.yaw_error[base_idx])
    #     dist = (dz_diff / self.eps_z) ** 2 + (yaw_diff / self.eps_yaw) ** 2
    #     return candidates[np.argmin(dist)]
    
    def _sample_random(self, base_idx, candidates):
        """Randomly sample one candidate from the valid candidates."""
        return candidates[np.random.randint(len(candidates))]

    def sample_triplets(self, M):
        if len(self.valid_anchors) == 0:
            return None, None, None, 0

        base_indices = np.random.choice(self.valid_anchors, size=M, replace=True)
        # app_indices = np.array([self._find_nearest(bi, self.app_pairs[bi]) for bi in base_indices])
        # geo_indices = np.array([self._find_nearest(bi, self.geo_pairs[bi]) for bi in base_indices])
        app_indices = np.array([self._sample_random(bi, self.app_pairs[bi]) for bi in base_indices])
        geo_indices = np.array([self._sample_random(bi, self.geo_pairs[bi]) for bi in base_indices])

        obs_base = self._build_obs_batch(base_indices)
        obs_app = self._build_obs_batch(app_indices)
        obs_geo = self._build_obs_batch(geo_indices)

        return obs_base, obs_app, obs_geo, M

    def _build_obs_batch(self, indices):
        ee_pos = torch.from_numpy(self.obs_ee_pos[indices]).float().to(self.device)
        ee_yaw = torch.from_numpy(self.obs_ee_yaw[indices]).float().to(self.device)
        images = torch.from_numpy(self.obs_images[indices].copy()).float().to(self.device)
        images = images / 255.0
        images = images.permute(0, 3, 1, 2)  # [M, C, H, W]
        return {
            "ee_pos_world": ee_pos,
            "ee_yaw_world": ee_yaw,
            "table_side_cam": images,
        }


# =============================================================================
# Diffusion-policy specific behavior descriptor
# =============================================================================
def _add_time_dim(obs_dict, model):
    """Ensure each observation has a leading time dimension matching ``To``."""
    To = model.algo_config.horizon.observation_horizon
    out = {}
    for k, v in obs_dict.items():
        expected_ndim = len(model.obs_shapes[k])
        if v.ndim - 1 == expected_ndim:
            v = v.unsqueeze(1)  # [M, 1, ...]
        if v.shape[1] != To:
            # repeat the single observation across all To frames
            repeats = [1] * v.ndim
            repeats[1] = To
            v = v.repeat(*repeats)
        out[k] = v
    return out


def _diffusion_action_trajectory(model, obs_dict, num_inference_timesteps=None):
    """Run the diffusion sampling loop and return the predicted action trajectory.

    Mirrors ``DiffusionPolicyUNet._get_action_trajectory`` but uses the *trainable*
    ``self.nets`` (not the EMA copy) and keeps gradients enabled so that the
    behavior descriptor can flow back through the network.

    Returns:
        actions: Tensor of shape [B, Ta, Da] (the action_horizon slice).
    """
    nets = model.nets  # use the trainable parameters (NOT EMA)
    device = model.device

    To = model.algo_config.horizon.observation_horizon
    Ta = model.algo_config.horizon.action_horizon
    Tp = model.algo_config.horizon.prediction_horizon
    action_dim = model.ac_dim

    obs_dict = _add_time_dim(obs_dict, model)
    inputs = {"obs": obs_dict, "goal": None}

    obs_features = TensorUtils.time_distributed(
        inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True
    )
    assert obs_features.ndim == 3  # [B, T, D]
    B = obs_features.shape[0]
    obs_cond = obs_features.flatten(start_dim=1)

    # determine inference step count
    if num_inference_timesteps is None:
        if model.algo_config.ddpm.enabled:
            num_inference_timesteps = model.algo_config.ddpm.num_inference_timesteps
        elif model.algo_config.ddim.enabled:
            num_inference_timesteps = model.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError("Neither DDPM nor DDIM is enabled.")

    # initialise from Gaussian noise
    naction = torch.randn((B, Tp, action_dim), device=device)

    model.noise_scheduler.set_timesteps(num_inference_timesteps)
    for k in model.noise_scheduler.timesteps:
        noise_pred = nets["policy"]["noise_pred_net"](
            sample=naction, timestep=k, global_cond=obs_cond
        )
        naction = model.noise_scheduler.step(
            model_output=noise_pred, timestep=k, sample=naction
        ).prev_sample

    start = To - 1
    end = start + Ta
    return naction[:, start:end]  # [B, Ta, Da]


def _compute_behavior_descriptor_diffusion(model, obs_dict, yaw_action_idx=3, num_inference_timesteps=None, K=None):
    """Behavior descriptor for diffusion policy: cumulative |yaw| over the first ``K`` action steps.

    If ``K`` is None, the full action_horizon (Ta) is used.
    """
    actions = _diffusion_action_trajectory(
        model, obs_dict, num_inference_timesteps=num_inference_timesteps
    )  # [B, Ta, Da]
    if K is not None:
        actions = actions[:, :K]
    return torch.abs(actions[..., yaw_action_idx]).sum(dim=1)  # [B]


def compute_geo_rel_loss(
    model,
    anchor_pool,
    M=16,
    margin=0.05,
    yaw_action_idx=3,
    num_inference_timesteps=None,
    log_diagnostics=False,
    K=None,
):
    """Compute geometric relational hinge loss from anchor triplets (diffusion variant)."""
    device = model.device
    obs_base, obs_app, obs_geo, valid_count = anchor_pool.sample_triplets(M)

    if valid_count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {}

    b_base = _compute_behavior_descriptor_diffusion(
        model, obs_base, yaw_action_idx=yaw_action_idx, num_inference_timesteps=num_inference_timesteps, K=K
    )
    b_app = _compute_behavior_descriptor_diffusion(
        model, obs_app, yaw_action_idx=yaw_action_idx, num_inference_timesteps=num_inference_timesteps, K=K
    )
    b_geo = _compute_behavior_descriptor_diffusion(
        model, obs_geo, yaw_action_idx=yaw_action_idx, num_inference_timesteps=num_inference_timesteps, K=K
    )

    D_app = (b_base - b_app) ** 2
    D_geo = (b_base - b_geo) ** 2
    L_rel = torch.mean(torch.clamp(margin + D_app - D_geo, min=0.0))

    diagnostics = {}
    if log_diagnostics:
        with torch.no_grad():
            diagnostics["b_base_mean"] = b_base.mean().item()
            diagnostics["b_base_std"] = b_base.std().item()
            diagnostics["D_app_mean"] = D_app.mean().item()
            diagnostics["D_app_std"] = D_app.std().item()
            diagnostics["D_geo_mean"] = D_geo.mean().item()
            diagnostics["D_geo_std"] = D_geo.std().item()
            diagnostics["gap_mean"] = (D_geo - D_app).mean().item()
            diagnostics["violation_rate"] = (D_app > D_geo).float().mean().item()

    return L_rel, diagnostics


# =============================================================================
# Custom epoch loop: standard DP loss + geo_rel loss in a single backward pass
# =============================================================================
def _diffusion_task_loss(model, batch):
    """Re-implement DiffusionPolicyUNet.train_on_batch's forward pass without backprop / EMA."""
    actions = batch["actions"]
    inputs = {"obs": batch["obs"], "goal": batch.get("goal_obs", None)}

    # sanity: actions must already be in [-1, 1]
    if not model.action_check_done:
        in_range = (-1 <= actions) & (actions <= 1)
        if not bool(torch.all(in_range).item()):
            raise ValueError(
                "'actions' must be in range [-1,1] for Diffusion Policy! "
                "Check if hdf5_normalize_action is enabled."
            )
        model.action_check_done = True

    obs_features = TensorUtils.time_distributed(
        inputs, model.nets["policy"]["obs_encoder"], inputs_as_kwargs=True
    )
    assert obs_features.ndim == 3  # [B, T, D]
    obs_cond = obs_features.flatten(start_dim=1)

    noise = torch.randn(actions.shape, device=model.device)
    B = actions.shape[0]
    timesteps = torch.randint(
        0, model.noise_scheduler.config.num_train_timesteps, (B,), device=model.device
    ).long()

    noisy_actions = model.noise_scheduler.add_noise(actions, noise, timesteps)
    noise_pred = model.nets["policy"]["noise_pred_net"](
        noisy_actions, timesteps, global_cond=obs_cond
    )

    return F.mse_loss(noise_pred, noise)


def run_epoch_with_geo_rel_loss(
    model,
    data_loader,
    anchor_pool,
    epoch,
    num_steps=None,
    obs_normalization_stats=None,
    M=16,
    margin=0.05,
    lambda_rel=0.1,
    yaw_action_idx=3,
    geo_rel_inference_timesteps=None,
    log_geo_diagnostics=False,
    geo_rel_warmup_epochs=0,
    geo_rel_active_epochs=None,
    geo_rel_K=None,
):
    """Run one training epoch with the diffusion-policy task loss plus geo relational loss.

    The geometric relational loss is only active for epochs in the half-open interval
    ``[geo_rel_warmup_epochs + 1, geo_rel_warmup_epochs + geo_rel_active_epochs]``.
    Outside this window L_rel is skipped entirely (no anchor sampling, no descriptor
    rollout) so that early training is unaffected and late training avoids the wasted
    compute when the hinge has already saturated.
    """
    model.set_train()

    # decide whether L_rel is active this epoch
    geo_active = epoch > geo_rel_warmup_epochs
    if geo_rel_active_epochs is not None:
        geo_active = geo_active and (epoch <= geo_rel_warmup_epochs + geo_rel_active_epochs)

    step_log_all = []
    data_loader_iter = iter(data_loader)
    if num_steps is None:
        num_steps = len(data_loader)

    for _ in range(num_steps):
        # --- fetch and pre-process the batch ---
        try:
            batch = next(data_loader_iter)
        except StopIteration:
            data_loader_iter = iter(data_loader)
            batch = next(data_loader_iter)

        input_batch = model.process_batch_for_training(batch)
        input_batch = model.postprocess_batch_for_training(
            input_batch, obs_normalization_stats=obs_normalization_stats
        )

        # --- diffusion-policy task loss (noise prediction MSE) ---
        L_task = _diffusion_task_loss(model, input_batch)

        # --- geometric relational loss (only when active) ---
        if geo_active:
            L_rel, geo_diag = compute_geo_rel_loss(
                model=model,
                anchor_pool=anchor_pool,
                M=M,
                margin=margin,
                yaw_action_idx=yaw_action_idx,
                num_inference_timesteps=geo_rel_inference_timesteps,
                log_diagnostics=log_geo_diagnostics,
                K=geo_rel_K,
            )
            L_total = L_task + lambda_rel * L_rel
        else:
            L_rel = torch.tensor(0.0, device=model.device)
            geo_diag = {}
            L_total = L_task

        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=model.nets,
            optim=model.optimizers["policy"],
            loss=L_total,
        )

        # update EMA after the gradient step (matches DiffusionPolicyUNet.train_on_batch)
        if getattr(model, "ema", None) is not None:
            model.ema.step(model.nets)

        model.on_gradient_step()

        # --- logging ---
        step_log = OrderedDict()
        step_log["Loss"] = L_task.item()
        step_log["L_rel"] = L_rel.item()
        step_log["L_total"] = L_total.item()
        step_log["Policy_Grad_Norms"] = policy_grad_norms
        for diag_k, diag_v in geo_diag.items():
            step_log[f"geo_{diag_k}"] = diag_v
        step_log_all.append(step_log)

    avg_log = OrderedDict()
    for k in step_log_all[0]:
        avg_log[k] = float(np.mean([s[k] for s in step_log_all]))
    return avg_log


# =============================================================================
# Train loop (mirrors train_v2.py with the custom epoch fn swapped in)
# =============================================================================
def train(
    config: Config,
    device: str,
    log_dir: str,
    ckpt_dir: str,
    video_dir: str,
    anchor_pool_path: str,
    lambda_rel: float = 0.1,
    geo_rel_M: int = 16,
    geo_rel_margin: float = 0.05,
    yaw_action_idx: int = 3,
    geo_rel_inference_timesteps=None,
    geo_rel_warmup_epochs: int = 0,
    geo_rel_active_epochs=None,
    geo_rel_K=None,
):
    """Train a diffusion policy with the geometric relational loss."""
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")

    print(f">>> Saving logs into directory: {log_dir}")
    print(f">>> Saving checkpoints into directory: {ckpt_dir}")
    print(f">>> Saving videos into directory: {video_dir}")

    if config.experiment.logging.terminal_output_to_txt:
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    ObsUtils.initialize_obs_utils_with_config(config)

    # convert single dataset path to list format for v0.5.0 compatibility
    if isinstance(config.train.data, str):
        with config.values_unlocked():
            config.train.data = [{"path": config.train.data}]

    env_meta_list = []
    shape_meta_list = []
    for dataset_cfg in config.train.data:
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset at provided path {dataset_path} not found!")

        print("\n============= Loaded Environment Metadata =============")
        with h5py.File(dataset_path, "r") as f_meta:
            env_meta = json.loads(f_meta["data"].attrs["env_args"])
        if "env_kwargs" not in env_meta:
            env_meta["env_kwargs"] = {}

        env_meta["lang"] = dataset_cfg.get("lang", "dummy")

        from robomimic.utils.python_utils import deep_update
        deep_update(env_meta, config.experiment.env_meta_update_dict)
        env_meta_list.append(env_meta)

        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_config=dataset_cfg,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            verbose=True,
        )
        shape_meta_list.append(shape_meta)

    if config.experiment.env is not None:
        env_meta = env_meta_list[0].copy()
        env_meta["env_name"] = config.experiment.env
        env_meta_list = [env_meta]
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environments for validation rollouts
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        for env_i in range(len(env_meta_list)):
            dataset_cfg = config.train.data[env_i]
            do_eval = dataset_cfg.get("eval", True)
            if not do_eval:
                continue

            env_meta = env_meta_list[env_i]
            shape_meta = shape_meta_list[env_i]

            env_names = [env_meta["env_name"]]
            if (env_i == 0) and (config.experiment.additional_envs is not None):
                for name in config.experiment.additional_envs:
                    env_names.append(name)

            for env_name in env_names:
                env = EnvUtils.create_env_from_metadata(
                    env_meta=env_meta,
                    env_name=env_name,
                    render=False,
                    render_offscreen=config.experiment.render_video,
                    use_image_obs=shape_meta["use_images"] or shape_meta.get("use_depths", False),
                )
                env = EnvUtils.wrap_env_from_config(env, config=config)
                env_key = (
                    os.path.splitext(os.path.basename(dataset_cfg["path"]))[0]
                    if not dataset_cfg.get("key", None)
                    else dataset_cfg["key"]
                )
                envs[env_key] = env
                print(env)

    print("")

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta_list[0]["all_obs_keys"]
    )
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")
    if validset is not None:
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")

    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    action_normalization_stats = trainset.get_action_normalization_stats()

    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
    )

    if config.experiment.validate:
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True,
        )
    else:
        valid_loader = None

    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    with config.values_unlocked():
        if "optim_params" in config.algo:
            for k in config.algo.optim_params:
                config.algo.optim_params[k]["num_train_batches"] = (
                    len(trainset) if train_num_steps is None else train_num_steps
                )
                config.algo.optim_params[k]["num_epochs"] = config.train.num_epochs

    data_logger = DataLogger(
        log_dir,
        config=config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta_list[0]["all_shapes"],
        ac_dim=shape_meta_list[0]["ac_dim"],
        device=device,
    )

    ckpt_path = config.experiment.ckpt_path
    if ckpt_path is not None:
        print("LOADING MODEL WEIGHTS FROM " + ckpt_path)
        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
        model.deserialize(ckpt_dict["model"])

    with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)
    print("")

    # load anchor pool
    anchor_pool = AnchorPool(anchor_pool_path, device)

    # main training loop
    best_valid_loss = None
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1.0 for k in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()

    for epoch in range(1, config.train.num_epochs + 1):
        step_log = run_epoch_with_geo_rel_loss(
            model=model,
            data_loader=train_loader,
            anchor_pool=anchor_pool,
            epoch=epoch,
            num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats,
            M=geo_rel_M,
            margin=geo_rel_margin,
            lambda_rel=lambda_rel,
            yaw_action_idx=yaw_action_idx,
            geo_rel_inference_timesteps=geo_rel_inference_timesteps,
            log_geo_diagnostics=True,
            geo_rel_warmup_epochs=geo_rel_warmup_epochs,
            geo_rel_active_epochs=geo_rel_active_epochs,
            geo_rel_K=geo_rel_K,
        )
        model.on_epoch_end(epoch)

        epoch_ckpt_name = f"model_epoch_{epoch}"

        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and (
                time.time() - last_ckpt_time > config.experiment.save.every_n_seconds
            )
            epoch_check = (
                (config.experiment.save.every_n_epochs is not None)
                and (epoch > 0)
                and (epoch % config.experiment.save.every_n_epochs == 0)
            )
            epoch_list_check = epoch in config.experiment.save.epochs
            should_save_ckpt = time_check or epoch_check or epoch_list_check
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        print(f"Train Epoch {epoch}")
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record(f"Timing_Stats/Train_{k[5:]}", v, epoch)
            else:
                data_logger.record(f"Train/{k}", v, epoch)

        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model,
                    data_loader=valid_loader,
                    epoch=epoch,
                    validate=True,
                    num_steps=valid_num_steps,
                    obs_normalization_stats=obs_normalization_stats,
                )
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record(f"Timing_Stats/Valid_{k[5:]}", v, epoch)
                else:
                    data_logger.record(f"Valid/{k}", v, epoch)

            print(f"Validation Epoch {epoch}")
            print(json.dumps(step_log, sort_keys=True, indent=4))

            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += f"_best_validation_{best_valid_loss}"
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        rollout_check = (epoch % config.experiment.rollout.rate == 0) or (
            should_save_ckpt and ckpt_reason == "time"
        )
        if config.experiment.rollout.enabled and (epoch > config.experiment.rollout.warmstart) and rollout_check:
            from robomimic.algo import RolloutPolicy
            rollout_model = RolloutPolicy(
                model,
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

            num_episodes = config.experiment.rollout.n
            all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                policy=rollout_model,
                envs=envs,
                horizon=config.experiment.rollout.horizon,
                use_goals=config.use_goals,
                num_episodes=num_episodes,
                render=False,
                video_dir=video_dir if config.experiment.render_video else None,
                epoch=epoch,
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
            )

            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                for k, v in rollout_logs.items():
                    if k.startswith("Time_"):
                        data_logger.record(f"Timing_Stats/Rollout_{env_name}_{k[5:]}", v, epoch)
                    else:
                        data_logger.record(f"Rollout/{k}/{env_name}", v, epoch, log_stats=True)

                print(f"\nEpoch {epoch} Rollouts took {rollout_logs['time']}s (avg) with results:")
                print(f"Env: {env_name}")
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

            updated_stats = TrainUtils.should_save_from_rollout_logs(
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            should_save_ckpt = (
                config.experiment.save.enabled and updated_stats["should_save_ckpt"]
            ) or should_save_ckpt
            if updated_stats["ckpt_reason"] is not None:
                ckpt_reason = updated_stats["ckpt_reason"]

        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta_list[0] if len(env_meta_list) == 1 else env_meta_list,
                shape_meta=shape_meta_list[0] if len(shape_meta_list) == 1 else shape_meta_list,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print(f"\nEpoch {epoch} Memory Usage: {mem_usage} MB\n")

    data_logger.close()


def main(args: argparse.Namespace):
    if args.task is not None:
        cfg_entry_point_key = f"robomimic_{args.algo}_cfg_entry_point"
        task_name = args.task.split(":")[-1]

        print(f"Loading configuration for task: {task_name}")
        print(gym.envs.registry.keys())
        print(" ")
        cfg_entry_point_file = gym.spec(task_name).kwargs.pop(cfg_entry_point_key)
        if cfg_entry_point_file is None:
            raise ValueError(
                f"Could not find configuration for the environment: '{task_name}'."
                f" Please check that the gym registry has the entry point: '{cfg_entry_point_key}'."
            )

        if ":" in cfg_entry_point_file:
            mod_name, file_name = cfg_entry_point_file.split(":")
            mod = importlib.import_module(mod_name)
            if mod.__file__ is None:
                raise ValueError(f"Could not find module file for: '{mod_name}'")
            mod_path = os.path.dirname(mod.__file__)
            config_file = os.path.join(mod_path, file_name)
        else:
            config_file = cfg_entry_point_file

        with open(config_file) as f:
            ext_cfg = json.load(f)
            config = config_factory(ext_cfg["algo_name"])
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        raise ValueError("Please provide a task name through CLI arguments.")

    if args.dataset is not None:
        config.train.data = args.dataset

    if args.name is not None:
        config.experiment.name = args.name

    if args.epochs is not None:
        config.train.num_epochs = args.epochs

    config.train.output_dir = os.path.abspath(os.path.join("./logs", args.log_dir, args.task))

    log_dir, ckpt_dir, video_dir, *_ = TrainUtils.get_exp_dir(config)

    if args.normalize_training_actions:
        config.train.data = normalize_hdf5_actions(config, log_dir)

    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    config.lock()

    res_str = "finished run successfully!"
    try:
        train(
            config,
            device,
            log_dir,
            ckpt_dir,
            video_dir,
            anchor_pool_path=args.anchor_pool,
            lambda_rel=args.lambda_rel,
            geo_rel_M=args.geo_rel_M,
            geo_rel_margin=args.geo_rel_margin,
            yaw_action_idx=args.yaw_action_idx,
            geo_rel_inference_timesteps=args.geo_rel_inference_timesteps,
            geo_rel_warmup_epochs=args.geo_rel_warmup_epochs,
            geo_rel_active_epochs=args.geo_rel_active_epochs,
            geo_rel_K=args.geo_rel_K,
        )
    except Exception as e:
        res_str = f"run failed with error:\n{e}\n\n{traceback.format_exc()}"
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name", type=str, default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--algo", type=str, default=None, help="Name of the algorithm.")
    parser.add_argument("--log_dir", type=str, default="robomimic", help="Path to log directory")
    parser.add_argument(
        "--normalize_training_actions", action="store_true", default=False,
        help="Normalize actions",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=(
            "Optional: Number of training epochs. If specified, overrides the number"
            " of epochs from the JSON training config."
        ),
    )

    # Geometric relational loss arguments
    parser.add_argument(
        "--anchor_pool",
        type=str,
        default="./datasets/MainExp_dataset_anchorPool_forTraining.hdf5",
        help="Path to the anchor pool HDF5 file for geometric relational loss",
    )
    parser.add_argument("--lambda_rel", type=float, default=0.1, help="Weight for geometric relational loss")
    parser.add_argument("--geo_rel_M", type=int, default=16, help="Number of anchor triplets per training step")
    parser.add_argument("--geo_rel_margin", type=float, default=3.0, help="Margin for hinge loss")
    parser.add_argument(
        "--geo_rel_K",
        type=int,
        default=8,
        help=(
            "Number of action steps used to compute the behavior descriptor"
            " (cumulative |yaw| over the first K predicted actions). Set to a"
            " negative value to use the full action_horizon (Ta)."
        ),
    )
    parser.add_argument("--yaw_action_idx", type=int, default=3, help="Index of yaw component in action vector")
    parser.add_argument(
        "--geo_rel_inference_timesteps",
        type=int,
        default=10,
        help=(
            "Number of denoising steps used when sampling action trajectories for the"
            " behavior descriptor. A small value (e.g. 5-10) keeps the auxiliary loss"
            " cheap. Set to a negative value to fall back to the algo config's"
            " num_inference_timesteps."
        ),
    )
    parser.add_argument(
        "--geo_rel_warmup_epochs",
        type=int,
        default=100,
        help=(
            "Number of warm-up epochs during which L_rel is disabled. The geometric"
            " relational loss is only applied for epochs > geo_rel_warmup_epochs."
        ),
    )
    parser.add_argument(
        "--geo_rel_active_epochs",
        type=int,
        default=-1,
        help=(
            "Number of consecutive epochs (after warm-up) during which L_rel is"
            " active. After geo_rel_warmup_epochs + geo_rel_active_epochs the loss"
            " is disabled again. Set to a negative value (default) to keep it active"
            " until the end of training."
        ),
    )

    args = parser.parse_args()
    if args.geo_rel_inference_timesteps is not None and args.geo_rel_inference_timesteps < 0:
        args.geo_rel_inference_timesteps = None
    if args.geo_rel_active_epochs is not None and args.geo_rel_active_epochs < 0:
        args.geo_rel_active_epochs = None
    if args.geo_rel_K is not None and args.geo_rel_K < 0:
        args.geo_rel_K = None

    main(args)
    simulation_app.close()
