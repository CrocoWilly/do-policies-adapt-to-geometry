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
The main entry point for training policies from pre-collected data.

This script loads dataset(s), creates a model based on the algorithm specified,
and trains the model. It supports training on various environments with multiple
algorithms from robomimic.

Args:
    algo: Name of the algorithm to run.
    task: Name of the environment.
    name: If provided, override the experiment name defined in the config.
    dataset: If provided, override the dataset path defined in the config.
    log_dir: Directory to save logs.
    normalize_training_actions: Whether to normalize actions in the training data.

This file has been modified from the original robomimic version to integrate with IsaacLab.
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
import traceback
from collections import OrderedDict
from torch.utils.data import DataLoader

import psutil

# Robomimic imports
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
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
    """Normalizes actions in hdf5 dataset to [-1, 1] range.

    Args:
        config: The configuration object containing dataset path.
        log_dir: Directory to save normalization parameters.

    Returns:
        Path to the normalized dataset.
    """
    base, ext = os.path.splitext(config.train.data)
    normalized_path = base + "_normalized" + ext

    # Copy the original dataset
    print(f"Creating normalized dataset at {normalized_path}")
    shutil.copyfile(config.train.data, normalized_path)

    # Open the new dataset and normalize the actions
    with h5py.File(normalized_path, "r+") as f:
        dataset_paths = [f"/data/demo_{str(i)}/actions" for i in range(len(f["data"].keys()))]

        # Compute the min and max of the dataset
        dataset = np.array(f[dataset_paths[0]]).flatten()
        for i, path in enumerate(dataset_paths):
            if i != 0:
                data = np.array(f[path]).flatten()
                dataset = np.append(dataset, data)

        max = np.max(dataset)
        min = np.min(dataset)

        # Normalize the actions
        for i, path in enumerate(dataset_paths):
            data = np.array(f[path])
            normalized_data = 2 * ((data - min) / (max - min)) - 1  # Scale to [-1, 1] range
            del f[path]
            f[path] = normalized_data

        # Save the min and max values to log directory
        with open(os.path.join(log_dir, "normalization_params.txt"), "w") as f:
            f.write(f"min: {min}\n")
            f.write(f"max: {max}\n")

    return normalized_path


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
        """Pre-compute valid appearance and geometry pair indices for each anchor."""
        n = len(self.colors)
        self.app_pairs = {}
        self.geo_pairs = {}

        for i in range(n):
            # Appearance pair: same size, different appearance (color OR peg_texture OR table_texture), similar geometry
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

            # Geometry pair: same appearance (color AND peg_texture AND table_texture), different size, similar context
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


    def _find_nearest(self, base_idx, candidates):
        """Find the nearest neighbor among candidates based on (delta_z, yaw_error)."""
        dz_diff = np.abs(self.delta_z[candidates] - self.delta_z[base_idx])
        yaw_diff = np.abs(self.yaw_error[candidates] - self.yaw_error[base_idx])
        dist = (dz_diff / self.eps_z) ** 2 + (yaw_diff / self.eps_yaw) ** 2
        return candidates[np.argmin(dist)]

    # def _sample_random(self, base_idx, candidates):
    #     """Randomly sample one candidate from the valid candidates."""
    #     return candidates[np.random.randint(len(candidates))]


    def sample_triplets(self, M):
        """Sample M anchor triplets (base, appearance_pair, geometry_pair).

        Returns:
            obs_base, obs_app, obs_geo: dicts of observation tensors on device.
            valid_count: number of valid triplets sampled.
        """
        if len(self.valid_anchors) == 0:
            return None, None, None, 0

        base_indices = np.random.choice(self.valid_anchors, size=M, replace=True)
        app_indices = np.array([self._find_nearest(bi, self.app_pairs[bi]) for bi in base_indices])
        geo_indices = np.array([self._find_nearest(bi, self.geo_pairs[bi]) for bi in base_indices])
        # app_indices = np.array([self._sample_random(bi, self.app_pairs[bi]) for bi in base_indices])
        # geo_indices = np.array([self._sample_random(bi, self.geo_pairs[bi]) for bi in base_indices])

        obs_base = self._build_obs_batch(base_indices)
        obs_app = self._build_obs_batch(app_indices)
        obs_geo = self._build_obs_batch(geo_indices)

        return obs_base, obs_app, obs_geo, M

    def _build_obs_batch(self, indices):
        """Build an observation dict batch from anchor indices."""
        ee_pos = torch.from_numpy(self.obs_ee_pos[indices]).float().to(self.device)
        ee_yaw = torch.from_numpy(self.obs_ee_yaw[indices]).float().to(self.device)
        # Images: uint8 HWC -> float CHW, normalised to [0, 1]
        images = torch.from_numpy(self.obs_images[indices].copy()).float().to(self.device)
        images = images / 255.0
        images = images.permute(0, 3, 1, 2)  # [M, C, H, W]
        return {
            "ee_pos_world": ee_pos,
            "ee_yaw_world": ee_yaw,
            "table_side_cam": images,
        }


def _compute_behavior_descriptor(policy, obs_dict, device, K=5, yaw_action_idx=3):
    """Compute behavior descriptor: cumulative |yaw| over K steps.

    For RNN policies, rolls out K steps with the same observation to get
    K different actions via evolving hidden state.
    For non-RNN policies, K steps with same input yield same output,
    so descriptor = K * |single-step yaw|.

    Returns:
        b: Tensor of shape [M], cumulative |yaw| descriptor per sample.
    """
    M = obs_dict["ee_pos_world"].shape[0]
    if hasattr(policy, "forward_step"):
        # RNN policy: roll out K steps
        rnn_state = policy.get_rnn_init_state(batch_size=M, device=device)
        cumulative_yaw = torch.zeros(M, device=device)
        for _ in range(K):
            actions, rnn_state = policy.forward_step(obs_dict, goal_dict=None, rnn_state=rnn_state)
            cumulative_yaw = cumulative_yaw + torch.abs(actions[:, yaw_action_idx])
    else:
        actions = policy(obs_dict=obs_dict, goal_dict=None)
        cumulative_yaw = K * torch.abs(actions[:, yaw_action_idx])
    return cumulative_yaw


def compute_geo_rel_loss(model, anchor_pool, M=16, margin=0.05, yaw_action_idx=3, K=5, log_diagnostics=False):
    """Compute geometric relational hinge loss from anchor triplets."""
    device = next(model.nets["policy"].parameters()).device
    obs_base, obs_app, obs_geo, valid_count = anchor_pool.sample_triplets(M)

    if valid_count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {}

    policy = model.nets["policy"]

    # Behavior descriptor: cumulative |yaw| over K steps
    b_base = _compute_behavior_descriptor(policy, obs_base, device, K=K, yaw_action_idx=yaw_action_idx)
    b_app = _compute_behavior_descriptor(policy, obs_app, device, K=K, yaw_action_idx=yaw_action_idx)
    b_geo = _compute_behavior_descriptor(policy, obs_geo, device, K=K, yaw_action_idx=yaw_action_idx)

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
    K=5,
    log_geo_diagnostics=False,
):
    """Run one training epoch with the standard BC loss plus geometric relational loss."""
    model.set_train()

    step_log_all = []
    data_loader_iter = iter(data_loader)
    if num_steps is None:
        num_steps = len(data_loader)

    for _ in range(num_steps):
        # --- STEP 1: standard BC forward + task loss ---
        try:
            batch = next(data_loader_iter)
        except StopIteration:
            data_loader_iter = iter(data_loader)
            batch = next(data_loader_iter)

        input_batch = model.process_batch_for_training(batch)
        input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=obs_normalization_stats)

        predictions = model._forward_training(input_batch)
        losses = model._compute_losses(predictions, input_batch)
        L_task = losses["action_loss"]

        # --- STEP 2-3: geometric relational loss ---
        L_rel, geo_diag = compute_geo_rel_loss(
            model=model,
            anchor_pool=anchor_pool,
            M=M,
            margin=margin,
            yaw_action_idx=yaw_action_idx,
            K=K,
            log_diagnostics=log_geo_diagnostics,
        )

        # --- STEP 3-4: combine and back-propagate ---
        L_total = L_task + lambda_rel * L_rel

        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=model.nets["policy"],
            optim=model.optimizers["policy"],
            loss=L_total,
        )

        # --- logging ---
        step_log = OrderedDict()
        step_log["Loss"] = L_task.item()
        step_log["L_rel"] = L_rel.item()
        step_log["L_total"] = L_total.item()
        if "l2_loss" in losses:
            step_log["l2_loss"] = losses["l2_loss"].item()
        if "l1_loss" in losses:
            step_log["l1_loss"] = losses["l1_loss"].item()
        if "cos_loss" in losses:
            step_log["cos_loss"] = losses["cos_loss"].item()
        step_log["Policy_Grad_Norms"] = policy_grad_norms
        for diag_k, diag_v in geo_diag.items():
            step_log[f"geo_{diag_k}"] = diag_v
        step_log_all.append(step_log)

    # average across all steps
    avg_log = OrderedDict()
    for k in step_log_all[0]:
        avg_log[k] = np.mean([s[k] for s in step_log_all])
    return avg_log


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
    behavior_K: int = 5,
):
    """Train a model using the algorithm specified in config.

    Args:
        config: Configuration object.
        device: PyTorch device to use for training.
        log_dir: Directory to save logs.
        ckpt_dir: Directory to save checkpoints.
        video_dir: Directory to save videos.
    """
    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")

    print(f">>> Saving logs into directory: {log_dir}")
    print(f">>> Saving checkpoints into directory: {ckpt_dir}")
    print(f">>> Saving videos into directory: {video_dir}")

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset at provided path {dataset_path} not found!")

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data, all_obs_keys=config.all_obs_keys, verbose=True
    )

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environment
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        # create environments for validation runs
        env_names = [env_meta["env_name"]]

        if config.experiment.additional_envs is not None:
            for name in config.experiment.additional_envs:
                env_names.append(name)

        for env_name in env_names:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
            )
            envs[env.name] = env
            print(envs[env.name])

    print("")

    # setup for a new training run
    data_logger = DataLogger(log_dir, config=config, log_tb=config.experiment.logging.log_tb)
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    # save the config as a json file
    with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    # maybe retrieve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
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

    # load anchor pool for geometric relational loss
    anchor_pool = AnchorPool(anchor_pool_path, device)

    # main training loop
    best_valid_loss = None
    last_ckpt_time = time.time()

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    for epoch in range(1, config.train.num_epochs + 1):  # epoch numbers start at 1
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
            K=behavior_K,
            log_geo_diagnostics=True,
        )
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = f"model_epoch_{epoch}"

        # check for recurring checkpoint saving conditions
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
            last_epoch_check = epoch == config.train.num_epochs
            should_save_ckpt = time_check or epoch_check or epoch_list_check or last_epoch_check
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

        # Evaluate the model on validation set
        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps
                )
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record(f"Timing_Stats/Valid_{k[5:]}", v, epoch)
                else:
                    data_logger.record(f"Valid/{k}", v, epoch)

            print(f"Validation Epoch {epoch}")
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += f"_best_validation_{best_valid_loss}"
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
            )

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print(f"\nEpoch {epoch} Memory Usage: {mem_usage} MB\n")

    # terminate logging
    data_logger.close()


def main(args: argparse.Namespace):
    """Train a model on a task using a specified algorithm.

    Args:
        args: Command line arguments.
    """
    # load config
    if args.task is not None:
        # obtain the configuration entry point
        cfg_entry_point_key = f"robomimic_{args.algo}_cfg_entry_point"
        task_name = args.task.split(":")[-1]

        print(f"Loading configuration for task: {task_name}")
        print(gym.envs.registry.keys())
        print(" ")
        cfg_entry_point_file = gym.spec(task_name).kwargs.pop(cfg_entry_point_key)
        # check if entry point exists
        if cfg_entry_point_file is None:
            raise ValueError(
                f"Could not find configuration for the environment: '{task_name}'."
                f" Please check that the gym registry has the entry point: '{cfg_entry_point_key}'."
            )

        # resolve module path if needed
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
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
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

    # change location of experiment directory
    config.train.output_dir = os.path.abspath(os.path.join("./logs", args.log_dir, args.task))

    log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config)

    if args.normalize_training_actions:
        config.train.data = normalize_hdf5_actions(config, log_dir)

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    config.lock()

    # catch error during training and print it
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
            behavior_K=args.behavior_K,
        )
    except Exception as e:
        res_str = f"run failed with error:\n{e}\n\n{traceback.format_exc()}"
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--algo", type=str, default=None, help="Name of the algorithm.")
    parser.add_argument("--log_dir", type=str, default="robomimic", help="Path to log directory")
    parser.add_argument("--normalize_training_actions", action="store_true", default=False, help="Normalize actions")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Optional: Number of training epochs. If specified, overrides the number of epochs from the JSON training"
            " config."
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
    parser.add_argument("--geo_rel_margin", type=float, default=0.3, help="Margin for hinge loss")
    parser.add_argument("--yaw_action_idx", type=int, default=3, help="Index of yaw component in action vector")
    parser.add_argument("--behavior_K", type=int, default=5, help="Number of rollout steps for behavior descriptor")

    args = parser.parse_args()

    # run training
    main(args)
    # close sim app
    simulation_app.close()
