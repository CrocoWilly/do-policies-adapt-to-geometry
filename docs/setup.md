# Setup and usage

## Hardware and OS

Everything in the thesis was produced on a single workstation:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24 GB |
| Driver / CUDA | 580.159.03 / CUDA 13.0 |
| OS | Ubuntu 22.04, kernel 6.8.0-111-generic |

Exact package lists are in [`env_setup/`](env_setup): `nvidia_smi.txt`,
`nvidia_packages.txt`, `kernel.txt`.

## Two conda environments

Behavior cloning and Diffusion Policy need **different robomimic versions**, so
they cannot share one environment. Diffusion Policy support arrived in robomimic
0.5.0 and pulls in `diffusers`; the BC pipeline was built against 0.4.0.

| Environment | robomimic | Python | Used for |
|---|---|---|---|
| `env_isaaclab` | 0.4.0 | 3.11.14 | demonstration collection, BC-LSTM training and inference |
| `diffusion_env_isaaclab` | 0.5.0 + `diffusers` 0.11.1 | 3.11.15 | Diffusion Policy training and inference |

Both use `torch` 2.7.0+cu128 and `numpy` 1.26.0.

```bash
conda env create -f docs/env_setup/bc_conda_env.yml   # env_isaaclab
conda env create -f docs/env_setup/dp_conda_env.yml   # diffusion_env_isaaclab
```

**Demonstrations are always collected in `env_isaaclab`**, including the data
later used to train Diffusion Policy. Only training and inference move to the
other environment.

## Installation

This repository is an **overlay** on Isaac Lab. Its `source/` and `scripts/`
trees mirror Isaac Lab's own layout, so installing means copying them over a
checkout. Developed against Isaac Lab at commit `f9154913b7`.

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
git clone https://github.com/CrocoWilly/do-policies-adapt-to-geometry.git

rsync -a --exclude '.git' --exclude 'README.md' --exclude 'docs' \
      do-policies-adapt-to-geometry/ IsaacLab/

cd IsaacLab && ./isaaclab.sh --install
```

Only files this work added or changed are included; everything else comes from
Isaac Lab itself.

Verify the tasks registered:

```bash
./isaaclab.sh -p -c "import isaaclab_tasks, gymnasium as gym; \
    print([k for k in gym.registry if 'Peg-Insert' in k])"
```

## Registered tasks

| Task ID | Observations |
|---|---|
| `Isaac-Peg-Insert-Franka-v0` | explicit, joint-position control |
| `Isaac-Peg-Insert-Franka-IK-Rel-v0` | explicit ("low dim"), relative IK |
| `Isaac-Peg-Insert-Franka-IK-Rel-Visuomotor-v0` | implicit (RGB), relative IK |

The action space is `[dx, dy, dz, dyaw]`. Roll and pitch are fixed and the
gripper stays closed on the peg, so the policy controls exactly the four degrees
of freedom the insertion needs.

## USD assets

Peg and hole-block geometry lives in `usd_assets/` and resolves automatically
relative to the repository root through `insert_asset()`. Override with
`INSERT_ASSETS_DIR`. A missing asset raises `FileNotFoundError` naming that
variable rather than failing inside the simulator.

## Selecting a diagnostic experiment

Each experiment is a different peg/hole geometry and yaw range. Switching between them
means editing
`source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/insert/config/franka/insert_joint_pos_env_cfg.py`
and uncommenting the block labeled with the diagnosis number.

| Diagnosis | `usd_path` (hole) | `yaw_range` |
|---|---|---|
| 0 — Skill Check | `HoleBlock.usd` | `(-0.2618, 0.2618)` — ±15° |
| 1 — Alignment Adaptation | `HoleBlock.usd`, vary `self.clearance` | fixed, e.g. `(-0.2618, -0.2617)` |
| 1 — generalized | `HoleBlock_..._hole{103,124,136,148,160}.usd` | as above |
| 2 — Strategy Diversity | `XHoleBlock.usd` | `(-0.7854, -0.7853)` — −45° |
| 3 — Strategy Disambiguation | `BigSmallXHoleBlock_case{1,2}.usd` | `(-1.5708, 1.4835)` — −90°..+85° |
| 4 — Hole Identification | `ThreeShapeHoleBlock_{1..6}.usd` | `(0.0, 0.0)` |

For Diagnosis 4 the peg also changes — `SquareCube.usd`, `CirclePeg.usd`,
`RealSquarePeg.usd` or `RectanglePeg.usd`. Hole clearance for Diagnosis 1 is set
by `self.clearance` (default `1.12`, the tight hole).

> **Diagnoses 2 and 3 need `--scale_yaw 0.2` at rollout time.** Both involve
> ±45° or larger reorientations, which the default yaw scaling cannot reach
> within the episode budget. Forgetting this makes a capable policy look like it
> failed.

> This is the one place the repository still carries its research-code shape.
> Parameterizing the selection behind a `--diagnosis` flag is a known
> improvement, deferred because it cannot be verified without a working Isaac
> Sim install.

---

# Pipelines

## BC-LSTM

Run entirely in `env_isaaclab`.

### 1. Collect demonstrations

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-Peg-Insert-Franka-IK-Rel-Visuomotor-v0 \
    --device cpu --enable_cameras --teleop_device keyboard \
    --dataset_file ./datasets/my_demos.hdf5 --num_demos 3
```

this is what the published datasets were collected with.

`record_demos_boost.py` boost the teleoperation speed, which makes the colleting process faster.

Teleoperation is tuned for insertion: position sensitivity 0.0125 for fine
alignment, recording at 25 Hz.

*Note: the collected datasets need to be pre-processed before training. The pre-processing process is descipted in [datasets.md](datasets.md).

### 2. Train

```bash
# explicit observations
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
    --task Isaac-Peg-Insert-Franka-IK-Rel-v0 --algo bc \
    --dataset ./datasets/Diagnosis1_dataset_forTraining.hdf5

# implicit observations (RGB)
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
    --task Isaac-Peg-Insert-Franka-IK-Rel-Visuomotor-v0 --algo bc \
    --dataset ./datasets/Diagnosis1_dataset_forTraining.hdf5

# with the relative geometry-appearance loss
./isaaclab.sh -p scripts/imitation_learning/robomimic/train_bc_with_geoRelLoss_mainExp.py \
    --task Isaac-Peg-Insert-Franka-IK-Rel-Visuomotor-v0 --algo bc \
    --dataset ./datasets/MainExp_dataset_forTraining.hdf5
```

See [relative-loss.md](relative-loss.md) for the anchor pool the third command
needs and for `--lambda_rel`, `--geo_rel_M` and `--geo_rel_margin`.

### 3. Roll out

```bash
# watch it run
./isaaclab.sh -p scripts/imitation_learning/robomimic/play_4d.py \
    --task Isaac-Peg-Insert-Franka-IK-Rel-Visuomotor-v0 \
    --device cpu --enable_cameras --num_rollouts 5 \
    --checkpoint /path/to/model.pth

# 100 rollouts, saved for analysis
./isaaclab.sh -p scripts/imitation_learning/robomimic/play_4d_and_save.py \
    --task Isaac-Peg-Insert-Franka-IK-Rel-Visuomotor-v0 \
    --device cpu --enable_cameras --num_rollouts 100 \
    --dataset_file ./datasets/rollout100_clearance112_yaw0.hdf5 \
    --checkpoint /path/to/model.pth
```

For Diagnoses 2 and 3, add `--scale_yaw 0.2`.

## Diffusion Policy

Collect in `env_isaaclab`, then **switch to `diffusion_env_isaaclab`** for
training and inference.

### 2. Train

```bash
# explicit observations
./isaaclab.sh -p scripts/imitation_learning/robomimic/train_dp.py \
    --task Isaac-Peg-Insert-Franka-IK-Rel-v0 --algo diffusion_policy \
    --dataset ./datasets/Diagnosis1_dataset_forTraining.hdf5

# implicit observations
./isaaclab.sh -p scripts/imitation_learning/robomimic/train_dp.py \
    --task Isaac-Peg-Insert-Franka-IK-Rel-Visuomotor-v0 --algo diffusion_policy \
    --dataset ./datasets/Diagnosis1_dataset_forTraining.hdf5

# with the relative geometry-appearance loss
./isaaclab.sh -p scripts/imitation_learning/robomimic/train_dp_with_geoRelLoss_mainExp.py \
    --task Isaac-Peg-Insert-Franka-IK-Rel-Visuomotor-v0 --algo diffusion_policy \
    --dataset ./datasets/MainExp_dataset_forTraining.hdf5
```

### 3. Roll out

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/play_4d_dp.py \
    --task Isaac-Peg-Insert-Franka-IK-Rel-Visuomotor-v0 \
    --device cpu --enable_cameras --num_rollouts 10 \
    --checkpoint /path/to/model.pth

./isaaclab.sh -p scripts/imitation_learning/robomimic/play_4d_and_save_dp.py \
    --task Isaac-Peg-Insert-Franka-IK-Rel-Visuomotor-v0 \
    --device cpu --enable_cameras --num_rollouts 100 \
    --dataset_file ./datasets/DP_rollout100.hdf5 --checkpoint /path/to/model.pth
```

Again, `--scale_yaw 0.2` for Diagnoses 2 and 3.
Scripts whose name contains `dp` are the Diffusion Policy variants; the rest are
for BC-LSTM.

### Which script for which policy

| | BC-LSTM | Diffusion Policy |
|---|---|---|
| train | `train.py --algo bc` | `train_dp.py --algo diffusion_policy` |
| train + rel. loss | `train_bc_with_geoRelLoss_mainExp.py` | `train_dp_with_geoRelLoss_mainExp.py` |
| roll out | `play_4d.py` | `play_4d_dp.py` |
| roll out + save | `play_4d_and_save.py` | `play_4d_and_save_dp.py` |

---

## Dataset preparation

The environment records a 7D action `[x, y, z, roll, pitch, yaw, gripper]`, but
every task here uses only `[x, y, z, yaw]`. Two passes convert a freshly
collected dataset into the 4D form robomimic trains on.

Both scripts take their input and output paths as **edits inside the file**, not
command-line flags.

```
STEP 0.  Collect with record_demos.py / record_demos_boost.py

STEP 1.  PostProcess_Add4DAction_and_ScalingClearanceAndDepth.py
         in:  SmallExp2_dataset0_Clearance120.hdf5
         out: SmallExp2_dataset0_Clearance120_with4DAction.hdf5
         Adds the 4D action and attaches per-episode clearance and depth scaling.

STEP 2.  Copy_action_world_4d_to_actions.py
         in:  ..._with4DAction.hdf5
         out: ..._with4DAction_forTraining.hdf5
         Points robomimic's `actions` key at the 4D action.
```

### Inspecting a dataset

```bash
python datasets/List_episodes.py               # episode list
python datasets/Inspect_hdf5.py                # quick structure
python datasets/Check_hdf5_structure.py --files T1.hdf5
python datasets/View_hdf5_images.py Test_dataset.hdf5   # play back camera frames
```

`Convert_hdf5ToJson.py` dumps the whole file as JSON. It will not finish on
datasets containing images — use it on low-dim data only.

### Merging collection sessions

Check structure, merge, then check again: `env_args` and the per-episode layout
must match exactly or training will fail in confusing ways.

```bash
python datasets/Check_hdf5_structure.py --files T1.hdf5 T2.hdf5
python datasets/Merge_hdf5_datasets.py --input_files T1.hdf5 T2.hdf5 --output_file merged.hdf5
python datasets/Check_hdf5_structure.py --files T1.hdf5 T2.hdf5 merged.hdf5
```

---

## Scoring rollouts

```bash
python datasets/behaviorAnalysis_scripts/Diagnosis_successRate_completionTime.py --input <rollout.hdf5>
python datasets/behaviorAnalysis_scripts/Diagnosis_alignment.py --input <rollout.csv>
python datasets/behaviorAnalysis_scripts/Diagnosis_insertion.py --input <rollout.csv>
```

See [benchmark.md](benchmark.md) for the metric definitions and the individual
`BehaviorAnalysis_Calc*.py` scripts.

---

## Troubleshooting

**`FileNotFoundError` naming `INSERT_ASSETS_DIR`.** The USD assets were not
found. Check `usd_assets/` came across in the overlay, or set the variable.

**A Diagnosis 2 or 3 policy never reaches the hole.** Missing `--scale_yaw 0.2`.

**`diffusion_policy` is not a known algo.** Wrong environment — Diffusion Policy
training needs `diffusion_env_isaaclab` with robomimic 0.5.0.

**Observation shape errors during rollout.** `torch.squeeze()` collapses the
feature dimension as well as the env dimension when rolling out a single
environment. `play.py` and the `play_4d*` scripts already remove only the env
dimension; carry that fix over when adapting another script.

**The peg drifts sideways while grasped.** Known issue in the "Setting 2" arm
pose in `insert_joint_pos_env_cfg.py`, which is commented out for that reason.
Use Setting 1.
