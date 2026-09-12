# Datasets

400+ human teleoperation demonstrations collected in Isaac Lab, packaged per
diagnostic experiment. Roughly 10.1 GB in total, so they are hosted on Hugging
Face rather than committed here.

**Download:** https://huggingface.co/datasets/CrocoWilly/geometry-conditioned-insertion

```bash
pip install huggingface_hub
hf download CrocoWilly/geometry-conditioned-insertion \
    --repo-type dataset --local-dir datasets/
```

Individual files can be fetched on their own — the Diagnosis sets are
independent, so there is no need to pull all 10.1 GB to run one experiment.

| File | Episodes | Size | Experiment |
|---|---|---|---|
| `Diagnosis0_dataset_forTraining.hdf5` | 50 | 1.1 GB | Skill Check |
| `Diagnosis1_dataset_forTraining.hdf5` | 50 | 1.1 GB | Alignment Adaptation |
| `Diagnosis2_dataset_forTraining.hdf5` | 50 | 1.3 GB | Strategy Diversity |
| `Diagnosis3_dataset_forTraining.hdf5` | 60 | 1.6 GB | Strategy Disambiguation |
| `Diagnosis4_dataset_forTraining.hdf5` | 90 | 1.9 GB | Hole Identification |
| `MainExp_dataset_forTraining.hdf5` | 120 | 2.6 GB | Appearance x geometry sweep |
| `MainExp_dataset_anchorPool_forTraining.hdf5` | — | 304 MB | Anchor pool for the relative loss |

At least 50 demonstrations per task.

Sample recordings of the human teleoperation, one per task:
[Diagnosis 2](../images/Diagnosis2_humanDemosSample.mp4) ·
[Diagnosis 3](../images/Diagnosis3_humanDemosSample.mp4) ·
[Diagnosis 4](../images/Diagnosis4_humanDemosSample.mp4)

## Format

Standard robomimic HDF5 layout: a `data` group containing one subgroup per
episode.

```
data/
  demo_0/
    obs/
      table_side_cam     (T, 200, 200, 3)  uint8   side-view RGB
      ee_pos_world       (T, 3)            float32 end-effector position, world frame
      ee_yaw_world       (T, 1)            float32 end-effector yaw, world frame
    action_world_4d      (T, 4)            float32 [dx, dy, dz, dyaw] in the world frame
    actions              (T, 4)            float32 a copy of action_world_4d, which
                                                   is the key robomimic trains on
  demo_1/
  ...
```

Episodes also carry the geometry and appearance parameters they were generated
under — hole clearance, yaw offset, box color, peg texture, background — which
is what makes the appearance/geometry pairing in the relative loss possible.

Inspect any file with:

```bash
python datasets/Check_hdf5_structure.py <dataset.hdf5>
python datasets/List_episodes.py
python datasets/View_hdf5_images.py <dataset.hdf5> --demo 0 --fps 15
```

## Anchor pool

A separate HDF5 built from `MainExp_dataset_forTraining.hdf5`, holding the
**2,578 decision-critical time-steps** the relative loss samples from — the
moments where the peg is close to the hole surface and still misaligned.

```
anchors/
  anchor_0/
    obs/
      table_side_cam   (200, 200, 3)
      ee_pos_world     (3,)
      ee_yaw_world     (1,)
    attrs: color, peg_texture, table_texture, size, delta_z, yaw_error_deg
  anchor_1/
  ...
attrs: source_file, z_diff_min, z_diff_max, yaw_error_threshold_deg,
       peg_length, hole_height, total_anchors
```

Regenerate rather than download, if preferred:

```bash
python datasets/behaviorAnalysis_scripts/MainExp_createAnchorPool.py \
    --input datasets/MainExp_dataset_forTraining.hdf5 \
    --yaw-error-threshold 3.5 --z-diff-max 0.0125
```

This writes both the HDF5 (consumed by training) and a CSV of the same metadata.

## Results

The thesis figures are in [`images/`](../images). Intermediate rollout logs are
not committed — regenerate them with the `play_4d_and_save*` scripts and score
them with the analysis suite, as described in [setup.md](setup.md) and
[benchmark.md](benchmark.md).

## Tooling

| Script | Purpose |
|---|---|
| `Check_hdf5_structure.py` | Full structure dump; run before and after a merge |
| `Inspect_hdf5.py`, `List_episodes.py` | Quick listings |
| `View_hdf5_images.py` | Play back an episode's camera frames |
| `Merge_hdf5_datasets.py` | Combine collection sessions |
| `PostProcess_Add4DAction_and_ScalingClearanceAndDepth.py` | Attach the 4D action and per-episode clearance/depth scaling |
| `Copy_action_world_4d_to_actions.py` | Point robomimic's `actions` key at the 4D action |
| `Convert_hdf5ToJson.py` | Export as JSON for inspection (low-dim data only) |
| `count_bc_rnn_params.py` | Parameter count excluding the pretrained ResNet18 backbone |

The two-step conversion from the recorded 7D action to the 4D action robomimic
trains on is described in [setup.md](setup.md#dataset-preparation).

## License

Released under the same BSD-3-Clause license as the code. The demonstrations
were collected by the author in simulation; no third-party or human-subject data
is involved.
