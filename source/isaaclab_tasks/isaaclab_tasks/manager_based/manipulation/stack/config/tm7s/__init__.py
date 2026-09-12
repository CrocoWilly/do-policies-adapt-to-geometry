# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Stack-Cube-Tm7s-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_joint_pos_env_cfg:Tm7sCubeStackEnvCfg",
    },
    disable_env_checker=True,
)

# gym.register(
#     id="Isaac-Stack-Cube-Instance-Randomize-Tm7s-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": (
#             f"{__name__}.stack_joint_pos_instance_randomize_env_cfg:Tm7sCubeStackInstanceRandomizeEnvCfg"
#         ),
#     },
#     disable_env_checker=True,
# )


##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Stack-Cube-Tm7s-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_ik_rel_env_cfg:Tm7sCubeStackEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)

# gym.register(
#     id="Isaac-Stack-Cube-Tm7s-IK-Rel-Visuomotor-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.stack_ik_rel_visuomotor_env_cfg:Tm7sCubeStackVisuomotorEnvCfg",
#         "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image_200.json",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-Stack-Cube-Tm7s-IK-Rel-Visuomotor-Cosmos-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": (
#             f"{__name__}.stack_ik_rel_visuomotor_cosmos_env_cfg:Tm7sCubeStackVisuomotorCosmosEnvCfg"
#         ),
#         "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image_cosmos.json",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-Stack-Cube-Tm7s-IK-Abs-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.stack_ik_abs_env_cfg:Tm7sCubeStackEnvCfg",
#         "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-Stack-Cube-Instance-Randomize-Tm7s-IK-Rel-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": (
#             f"{__name__}.stack_ik_rel_instance_randomize_env_cfg:Tm7sCubeStackInstanceRandomizeEnvCfg"
#         ),
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-Stack-Cube-Tm7s-IK-Rel-Blueprint-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.stack_ik_rel_blueprint_env_cfg:Tm7sCubeStackBlueprintEnvCfg",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-Stack-Cube-Tm7s-IK-Rel-Skillgen-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.stack_ik_rel_env_cfg_skillgen:Tm7sCubeStackSkillgenEnvCfg",
#         "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-Stack-Cube-Bin-Tm7s-IK-Rel-Mimic-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.bin_stack_ik_rel_env_cfg:Tm7sBinStackEnvCfg",
#         "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
#     },
#     disable_env_checker=True,
# )
