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
    id="Isaac-Peg-Insert-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.insert_joint_pos_env_cfg:FrankaPegInsertEnvCfg",
    },
    disable_env_checker=True,
)

# gym.register(
#     id="Isaac-Peg-Insert-Instance-Randomize-Franka-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": (
#             f"{__name__}.stack_joint_pos_instance_randomize_env_cfg:FrankaCubeStackInstanceRandomizeEnvCfg"
#         ),
#     },
#     disable_env_checker=True,
# )


##
# Inverse Kinematics - Relative Pose Control
##

print(">>> [DEBUG] before gym.register Peg Insert <<<")

gym.register(
    id="Isaac-Peg-Insert-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    # entry_point="isaaclab_tasks.manager_based.manipulation.insert:PegInsertEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.insert_ik_rel_env_cfg:FrankaPegInsertEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
        "robomimic_diffusion_policy_cfg_entry_point": f"{agents.__name__}:robomimic/diffusion_policy_low_dim.json",
    },
    disable_env_checker=True,
)

print(">>> [DEBUG] after gym.register Peg Insert <<<")


gym.register(
    id="Isaac-Peg-Insert-Franka-IK-Rel-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.insert_ik_rel_visuomotor_env_cfg:FrankaPegInsertVisuomotorEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image_200.json",
        "robomimic_diffusion_policy_cfg_entry_point": f"{agents.__name__}:robomimic/diffusion_policy_image_200.json",
    },
    disable_env_checker=True,
)

# gym.register(
    # id="Isaac-Peg-Insert-Franka-IK-Rel-Visuomotor-Cosmos-v0",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    # kwargs={
    #     "env_cfg_entry_point": (
    #         f"{__name__}.stack_ik_rel_visuomotor_cosmos_env_cfg:FrankaCubeStackVisuomotorCosmosEnvCfg"
    #     ),
    #     "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image_cosmos.json",
    # },
    # disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-Peg-Insert-Franka-IK-Abs-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.stack_ik_abs_env_cfg:FrankaCubeStackEnvCfg",
#         "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-Peg-Insert-Instance-Randomize-Franka-IK-Rel-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": (
#             f"{__name__}.stack_ik_rel_instance_randomize_env_cfg:FrankaCubeStackInstanceRandomizeEnvCfg"
#         ),
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-Peg-Insert-Franka-IK-Rel-Blueprint-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.stack_ik_rel_blueprint_env_cfg:FrankaCubeStackBlueprintEnvCfg",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-Peg-Insert-Franka-IK-Rel-Skillgen-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.stack_ik_rel_env_cfg_skillgen:FrankaCubeStackSkillgenEnvCfg",
#         "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-Peg-Insert-Bin-Franka-IK-Rel-Mimic-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.bin_stack_ik_rel_env_cfg:FrankaBinStackEnvCfg",
#         "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
#     },
#     disable_env_checker=True,
# )
