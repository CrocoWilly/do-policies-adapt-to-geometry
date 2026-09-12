"""
Count the model parameters of bc_rnn_image_200.json, excluding the pretrained ResNet18 backbone.
"""
import os
import sys
import json
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

CONFIG_PATH = os.environ.get(
    "BC_RNN_CONFIG",
    os.path.join(
        REPO_ROOT,
        "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/insert"
        "/config/franka/agents/robomimic/bc_rnn_image_200.json",
    ),
)

def main():
    # ── Load robomimic ────────────────────────────────────────────────────────────────────
    try:
        import robomimic.utils.obs_utils as ObsUtils
        from robomimic.config import config_factory
        from robomimic.algo import algo_factory
    except ImportError:
        print("[ERROR] robomimic not found. Please activate the IsaacLab env first.")
        sys.exit(1)

    # ── Parse the config ──────────────────────────────────────────────────────────────────
    with open(CONFIG_PATH) as f:
        ext_cfg = json.load(f)

    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)

    ObsUtils.initialize_obs_utils_with_config(config)
    config.lock()

    # ── Provide shape_meta manually (read from the env cfg) ───────────────────────────────
    # table_side_cam: 200×200 RGB → shape (3, 200, 200)
    # ee_pos_world: (3,)   ee_yaw_world: (1,)
    # ac_dim: 4  (Δx, Δy, Δz, Δyaw in hole frame)
    shape_meta = {
        "ac_dim": 4,
        "all_shapes": {
            "ee_pos_world":   (3,),
            "ee_yaw_world":   (1,),
            "table_side_cam": (3, 200, 200),
        },
        "use_images": True,
        "use_depths": False,
    }

    # ── Build the model ───────────────────────────────────────────────────────────────────
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device="cpu",
    )

    policy = model.nets["policy"]

    # ── All trainable parameters ──────────────────────────────────────────────────────────
    all_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    # ── Find the ResNet18 backbone parameters (pretrained, excluded) ──────────────────────
    resnet_params = 0
    resnet_module = None
    for name, module in policy.named_modules():
        cls_name = type(module).__name__
        if cls_name == "ResNet18Conv":
            resnet_module = module
            resnet_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"[ResNet18Conv]  found at '{name}',  params = {resnet_params:>12,}")
            break

    non_resnet = all_trainable - resnet_params

    # ── Per-module breakdown ──────────────────────────────────────────────────────────────
    print("\n── Parameter counts per submodule, excluding the ResNet backbone ─────────")
    for name, module in policy.named_modules():
        # Skip ResNet18Conv and its submodules
        if resnet_module is not None:
            # Check whether module is resnet_module itself or one of its submodules
            try:
                is_child = any(
                    module is m for m in resnet_module.modules()
                )
            except Exception:
                is_child = False
            if is_child:
                continue

        own_params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        if own_params > 0:
            print(f"  {name:<60s}  {own_params:>10,}")

    print("\n── Summary ───────────────────────────────────────────────────────────────")
    print(f"  All trainable parameters (incl. ResNet backbone) : {all_trainable:>12,}")
    print(f"  ResNet18 backbone (pretrained)      : {resnet_params:>12,}")
    print(f"  Non-pretrained trainable parameters (target) : {non_resnet:>12,}")


if __name__ == "__main__":
    main()
