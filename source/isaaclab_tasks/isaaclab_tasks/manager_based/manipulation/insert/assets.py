# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolution of the USD geometry assets used by the insertion task.

The pegs and hole blocks that the diagnostic benchmark varies are shipped with
the repository under ``usd_assets/``. Assets are looked up relative to the
repository root so a fresh clone works without editing any paths; set
``INSERT_ASSETS_DIR`` to point somewhere else.
"""

import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 6)))

INSERT_ASSETS_DIR = os.path.abspath(os.environ.get("INSERT_ASSETS_DIR", os.path.join(_REPO_ROOT, "usd_assets")))
"""Directory holding the insertion task's USD assets."""


def insert_asset(name: str) -> str:
    """Return the absolute path to a USD asset shipped with the insertion task.

    Args:
        name: File name of the asset, e.g. ``"HoleBlock.usd"``.

    Returns:
        Absolute path to the asset.

    Raises:
        FileNotFoundError: If the asset is missing, with a hint about
            ``INSERT_ASSETS_DIR``.
    """
    path = os.path.join(INSERT_ASSETS_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Insertion asset '{name}' not found at '{path}'. Set the INSERT_ASSETS_DIR environment"
            " variable if the USD assets live outside the repository."
        )
    return path
