# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# NOTE:
# Keep the package import side-effect small.
# Importing the sample UI extension here makes every plain `import aerial_lab`
# pull in Omniverse UI modules, which is unnecessary for training/tutorial
# scripts and can destabilize GUI startup. The example extension stays
# available via `aerial_lab.ui_extension_example` when explicitly needed.
