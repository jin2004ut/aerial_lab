from isaaclab.app import AppLauncher

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from isaaclab.utils.math import (
    compute_pose_error,
    matrix_from_quat,
    normalize,
    quat_apply,
    quat_conjugate,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
)

eulur_xyz = torch.tensor([0.0, 0.0, torch.pi / 2.0])
quat = quat_from_euler_xyz(roll=eulur_xyz[0], pitch=eulur_xyz[1], yaw=eulur_xyz[2])
print("quat:", quat)
rot_vec = matrix_from_quat(quat)[:2, :]
print("rot_vec:", rot_vec)
