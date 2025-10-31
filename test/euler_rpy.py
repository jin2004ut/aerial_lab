from isaaclab.app import AppLauncher

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app

import numpy as np
import math
import torch
from isaaclab.utils.math import (
    compute_pose_error,
    matrix_from_quat,
    normalize,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    matrix_from_euler,
    quat_apply,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
    quat_conjugate,
)


if __name__ == "__main__":

    ZYX_euler_angles = torch.tensor([[-math.pi * 0.5, -math.pi * 0.5, -math.pi], [math.pi * 0.5, math.pi * 0.5, math.pi]])  # RPY
    print("ZYX Euler angles:\n", ZYX_euler_angles)

    ZYXMatrix = matrix_from_euler(ZYX_euler_angles, "ZYX")
    print("Rotation matrices from ZYX Euler angles:\n", ZYXMatrix)

    ZYX_quat = quat_from_euler_xyz(ZYX_euler_angles)
    print("Quaternions from ZYX Euler angles:\n", ZYX_quat)

    XYZ_euler_angles = torch.tensor([[-math.pi, -math.pi * 0.5, -math.pi * 0.5], [math.pi, math.pi * 0.5, math.pi * 0.5]])  # YPR
    print("XYZ Euler angles:\n", XYZ_euler_angles)

    XYZMatrix = matrix_from_euler(XYZ_euler_angles, "XYZ")
    print("Rotation matrices from XYZ Euler angles:\n", XYZMatrix)

    XYZ_quat = quat_from_euler_xyz(XYZ_euler_angles)
    print("Quaternions from XYZ Euler angles:\n", XYZ_quat)
