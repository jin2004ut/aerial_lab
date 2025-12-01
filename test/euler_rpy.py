from isaaclab.app import AppLauncher

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app

import math

import numpy as np
import torch
from aerial_lab.utility.utilitymath import samlpeUniformQuatwithTilt
from isaaclab.utils.math import (
    compute_pose_error,
    matrix_from_euler,
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

if __name__ == "__main__":

    ZYX_euler_angles = torch.tensor(
        [[-math.pi * 0.5, -math.pi * 0.5, -math.pi], [math.pi * 0.5, math.pi * 0.5, math.pi]]
    )  # RPY
    print("ZYX Euler angles:\n", ZYX_euler_angles)

    ZYXMatrix = matrix_from_euler(ZYX_euler_angles, "ZYX")
    print("Rotation matrices from ZYX Euler angles:\n", ZYXMatrix)

    ZYX_quat = quat_from_euler_xyz(
        roll=ZYX_euler_angles[:, 0], pitch=ZYX_euler_angles[:, 1], yaw=ZYX_euler_angles[:, 2]
    )
    print("Quaternions from ZYX Euler angles:\n", ZYX_quat)

    XYZ_euler_angles = torch.tensor(
        [[-math.pi, -math.pi * 0.5, -math.pi * 0.5], [math.pi, math.pi * 0.5, math.pi * 0.5]]
    )  # YPR
    print("XYZ Euler angles:\n", XYZ_euler_angles)

    XYZMatrix = matrix_from_euler(XYZ_euler_angles, "XYZ")
    print("Rotation matrices from XYZ Euler angles:\n", XYZMatrix)

    XYZ_quat = quat_from_euler_xyz(
        roll=XYZ_euler_angles[:, 0], pitch=XYZ_euler_angles[:, 1], yaw=XYZ_euler_angles[:, 2]
    )
    print("Quaternions from XYZ Euler angles:\n", XYZ_quat)

    tile = torch.tensor(math.pi * 0.5)
    size = 3
    quats = samlpeUniformQuatwithTilt(tile, size)
    print("Sampled quaternions with tilt limit:\n", quats)
