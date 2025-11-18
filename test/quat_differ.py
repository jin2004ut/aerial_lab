from isaaclab.app import AppLauncher

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app

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


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention


@torch.jit.script
def so3_log_rotation(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    object_R = matrix_from_quat(object_rot)
    target_R = matrix_from_quat(target_rot)
    R_diff = torch.matmul(target_R.transpose(-2, -1), object_R)
    cos_theta = (torch.trace(R_diff) - 3) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    if theta < 1e-6:
        # small angle: use approximation
        return 0.5 * torch.tensor(
            [R_diff[2, 1] - R_diff[1, 2], R_diff[0, 2] - R_diff[2, 0], R_diff[1, 0] - R_diff[0, 1]]
        )

    # normal case
    return (theta / (2 * torch.sin(theta))) * torch.tensor(
        [R_diff[2, 1] - R_diff[1, 2], R_diff[0, 2] - R_diff[2, 0], R_diff[1, 0] - R_diff[0, 1]]
    )


if __name__ == "__main__":
    quat_1 = normalize(torch.randn(2, 4))
    quat_2 = normalize(torch.randn(2, 4))
    pos_1 = torch.randn(2, 3)
    pos_2 = torch.randn(2, 3)

    dist = rotation_distance(quat_1, quat_2)
    print("Rotation distance:", dist)
    pose_err, axis_angle_error = compute_pose_error(
        pos_1,
        quat_1,
        pos_2,
        quat_2,
    )
    print("Axis angle error norm:", torch.linalg.norm(axis_angle_error, dim=1))
    print("Axis angle error:", axis_angle_error)
