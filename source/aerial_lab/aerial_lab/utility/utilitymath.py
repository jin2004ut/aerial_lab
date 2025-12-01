import math

import torch
from isaaclab.utils.math import quat_from_angle_axis, quat_from_euler_xyz


def samlpeUniformQuatwithTilt(tile: torch.Tensor, size: int) -> torch.Tensor:
    """Sample uniform quaternions with tilt angle limit.

    Args:
        tile (torch.Tensor): Tilt angle limit in radians.
        size (int): Number of quaternions to sample.

    Returns:
        torch.Tensor: Sampled quaternions of shape (size, 4).
    """
    phi = torch.rand((size), device=tile.device) * 2.0 * math.pi
    theta = torch.rand((size), device=tile.device) * tile
    axisZ = torch.zeros((size, 3), device=tile.device)
    xyL = torch.sin(theta)
    # import ipdb; ipdb.set_trace()
    axisZ[:, 2] = torch.cos(theta)
    axisZ[:, 0] = xyL * torch.cos(phi)
    axisZ[:, 1] = xyL * torch.sin(phi)

    # print("axisZ:\n", axisZ)
    # print("norm axisZ:\n", torch.linalg.norm(axisZ, dim=-1))

    X_euler = torch.asin(-axisZ[:, 0])
    Y_euler = torch.atan2(axisZ[:, 1], axisZ[:, 2])
    # print("X_euler:\n", X_euler)
    # print("Y_euler:\n", Y_euler)
    Z_euler = torch.rand((size), device=tile.device) * 2.0 * math.pi
    # print("Z_euler:\n", Z_euler)
    quats = quat_from_euler_xyz(roll=X_euler, pitch=Y_euler, yaw=Z_euler)
    # print("quats:\n", quats)
    return quats


if __name__ == "__main__":
    tile = torch.tensor(math.pi * 0.5)
    size = 10
    quats = samlpeUniformQuatwithTilt(tile, size)
    print("Sampled quaternions with tilt limit:\n", quats)
