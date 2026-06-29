# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the kinikun tutorial track."""

from __future__ import annotations

import copy

import isaaclab.sim as sim_utils
import torch
from aerial_lab.assets.aerialrobot import KINIKUN_CFG
from isaaclab.assets import Articulation
from isaaclab.sensors import ImuCfg
from isaaclab.utils.math import subtract_frame_transforms


def resolve_kinikun_cfg(enable_gravity: bool | None = None):
    cfg = copy.deepcopy(KINIKUN_CFG)
    if enable_gravity is not None:
        cfg.spawn.rigid_props.disable_gravity = not enable_gravity
    return cfg.replace(prim_path="/World/Robot")


def resolve_camera_view() -> tuple[list[float], list[float]]:
    return [2.1, 2.1, 1.5], [0.0, 0.0, 0.8]


def spawn_ground_and_light():
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

    spawn_ground_plane("/World/ground", GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)


def reset_robot(robot: Articulation, spawn_height: float = 0.8):
    root_state = robot.data.default_root_state.clone()
    root_state[:, 2] = spawn_height
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()


def build_imu_sensor_cfg(robot: Articulation) -> ImuCfg:
    body_id = robot.find_bodies("main_body")[0]
    imu_id = robot.find_bodies("fc")[0]

    body_pos_w = robot.data.body_pos_w[:, body_id]
    body_quat_w = robot.data.body_quat_w[:, body_id]
    imu_pos_w = robot.data.body_pos_w[:, imu_id]
    imu_quat_w = robot.data.body_quat_w[:, imu_id]

    imu_pos_body, imu_quat_body = subtract_frame_transforms(body_pos_w, body_quat_w, imu_pos_w, imu_quat_w)

    return ImuCfg(
        prim_path="/World/Robot/fc",
        update_period=0,
        debug_vis=False,
        offset=ImuCfg.OffsetCfg(
            pos=tuple(float(v) for v in imu_pos_body[0].detach().cpu().tolist()),
            rot=tuple(float(v) for v in imu_quat_body[0].detach().cpu().tolist()),
        ),
    )


def fmt_tensor_row(tensor: torch.Tensor, decimals: int = 4) -> str:
    values = tensor.detach().cpu().tolist()
    return "[" + ", ".join(f"{value:.{decimals}f}" for value in values) + "]"
