# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare root-state quantities with IMU sensor outputs.

This tutorial is the practical follow-up to `05_add_imu_sensor.py`.
It prints the values side by side and also shows the difference between them.

Why this matters:

- `robot.data.root_*` is tied to the robot root body state.
- `imu_sensor.data.*` is tied to the IMU sensor frame and offset.
- If the IMU is not exactly at the root, the values can differ.

This file is closely related to the commented IMU debug blocks in:

- `tasks/direct/quadcopter/mini_quadcopter_env.py`
- `tasks/direct/beetle/beetle_env.py`
- `tasks/direct/beetle/beetle_omni_env.py`
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare root-state signals with IMU sensor outputs.")
parser.add_argument(
    "--robot",
    type=str,
    default="mini_quad",
    choices=["mini_quad", "beetle", "beetle_omni"],
    help="Robot asset to spawn.",
)
parser.add_argument("--steps", type=int, default=240, help="Number of simulation steps to run.")
parser.add_argument("--print_every", type=int, default=60, help="How often to print the comparison.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import torch
from aerial_lab.assets.aerialrobot import BEETLE_CFG, BEETLE_OMNI_CFG, MINI_QUADROTOR_CFG
from isaaclab.assets import Articulation
from isaaclab.sensors import Imu, ImuCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, subtract_frame_transforms


def resolve_camera_view(name: str) -> tuple[list[float], list[float]]:
    camera = {
        "mini_quad": ([2.0, 2.0, 1.4], [0.0, 0.0, 0.6]),
        "beetle": ([2.4, 2.4, 1.6], [0.0, 0.0, 0.9]),
        "beetle_omni": ([2.4, 2.4, 1.6], [0.0, 0.0, 0.9]),
    }
    return camera[name]


def resolve_spawn_height(name: str) -> float:
    heights = {
        "mini_quad": 0.6,
        "beetle": 0.9,
        "beetle_omni": 0.9,
    }
    return heights[name]


def resolve_robot_cfg(name: str):
    robot_cfgs = {
        "mini_quad": MINI_QUADROTOR_CFG,
        "beetle": BEETLE_CFG,
        "beetle_omni": BEETLE_OMNI_CFG,
    }
    return robot_cfgs[name].replace(prim_path="/World/Robot")


def resolve_imu_link_name(name: str) -> str:
    return "fc"


def reset_robot(robot: Articulation, robot_name: str):
    root_state = robot.data.default_root_state.clone()
    root_state[:, 2] = resolve_spawn_height(robot_name)
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()


def format_tensor_row(tensor: torch.Tensor, decimals: int = 4) -> str:
    values = tensor.detach().cpu().tolist()
    return "[" + ", ".join(f"{value:.{decimals}f}" for value in values) + "]"


def build_imu_sensor_cfg(robot: Articulation, robot_name: str) -> tuple[ImuCfg, torch.Tensor, torch.Tensor, torch.Tensor]:
    body_id = robot.find_bodies("base_link")[0]
    imu_link_name = resolve_imu_link_name(robot_name)
    imu_id = robot.find_bodies(imu_link_name)[0]

    body_pos_w = robot.data.body_pos_w[:, body_id]
    body_quat_w = robot.data.body_quat_w[:, body_id]
    imu_pos_w = robot.data.body_pos_w[:, imu_id]
    imu_quat_w = robot.data.body_quat_w[:, imu_id]

    imu_pos_body, imu_quat_body = subtract_frame_transforms(
        body_pos_w,
        body_quat_w,
        imu_pos_w,
        imu_quat_w,
    )

    imu_sensor_cfg = ImuCfg(
        prim_path=f"/World/Robot/{imu_link_name}",
        update_period=0,
        history_length=1,
        offset=ImuCfg.OffsetCfg(
            pos=imu_pos_body[0, 0, :].cpu().numpy().tolist(),
            rot=imu_quat_body[0, 0, :].cpu().numpy().tolist(),
        ),
        debug_vis=True,
    )
    return imu_sensor_cfg, body_id, imu_id, imu_quat_body[0, 0, :]


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    camera_eye, camera_target = resolve_camera_view(args_cli.robot)
    sim.set_camera_view(camera_eye, camera_target)

    spawn_ground_plane("/World/ground", GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(resolve_robot_cfg(args_cli.robot))

    sim.reset()
    reset_robot(robot, args_cli.robot)
    robot.update(sim.get_physics_dt())

    imu_sensor_cfg, body_id, imu_id, imu_quat_body = build_imu_sensor_cfg(robot, args_cli.robot)
    imu_sensor = Imu(imu_sensor_cfg)

    sim.reset()
    reset_robot(robot, args_cli.robot)
    robot.update(sim.get_physics_dt())
    imu_sensor.update(sim.get_physics_dt())

    print(f"[INFO] Comparing root-state vs IMU outputs for robot: {args_cli.robot}")
    print("[INFO] Root signals come from the robot wrapper. IMU signals come from the Imu sensor object.")
    print("[INFO] When the IMU is offset from the root body, the values may differ slightly.")

    for step in range(1, args_cli.steps + 1):
        if not simulation_app.is_running():
            break

        with torch.inference_mode():
            sim.step()
            robot.update(sim.get_physics_dt())
            imu_sensor.update(sim.get_physics_dt())

            root_ang_vel_b = robot.data.root_ang_vel_b[0]
            imu_ang_vel_b = imu_sensor.data.ang_vel_b[0]
            ang_vel_diff = imu_ang_vel_b - root_ang_vel_b

            root_projected_gravity_b = robot.data.projected_gravity_b[0]
            imu_projected_gravity_b = imu_sensor.data.projected_gravity_b[0]
            projected_gravity_diff = imu_projected_gravity_b - root_projected_gravity_b

            # This reproduces the kind of debug comparison commented in aerial_lab envs:
            # transform the IMU link angular velocity into the body frame and compare it.
            imu_link_ang_vel_b = quat_apply(imu_quat_body.unsqueeze(0), robot.data.body_ang_vel_w[:, imu_id])[0]
            imu_link_ang_vel_diff = imu_sensor.data.ang_vel_b[0] - imu_link_ang_vel_b

        if step % args_cli.print_every == 0 or step == 1 or step == args_cli.steps:
            print(f"[STEP {step:04d}]")
            print(f"  root_ang_vel_b          = {format_tensor_row(root_ang_vel_b)}")
            print(f"  imu_link_ang_vel_b      = {format_tensor_row(imu_link_ang_vel_b)}")
            print(f"  imu_sensor.ang_vel_b    = {format_tensor_row(imu_ang_vel_b)}")
            print(f"  imu-root ang diff       = {format_tensor_row(ang_vel_diff)}")
            print(f"  imu-link ang diff       = {format_tensor_row(imu_link_ang_vel_diff)}")
            print(f"  root_projected_gravity  = {format_tensor_row(root_projected_gravity_b)}")
            print(f"  imu.projected_gravity_b = {format_tensor_row(imu_projected_gravity_b)}")
            print(f"  gravity diff            = {format_tensor_row(projected_gravity_diff)}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
