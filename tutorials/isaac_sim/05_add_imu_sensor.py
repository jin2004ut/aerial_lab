# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Attach a real IMU sensor object and read its outputs.

This tutorial is the next step after `04_read_robot_state.py`.
There we read state tensors directly from the robot wrapper.
Here we instantiate an Isaac Lab `Imu` sensor and read from `imu_sensor.data.*`.

This mirrors the pattern used in Aerial Lab task code such as:

- `tasks/direct/quadcopter/mini_quadcopter_env.py`
- `tasks/direct/beetle/beetle_env.py`
- `tasks/direct/beetle/beetle_omni_env.py`
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Add and read an IMU sensor in Isaac Sim.")
parser.add_argument(
    "--robot",
    type=str,
    default="mini_quad",
    choices=["mini_quad", "beetle", "beetle_omni"],
    help="Robot asset to spawn. These robots already use an IMU pattern in aerial_lab.",
)
parser.add_argument("--steps", type=int, default=240, help="Number of simulation steps to run.")
parser.add_argument("--print_every", type=int, default=60, help="How often to print sensor values.")
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
from isaaclab.utils.math import subtract_frame_transforms


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
    # In aerial_lab's current direct aerial tasks, the IMU is attached to `fc`.
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


def build_imu_sensor_cfg(robot: Articulation, robot_name: str) -> ImuCfg:
    # This follows the same idea as aerial_lab's envs:
    # 1. find the root body and the imu link body
    # 2. compute the IMU offset in the body frame
    # 3. pass that offset into ImuCfg
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
    return imu_sensor_cfg


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    camera_eye, camera_target = resolve_camera_view(args_cli.robot)
    sim.set_camera_view(camera_eye, camera_target)

    spawn_ground_plane("/World/ground", GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(resolve_robot_cfg(args_cli.robot))

    # First reset: create the robot in physics and let us inspect body poses.
    sim.reset()
    reset_robot(robot, args_cli.robot)
    robot.update(sim.get_physics_dt())

    # Create a real sensor object, similar to:
    #   self._imu_sensor = Imu(self.cfg.imu_sensor)
    # inside aerial_lab task environments.
    imu_sensor_cfg = build_imu_sensor_cfg(robot, args_cli.robot)
    imu_sensor = Imu(imu_sensor_cfg)

    # Second reset: after the sensor object exists, refresh the world again.
    sim.reset()
    reset_robot(robot, args_cli.robot)
    robot.update(sim.get_physics_dt())
    imu_sensor.update(sim.get_physics_dt())

    print(f"[INFO] Added IMU sensor for robot: {args_cli.robot}")
    print("[INFO] Compare `imu_sensor.data.*` below with the corresponding aerial_lab env code.")
    print("[INFO] The most useful outputs to inspect first are ang_vel_b and projected_gravity_b.")

    for step in range(1, args_cli.steps + 1):
        if not simulation_app.is_running():
            break

        with torch.inference_mode():
            sim.step()
            robot.update(sim.get_physics_dt())
            imu_sensor.update(sim.get_physics_dt())

            root_ang_vel_b = robot.data.root_ang_vel_b[0]
            root_projected_gravity_b = robot.data.projected_gravity_b[0]
            imu_ang_vel_b = imu_sensor.data.ang_vel_b[0]
            imu_projected_gravity_b = imu_sensor.data.projected_gravity_b[0]

        if step % args_cli.print_every == 0 or step == 1 or step == args_cli.steps:
            print(f"[STEP {step:04d}]")
            print(f"  root_ang_vel_b          = {format_tensor_row(root_ang_vel_b)}")
            print(f"  imu_sensor.ang_vel_b    = {format_tensor_row(imu_ang_vel_b)}")
            print(f"  root_projected_gravity  = {format_tensor_row(root_projected_gravity_b)}")
            print(f"  imu.projected_gravity_b = {format_tensor_row(imu_projected_gravity_b)}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
