# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Attach an IMU sensor to kinikun's fc link and read it."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Add and read an IMU sensor on kinikun.")
parser.add_argument("--steps", type=int, default=240, help="Number of simulation steps.")
parser.add_argument("--print_every", type=int, default=60, help="How often to print sensor data.")
parser.add_argument("--enable_gravity", action="store_true", help="Enable gravity instead of using the asset default.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Imu

try:
    from .common import (
        build_imu_sensor_cfg,
        fmt_tensor_row,
        resolve_camera_view,
        resolve_kinikun_cfg,
        reset_robot,
        spawn_ground_and_light,
    )
except ImportError:
    from common import (
        build_imu_sensor_cfg,
        fmt_tensor_row,
        resolve_camera_view,
        resolve_kinikun_cfg,
        reset_robot,
        spawn_ground_and_light,
    )


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    camera_eye, camera_target = resolve_camera_view()
    sim.set_camera_view(camera_eye, camera_target)
    spawn_ground_and_light()

    robot = Articulation(resolve_kinikun_cfg(enable_gravity=args_cli.enable_gravity))
    sim.reset()
    reset_robot(robot)
    robot.update(sim.get_physics_dt())

    imu_sensor = Imu(build_imu_sensor_cfg(robot))
    sim.reset()
    reset_robot(robot)
    robot.update(sim.get_physics_dt())
    imu_sensor.update(sim.get_physics_dt())

    for step in range(1, args_cli.steps + 1):
        if not simulation_app.is_running():
            break

        sim.step()
        robot.update(sim.get_physics_dt())
        imu_sensor.update(sim.get_physics_dt())

        if step % args_cli.print_every == 0 or step == 1 or step == args_cli.steps:
            print(f"[STEP {step:04d}]")
            print(f"  imu_pos_w              = {fmt_tensor_row(imu_sensor.data.pos_w[0])}")
            print(f"  imu_quat_w             = {fmt_tensor_row(imu_sensor.data.quat_w[0])}")
            print(f"  imu_lin_acc_b         = {fmt_tensor_row(imu_sensor.data.lin_acc_b[0])}")
            print(f"  imu_ang_vel_b         = {fmt_tensor_row(imu_sensor.data.ang_vel_b[0])}")
            print(f"  projected_gravity_b   = {fmt_tensor_row(imu_sensor.data.projected_gravity_b[0])}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
