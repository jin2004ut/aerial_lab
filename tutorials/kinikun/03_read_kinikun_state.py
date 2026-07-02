# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read kinikun root state and joint state tensors."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Read kinikun state tensors.")
parser.add_argument("--steps", type=int, default=240, help="Number of simulation steps.")
parser.add_argument("--print_every", type=int, default=60, help="How often to print state.")
parser.add_argument("--enable_gravity", action="store_true", help="Enable gravity instead of using the asset default.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

try:
    from .common import fmt_tensor_row, resolve_camera_view, resolve_kinikun_cfg, reset_robot, spawn_ground_and_light
except ImportError:
    from common import fmt_tensor_row, resolve_camera_view, resolve_kinikun_cfg, reset_robot, spawn_ground_and_light


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

    arm_ids, _ = robot.find_joints("arm.*_joint")
    rotor_ids, _ = robot.find_joints("rotor.*")

    for step in range(1, args_cli.steps + 1):
        if not simulation_app.is_running():
            break

        sim.step()
        robot.update(sim.get_physics_dt())

        if step % args_cli.print_every == 0 or step == 1 or step == args_cli.steps:
            print(f"[STEP {step:04d}]")
            print(f"  root_pos_w       = {fmt_tensor_row(robot.data.root_pos_w[0])}")
            print(f"  root_quat_w      = {fmt_tensor_row(robot.data.root_quat_w[0])}")
            print(f"  root_lin_vel_b   = {fmt_tensor_row(robot.data.root_lin_vel_b[0])}")
            print(f"  root_ang_vel_b   = {fmt_tensor_row(robot.data.root_ang_vel_b[0])}")
            print(f"  arm_joint_pos    = {fmt_tensor_row(robot.data.joint_pos[0, arm_ids])}")
            print(f"  rotor_joint_vel  = {fmt_tensor_row(robot.data.joint_vel[0, rotor_ids])}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
