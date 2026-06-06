# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn the kinikun robot asset in Isaac Sim."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn kinikun in Isaac Sim.")
parser.add_argument("--enable_gravity", action="store_true", help="Enable gravity instead of using the asset default.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from tutorials.kinikun.common import resolve_camera_view, resolve_kinikun_cfg, reset_robot, spawn_ground_and_light


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

    print("[INFO] Spawned kinikun")
    print(f"[INFO] gravity_enabled={args_cli.enable_gravity}")
    print(f"[INFO] body_count={robot.num_bodies}")
    print(f"[INFO] joint_count={robot.num_joints}")
    print(f"[INFO] body_names={robot.body_names}")
    print(f"[INFO] joint_names={robot.joint_names}")

    while simulation_app.is_running():
        sim.step()
        robot.update(sim.get_physics_dt())


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
