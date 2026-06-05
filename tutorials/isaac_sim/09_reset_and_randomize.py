# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Practice reset logic and simple randomization.

This is a tutorial-scale version of what `_reset_idx(...)` does in aerial_lab envs.
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Reset and randomize an aerial robot state.")
parser.add_argument("--episodes", type=int, default=5, help="Number of reset demonstrations.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import torch
from aerial_lab.assets.aerialrobot import MINI_QUADROTOR_CFG
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_from_euler_xyz


def fmt(x: torch.Tensor) -> str:
    return "[" + ", ".join(f"{v:.4f}" for v in x.detach().cpu().tolist()) + "]"


def randomized_reset(robot: Articulation):
    root_state = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    root_state[:, 0:2] = torch.empty_like(root_state[:, 0:2]).uniform_(-0.5, 0.5)
    root_state[:, 2] = torch.empty_like(root_state[:, 2]).uniform_(0.5, 1.2)

    roll = torch.empty((1,), device=robot.device).uniform_(-0.2, 0.2)
    pitch = torch.empty((1,), device=robot.device).uniform_(-0.2, 0.2)
    yaw = torch.ones((1,), device=robot.device) * (math.pi * 0.5)
    root_state[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)

    root_state[:, 7:10] = torch.empty_like(root_state[:, 7:10]).uniform_(-0.5, 0.5)
    root_state[:, 10:13] = torch.empty_like(root_state[:, 10:13]).uniform_(-0.8, 0.8)

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    return root_state


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 2.0, 1.4], [0.0, 0.0, 0.6])
    spawn_ground_plane("/World/ground", GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(MINI_QUADROTOR_CFG.replace(prim_path="/World/Robot"))
    sim.reset()

    print("[INFO] Demonstrating randomized resets similar to aerial_lab env reset logic.")

    for episode in range(1, args_cli.episodes + 1):
        if not simulation_app.is_running():
            break

        with torch.inference_mode():
            root_state = randomized_reset(robot)
            robot.update(sim.get_physics_dt())

        print(f"[EPISODE {episode:02d}]")
        print(f"  pos      = {fmt(root_state[0, 0:3])}")
        print(f"  quat     = {fmt(root_state[0, 3:7])}")
        print(f"  lin_vel  = {fmt(root_state[0, 7:10])}")
        print(f"  ang_vel  = {fmt(root_state[0, 10:13])}")

        for _ in range(40):
            if not simulation_app.is_running():
                break
            sim.step()
            robot.update(sim.get_physics_dt())


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
