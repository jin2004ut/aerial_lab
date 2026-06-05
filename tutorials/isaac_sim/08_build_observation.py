# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build an RL-style observation tensor by hand.

This mirrors the observation construction pattern in
`tasks/direct/quadcopter/mini_quadcopter_env.py`.
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Build an RL observation manually from robot state.")
parser.add_argument("--steps", type=int, default=240, help="Number of simulation steps.")
parser.add_argument("--print_every", type=int, default=60, help="How often to print observation details.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import torch
from aerial_lab.assets.aerialrobot import MINI_QUADROTOR_CFG
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_from_euler_xyz, subtract_frame_transforms


def reset_robot(robot: Articulation):
    root_state = robot.data.default_root_state.clone()
    root_state[:, 2] = 0.6
    root_state[:, 3:7] = quat_from_euler_xyz(
        torch.zeros_like(root_state[:, 0]),
        torch.zeros_like(root_state[:, 0]),
        torch.ones_like(root_state[:, 0]) * (math.pi * 0.5),
    )
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset()


def fmt(x: torch.Tensor) -> str:
    return "[" + ", ".join(f"{v:.4f}" for v in x.detach().cpu().tolist()) + "]"


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 2.0, 1.4], [0.0, 0.0, 0.6])
    spawn_ground_plane("/World/ground", GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(MINI_QUADROTOR_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    reset_robot(robot)
    robot.update(sim.get_physics_dt())

    desired_pos_w = torch.tensor([[0.3, -0.2, 0.8]], device=robot.device)
    desired_quat_w = quat_from_euler_xyz(
        torch.tensor([0.0], device=robot.device),
        torch.tensor([0.0], device=robot.device),
        torch.tensor([math.pi * 0.5], device=robot.device),
    )
    last_actions = torch.zeros((1, 4), device=robot.device)

    print("[INFO] Building observation pieces similar to mini_quadcopter_env._get_observations().")

    for step in range(1, args_cli.steps + 1):
        if not simulation_app.is_running():
            break

        with torch.inference_mode():
            sim.step()
            robot.update(sim.get_physics_dt())

            root_pos_w = robot.data.root_pos_w
            root_quat_w = robot.data.root_quat_w
            root_lin_vel_w = robot.data.root_lin_vel_w
            root_ang_vel_b = robot.data.root_ang_vel_b
            projected_gravity_w = robot.data.GRAVITY_VEC_W

            root_lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
            projected_gravity_b = quat_apply_inverse(root_quat_w, projected_gravity_w)
            goal_pos_b, goal_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, desired_pos_w, desired_quat_w)

            root_rot_mat = matrix_from_quat(root_quat_w)
            root_rot_vec = root_rot_mat[:, :2, :].reshape(1, 6)
            goal_rot_mat = matrix_from_quat(goal_quat_b)
            goal_rot_vec = goal_rot_mat[:, :2, :].reshape(1, 6)

            obs = torch.cat(
                (
                    root_lin_vel_b,
                    root_ang_vel_b * 0.2,
                    projected_gravity_b,
                    goal_pos_b,
                    root_rot_vec,
                    goal_rot_vec,
                    last_actions,
                ),
                dim=-1,
            )

        if step % args_cli.print_every == 0 or step == 1 or step == args_cli.steps:
            print(f"[STEP {step:04d}] obs_dim={obs.shape[-1]}")
            print(f"  root_lin_vel_b     = {fmt(root_lin_vel_b[0])}")
            print(f"  root_ang_vel_b     = {fmt(root_ang_vel_b[0])}")
            print(f"  projected_gravity  = {fmt(projected_gravity_b[0])}")
            print(f"  goal_pos_b         = {fmt(goal_pos_b[0])}")
            print(f"  root_rot_vec[:6]   = {fmt(root_rot_vec[0])}")
            print(f"  goal_rot_vec[:6]   = {fmt(goal_rot_vec[0])}")
            print(f"  last_actions       = {fmt(last_actions[0])}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
