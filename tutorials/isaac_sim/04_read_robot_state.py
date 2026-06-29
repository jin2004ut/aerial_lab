# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read root state, joint state, and IMU-like quantities from an Aerial Lab robot.

This tutorial focuses on the read side of the simulation loop.

The flow is:

1. Create scene and robot.
2. Reset to a known initial state.
3. Step the simulator.
4. Read tensors from `robot.data.*`.
5. Interpret those tensors as pose, velocity, and sensor-like signals.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Read robot state tensors from Isaac Sim.")
parser.add_argument(
    "--robot",
    type=str,
    default="mini_quad",
    choices=["mini_quad", "beetle", "beetle_omni", "dragon", "spidar"],
    help="Robot asset to spawn.",
)
parser.add_argument("--steps", type=int, default=240, help="Number of simulation steps to run.")
parser.add_argument("--print_every", type=int, default=60, help="How often to print the current state.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import torch
from aerial_lab.assets.aerialrobot import BEETLE_CFG, BEETLE_OMNI_CFG, DRAGON_CFG, MINI_QUADROTOR_CFG, SPIDAR_CFG
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane


def resolve_camera_view(name: str) -> tuple[list[float], list[float]]:
    camera = {
        "mini_quad": ([2.0, 2.0, 1.4], [0.0, 0.0, 0.6]),
        "beetle": ([2.4, 2.4, 1.6], [0.0, 0.0, 0.9]),
        "beetle_omni": ([2.4, 2.4, 1.6], [0.0, 0.0, 0.9]),
        "dragon": ([2.8, 2.8, 1.8], [0.0, 0.0, 1.0]),
        "spidar": ([2.8, 2.8, 1.8], [0.0, 0.0, 1.0]),
    }
    return camera[name]


def resolve_spawn_height(name: str) -> float:
    heights = {
        "mini_quad": 0.6,
        "beetle": 0.9,
        "beetle_omni": 0.9,
        "dragon": 1.0,
        "spidar": 1.0,
    }
    return heights[name]


def resolve_robot_cfg(name: str):
    # Reuse the same robot asset configs as the previous tutorials.
    # As before, we bind the chosen asset to /World/Robot for a single-robot scene.
    robot_cfgs = {
        "mini_quad": MINI_QUADROTOR_CFG,
        "beetle": BEETLE_CFG,
        "beetle_omni": BEETLE_OMNI_CFG,
        "dragon": DRAGON_CFG,
        "spidar": SPIDAR_CFG,
    }
    return robot_cfgs[name].replace(prim_path="/World/Robot")


def reset_robot(robot: Articulation, robot_name: str):
    # Clone the authored default state so we can write it into the simulator cleanly.
    root_state = robot.data.default_root_state.clone()
    root_state[:, 2] = resolve_spawn_height(robot_name)
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    # Root pose is position + quaternion.
    robot.write_root_pose_to_sim(root_state[:, :7])
    # Root velocity is linear velocity + angular velocity.
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    # Joint state means all joint positions and velocities.
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    # reset() clears internal cached state inside the articulation wrapper.
    robot.reset()


def format_tensor_row(tensor: torch.Tensor, decimals: int = 4) -> str:
    # Helper to print one tensor row in a compact human-readable format.
    values = tensor.detach().cpu().tolist()
    return "[" + ", ".join(f"{value:.{decimals}f}" for value in values) + "]"


def main():
    # Same physics setup as the previous tutorial: 0.01 s timestep on the chosen device.
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    camera_eye, camera_target = resolve_camera_view(args_cli.robot)
    sim.set_camera_view(camera_eye, camera_target)

    # Minimal scene setup so the robot has a floor and visible lighting.
    spawn_ground_plane("/World/ground", GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # Instantiate one runtime robot from the selected asset config.
    robot = Articulation(resolve_robot_cfg(args_cli.robot))

    # Finalize the scene, then place the robot in its default initial condition.
    sim.reset()
    reset_robot(robot, args_cli.robot)
    # Pull the current simulation buffers into robot.data.*.
    robot.update(sim.get_physics_dt())

    # We keep the previous world-frame linear velocity so we can estimate acceleration
    # using a simple finite difference: a ~= (v_t - v_{t-1}) / dt.
    prev_root_lin_vel_w = robot.data.root_lin_vel_w.clone()

    print(f"[INFO] Reading state for robot: {args_cli.robot}")
    print("[INFO] We will print root pose, joint state, and IMU-like quantities.")
    print("[INFO] IMU-like quantities here mean body angular velocity and finite-difference linear acceleration.")

    for step in range(1, args_cli.steps + 1):
        if not simulation_app.is_running():
            break

        with torch.inference_mode():
            # Advance the physics world by one step.
            sim.step()
            # Refresh all tensors exposed through robot.data.*.
            robot.update(sim.get_physics_dt())

            # Root state in world frame:
            #   - position [x, y, z]
            #   - quaternion [w/x/y/z or library-native order depending on Isaac Lab internals]
            root_pos_w = robot.data.root_pos_w[0]
            root_quat_w = robot.data.root_quat_w[0]

            # Linear velocity in the world frame and in the body frame.
            # body frame is often more intuitive for control because it is attached to the robot.
            root_lin_vel_w = robot.data.root_lin_vel_w[0]
            root_lin_vel_b = robot.data.root_lin_vel_b[0]

            # Angular velocity of the root body in the world frame.
            root_ang_vel_w = robot.data.root_ang_vel_w[0]

            # Joint positions and velocities for the first robot instance in the batch.
            joint_pos = robot.data.joint_pos[0]
            joint_vel = robot.data.joint_vel[0]

            # This is not a built-in IMU sensor readout.
            # It is an IMU-like approximation of linear acceleration from velocity difference.
            root_lin_acc_w = (robot.data.root_lin_vel_w - prev_root_lin_vel_w) / sim.get_physics_dt()
            root_lin_acc_w = root_lin_acc_w[0]
            prev_root_lin_vel_w = robot.data.root_lin_vel_w.clone()

        if step % args_cli.print_every == 0 or step == 1 or step == args_cli.steps:
            # Print only every few steps so the console stays readable.
            print(f"[STEP {step:04d}]")
            print(f"  root_pos_w      = {format_tensor_row(root_pos_w)}")
            print(f"  root_quat_w     = {format_tensor_row(root_quat_w)}")
            print(f"  root_lin_vel_w  = {format_tensor_row(root_lin_vel_w)}")
            print(f"  root_lin_vel_b  = {format_tensor_row(root_lin_vel_b)}")
            print(f"  root_ang_vel_w  = {format_tensor_row(root_ang_vel_w)}")
            print(f"  lin_acc_w_like  = {format_tensor_row(root_lin_acc_w)}")
            print(f"  joint_pos[:6]   = {format_tensor_row(joint_pos[:6])}")
            print(f"  joint_vel[:6]   = {format_tensor_row(joint_vel[:6])}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
