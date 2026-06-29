# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Step physics manually and send simple joint targets to an Aerial Lab robot.

This tutorial is the bridge between "I can spawn a robot" and
"I understand the control loop behind an RL environment."

The important flow is:

1. Create scene and robot.
2. Reset robot to a known state.
3. Build a target command tensor.
4. Write commands into the simulator.
5. Step physics.
6. Read new state back from the simulator.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step an Aerial Lab robot inside Isaac Sim.")
parser.add_argument(
    "--robot",
    type=str,
    default="mini_quad",
    choices=["mini_quad", "beetle", "beetle_omni", "dragon", "spidar"],
    help="Robot asset to spawn.",
)
parser.add_argument("--steps", type=int, default=600, help="Number of simulation steps to run.")
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
    # Pick one reusable robot asset config and bind it to /World/Robot for this demo.
    robot_cfgs = {
        "mini_quad": MINI_QUADROTOR_CFG,
        "beetle": BEETLE_CFG,
        "beetle_omni": BEETLE_OMNI_CFG,
        "dragon": DRAGON_CFG,
        "spidar": SPIDAR_CFG,
    }
    return robot_cfgs[name].replace(prim_path="/World/Robot")


def reset_robot(robot: Articulation, robot_name: str):
    # default_root_state stores the spawn pose and root velocities authored in the asset config.
    root_state = robot.data.default_root_state.clone()
    root_state[:, 2] = resolve_spawn_height(robot_name)
    # default_joint_pos / vel are the robot's nominal joint values at reset.
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    # The root state is split into pose and velocity before being written to sim.
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    # This writes all joint positions and velocities in one shot.
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    # reset() clears internal buffers used by the articulation wrapper.
    robot.reset()


def main():
    # Physics runs at dt=0.01 s, so 100 simulation steps correspond to 1 second.
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    camera_eye, camera_target = resolve_camera_view(args_cli.robot)
    sim.set_camera_view(camera_eye, camera_target)

    # Scene setup: a floor plus one light, same as in the previous tutorial.
    spawn_ground_plane("/World/ground", GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # Instantiate the robot from its config.
    robot = Articulation(resolve_robot_cfg(args_cli.robot))

    # Finalize authored scene objects into the physics engine.
    sim.reset()
    # Put the robot into a known initial condition before stepping.
    reset_robot(robot, args_cli.robot)
    # Pull fresh state tensors into robot.data.* so reads below are current.
    robot.update(sim.get_physics_dt())

    print(f"[INFO] Running {args_cli.steps} steps for robot: {args_cli.robot}")
    print("[INFO] This script shows the low-level loop before Gym/RL wrappers are added.")

    for step in range(1, args_cli.steps + 1):
        if not simulation_app.is_running():
            break

        with torch.inference_mode():
            # Start from the nominal joint pose every frame.
            joint_target = robot.data.default_joint_pos.clone()
            if joint_target.numel() > 0:
                # Create a tiny sinusoidal motion so the robot has a visible command.
                amplitude = 1.0
                phase = 0.01 * step
                # ones_like(joint_target) matches the number of joints automatically.
                joint_target += amplitude * torch.sin(torch.ones_like(joint_target) * phase)
                # This only sets the desired target in the wrapper. Physics has not advanced yet.
                robot.set_joint_position_target(joint_target)

            # Push pending actuator commands into the simulator buffers.
            robot.write_data_to_sim()
            # Advance the simulator by one timestep.
            sim.step()
            # Refresh robot.data.* tensors from the new simulation state.
            robot.update(sim.get_physics_dt())

        if step % 120 == 0 or step == args_cli.steps:
            print(f"[STEP {step:04d}] sim_time={step * sim.get_physics_dt():.2f}s")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
