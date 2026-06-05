# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn one Aerial Lab robot in Isaac Sim without wrapping it as an RL environment.

Read this file top to bottom. The script is intentionally small so you can map each
part to one idea:

1. Parse CLI arguments.
2. Launch Isaac Sim.
3. Build a simple scene.
4. Turn one robot config into a live articulation.
5. Step the simulator loop.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn an Aerial Lab robot into Isaac Sim.")
parser.add_argument(
    "--robot",
    type=str,
    default="mini_quad",
    choices=["mini_quad", "beetle", "beetle_omni", "dragon", "spidar"],
    help="Robot asset to spawn.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
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
    # Each entry here is a reusable asset definition from aerial_lab/assets/aerialrobot.py.
    # We only replace the prim path so the robot appears at /World/Robot in this tutorial.
    robot_cfgs = {
        "mini_quad": MINI_QUADROTOR_CFG,
        "beetle": BEETLE_CFG,
        "beetle_omni": BEETLE_OMNI_CFG,
        "dragon": DRAGON_CFG,
        "spidar": SPIDAR_CFG,
    }
    return robot_cfgs[name].replace(prim_path="/World/Robot")


def place_robot_at_tutorial_pose(robot: Articulation, robot_name: str):
    # Override the asset default spawn height so the robot starts closer to the ground
    # and fills more of the camera view in these tutorials.
    root_state = robot.data.default_root_state.clone()
    root_state[:, 2] = resolve_spawn_height(robot_name)

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset()


def main():
    # SimulationCfg defines low-level physics settings such as device and timestep.
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01)
    # SimulationContext is the object we actually step every frame.
    sim = sim_utils.SimulationContext(sim_cfg)
    # Camera is only for viewer convenience. It does not affect physics.
    camera_eye, camera_target = resolve_camera_view(args_cli.robot)
    sim.set_camera_view(camera_eye, camera_target)

    # Ground plane gives the robot a visible reference and a collision surface.
    spawn_ground_plane("/World/ground", GroundPlaneCfg())
    # Dome light makes the robot visible in the renderer.
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # Articulation is Isaac Lab's runtime wrapper around a multibody robot.
    # The config contains URDF path, joint defaults, actuators, and physics properties.
    robot = Articulation(resolve_robot_cfg(args_cli.robot))

    # sim.reset() finalizes scene creation and pushes authored prims into the physics world.
    sim.reset()
    place_robot_at_tutorial_pose(robot, args_cli.robot)
    # update() pulls the latest simulation buffers back into robot.data.* tensors.
    robot.update(sim.get_physics_dt())

    print(f"[INFO] Spawned robot: {args_cli.robot}")
    print(f"[INFO] Joint count: {robot.num_joints}")
    print(f"[INFO] Body count: {robot.num_bodies}")
    print("[INFO] Close the simulator window when you are done inspecting the model.")

    # This is the simplest possible simulation loop:
    # keep stepping physics until the app window closes.
    while simulation_app.is_running():
        sim.step()
        robot.update(sim.get_physics_dt())


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
