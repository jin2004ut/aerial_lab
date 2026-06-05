# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Add a contact sensor and inspect contact forces.

This mirrors the contact sensor usage in aerial_lab direct envs such as
`beetle_env.py`, `mini_quadcopter_env.py`, and `spidar_env.py`.
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Attach a contact sensor and inspect collisions.")
parser.add_argument(
    "--robot",
    type=str,
    default="mini_quad",
    choices=["mini_quad", "beetle", "beetle_omni"],
    help="Robot asset to spawn.",
)
parser.add_argument("--steps", type=int, default=300, help="Number of simulation steps.")
parser.add_argument("--print_every", type=int, default=30, help="How often to print contact data.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import torch
from aerial_lab.assets.aerialrobot import BEETLE_CFG, BEETLE_OMNI_CFG, MINI_QUADROTOR_CFG
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane


def resolve_camera_view(name: str) -> tuple[list[float], list[float]]:
    camera = {
        "mini_quad": ([2.0, 2.0, 1.4], [0.0, 0.0, 0.6]),
        "beetle": ([2.4, 2.4, 1.6], [0.0, 0.0, 0.9]),
        "beetle_omni": ([2.4, 2.4, 1.6], [0.0, 0.0, 0.9]),
    }
    return camera[name]


def resolve_robot_cfg(name: str):
    robot_cfgs = {"mini_quad": MINI_QUADROTOR_CFG, "beetle": BEETLE_CFG, "beetle_omni": BEETLE_OMNI_CFG}
    return robot_cfgs[name].replace(prim_path="/World/Robot")


def reset_robot(robot: Articulation):
    root_state = robot.data.default_root_state.clone()
    root_state[:, 2] = 0.25
    root_state[:, 7:10] = 0.0
    root_state[:, 10:13] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset()


def fmt(x: torch.Tensor) -> str:
    return "[" + ", ".join(f"{v:.4f}" for v in x.detach().cpu().tolist()) + "]"


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    camera_eye, camera_target = resolve_camera_view(args_cli.robot)
    sim.set_camera_view(camera_eye, camera_target)

    spawn_ground_plane("/World/ground", GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(resolve_robot_cfg(args_cli.robot))
    contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/Robot/base_link",
        history_length=1,
        update_period=0,
        track_air_time=True,
        debug_vis=True,
    )
    contact_sensor = ContactSensor(contact_sensor_cfg)

    sim.reset()
    reset_robot(robot)
    robot.update(sim.get_physics_dt())
    contact_sensor.update(sim.get_physics_dt())

    print("[INFO] Watching contact forces on base_link. The robot starts low so it quickly touches the ground.")

    for step in range(1, args_cli.steps + 1):
        if not simulation_app.is_running():
            break

        with torch.inference_mode():
            # Give the robot a slight downward bias for a visible contact event.
            if step == 1:
                forces = torch.zeros((1, 1, 3), device=robot.device)
                torques = torch.zeros((1, 1, 3), device=robot.device)
                forces[:, :, 2] = -2.0
                body_id = robot.find_bodies("base_link")[0]
                robot.set_external_force_and_torque(forces=forces, torques=torques, body_ids=body_id)

            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            contact_sensor.update(sim.get_physics_dt())

            net_forces_w = contact_sensor.data.net_forces_w.squeeze(1)[0]
            force_norm = torch.linalg.norm(net_forces_w)

        if step % args_cli.print_every == 0 or step == 1 or step == args_cli.steps:
            print(f"[STEP {step:04d}]")
            print(f"  root_pos_w     = {fmt(robot.data.root_pos_w[0])}")
            print(f"  net_forces_w   = {fmt(net_forces_w)}")
            print(f"  force_norm     = {force_norm.item():.4f}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
