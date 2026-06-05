# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare a few common control entry points used in Isaac Sim / Isaac Lab.

This tutorial does not try to be a full controller.
It is meant to answer: "where does my action go?"

Modes:
- `position`: send joint position targets to `gimbal.*`
- `velocity`: send joint velocity targets to `rotor.*`
- `force`: apply external thrust-like forces at rotor parent bodies

Related Aerial Lab code:
- joint position targets:
  `tasks/direct/beetle/beetle_env.py`
- external force / torque application:
  `tasks/direct/quadcopter/mini_quadcopter_env.py`
  `tasks/direct/beetle/beetle_omni_env.py`
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare control modes on an aerial_lab robot.")
parser.add_argument(
    "--robot",
    type=str,
    default="beetle",
    choices=["beetle", "beetle_omni", "mini_quad"],
    help="Robot asset to spawn.",
)
parser.add_argument(
    "--mode",
    type=str,
    default="position",
    choices=["position", "velocity", "force"],
    help="Control entry point to exercise.",
)
parser.add_argument("--steps", type=int, default=600, help="Number of simulation steps.")
parser.add_argument("--print_every", type=int, default=120, help="How often to print step logs.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import torch
from aerial_lab.assets.aerialrobot import BEETLE_CFG, BEETLE_OMNI_CFG, MINI_QUADROTOR_CFG
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane


def resolve_camera_view(name: str) -> tuple[list[float], list[float]]:
    camera = {
        "mini_quad": ([2.0, 2.0, 1.4], [0.0, 0.0, 0.6]),
        "beetle": ([2.4, 2.4, 1.6], [0.0, 0.0, 0.9]),
        "beetle_omni": ([2.4, 2.4, 1.6], [0.0, 0.0, 0.9]),
    }
    return camera[name]


def resolve_spawn_height(name: str) -> float:
    heights = {"mini_quad": 0.6, "beetle": 0.9, "beetle_omni": 0.9}
    return heights[name]


def resolve_robot_cfg(name: str):
    robot_cfgs = {"mini_quad": MINI_QUADROTOR_CFG, "beetle": BEETLE_CFG, "beetle_omni": BEETLE_OMNI_CFG}
    return robot_cfgs[name].replace(prim_path="/World/Robot")


def resolve_thrust_body_expr(name: str) -> str:
    # Avoid broad alternation regex here. In practice these names differ by robot.
    exprs = {
        "mini_quad": "thrust.*",
        "beetle": "rotor_parent.*",
        "beetle_omni": "rotor_parent.*",
    }
    return exprs[name]


def empty_ids_names() -> tuple[list[int], list[str]]:
    return [], []


def reset_robot(robot: Articulation, robot_name: str):
    root_state = robot.data.default_root_state.clone()
    root_state[:, 2] = resolve_spawn_height(robot_name)
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()


def format_named_values(names: list[str], values: torch.Tensor, decimals: int = 4) -> str:
    pairs = []
    for name, value in zip(names, values.detach().cpu().tolist(), strict=False):
        pairs.append(f"{name}={value:.{decimals}f}")
    return ", ".join(pairs)


def format_named_vectors(names: list[str], values: torch.Tensor, decimals: int = 4) -> str:
    pairs = []
    for name, vec in zip(names, values.detach().cpu().tolist(), strict=False):
        vec_str = "[" + ", ".join(f"{value:.{decimals}f}" for value in vec) + "]"
        pairs.append(f"{name}={vec_str}")
    return ", ".join(pairs)


def main():
    print("[SETUP] creating SimulationCfg", flush=True)
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01)
    print("[SETUP] creating SimulationContext", flush=True)
    sim = sim_utils.SimulationContext(sim_cfg)
    camera_eye, camera_target = resolve_camera_view(args_cli.robot)
    print("[SETUP] setting camera", flush=True)
    sim.set_camera_view(camera_eye, camera_target)

    print("[SETUP] spawning ground", flush=True)
    spawn_ground_plane("/World/ground", GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    print("[SETUP] creating light", flush=True)
    light_cfg.func("/World/Light", light_cfg)

    print(f"[SETUP] creating robot articulation for {args_cli.robot}", flush=True)
    robot = Articulation(resolve_robot_cfg(args_cli.robot))
    print("[SETUP] sim.reset()", flush=True)
    sim.reset()
    print("[SETUP] reset_robot()", flush=True)
    reset_robot(robot, args_cli.robot)
    print("[SETUP] robot.update()", flush=True)
    robot.update(sim.get_physics_dt())

    gimbal_ids, gimbal_names = empty_ids_names()
    rotor_ids, rotor_names = empty_ids_names()
    thrust_body_ids, thrust_body_names = empty_ids_names()

    if args_cli.mode == "position":
        print("[SETUP] finding gimbal joints", flush=True)
        gimbal_ids, gimbal_names = robot.find_joints("gimbal.*")
    if args_cli.mode == "velocity":
        print("[SETUP] finding rotor joints", flush=True)
        rotor_ids, rotor_names = robot.find_joints("rotor.*")
    if args_cli.mode == "force":
        thrust_body_expr = resolve_thrust_body_expr(args_cli.robot)
        print(f"[SETUP] finding thrust bodies with expr={thrust_body_expr}", flush=True)
        thrust_body_ids, thrust_body_names = robot.find_bodies(thrust_body_expr)

    print(f"[INFO] mode={args_cli.mode} robot={args_cli.robot}", flush=True)
    print(f"[INFO] gimbals={gimbal_names}", flush=True)
    print(f"[INFO] rotors={rotor_names}", flush=True)
    print(f"[INFO] thrust bodies={thrust_body_names}", flush=True)
    if args_cli.robot == "mini_quad" and args_cli.mode == "velocity":
        print(
            "[INFO] mini_quad + velocity only sends rotor velocity targets. It does not generate thrust in this tutorial.",
            flush=True,
        )

    for step in range(1, args_cli.steps + 1):
        if not simulation_app.is_running():
            break

        with torch.inference_mode():
            phase = 0.02 * step
            command_log = None

            if args_cli.mode == "position" and len(gimbal_ids) > 0:
                target = robot.data.default_joint_pos.clone()
                target[:, gimbal_ids] += 0.25 * torch.sin(
                    torch.ones((robot.data.default_joint_pos.shape[0], len(gimbal_ids)), device=robot.device) * phase
                )
                robot.set_joint_position_target(target[:, gimbal_ids], gimbal_ids)
                command_log = ("gimbal position targets", gimbal_names, target[0, gimbal_ids])

            elif args_cli.mode == "velocity" and len(rotor_ids) > 0:
                vel_target = 10.0 + 5.0 * math.sin(phase)
                vel_tensor = torch.ones((robot.data.default_joint_pos.shape[0], len(rotor_ids)), device=robot.device)
                vel_tensor *= vel_target
                robot.set_joint_velocity_target(vel_tensor, rotor_ids)
                command_log = ("rotor velocity targets", rotor_names, vel_tensor[0])

            elif args_cli.mode == "force" and len(thrust_body_ids) > 0:
                num_envs = robot.data.default_root_state.shape[0]
                num_bodies = len(thrust_body_ids)
                forces = torch.zeros((num_envs, num_bodies, 3), device=robot.device)
                torques = torch.zeros((num_envs, num_bodies, 3), device=robot.device)
                forces[:, :, 2] = 10.5 + 0.75 * math.sin(phase)
                robot.set_external_force_and_torque(forces=forces, torques=torques, body_ids=thrust_body_ids)
                command_log = ("external forces", thrust_body_names, forces[0])

            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())

        if step % args_cli.print_every == 0 or step == 1 or step == args_cli.steps:
            print(f"[STEP {step:04d}] root_pos={robot.data.root_pos_w[0].detach().cpu().tolist()}", flush=True)
            if command_log is not None:
                label, names, values = command_log
                if values.ndim == 1:
                    print(f"  {label}: {format_named_values(names, values)}", flush=True)
                else:
                    print(f"  {label}: {format_named_vectors(names, values)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
