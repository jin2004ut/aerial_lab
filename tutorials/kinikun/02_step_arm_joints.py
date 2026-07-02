# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drive the four kinikun arm joints with simple position targets."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step kinikun and move the arm joints.")
parser.add_argument("--steps", type=int, default=600, help="Number of simulation steps.")
parser.add_argument("--enable_gravity", action="store_true", help="Enable gravity instead of using the asset default.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import torch
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

    arm_ids, arm_names = robot.find_joints("arm.*_joint")
    print(f"[INFO] arm_joints={arm_names}")
    print(f"[INFO] gravity_enabled={args_cli.enable_gravity}")

    for step in range(1, args_cli.steps + 1):
        if not simulation_app.is_running():
            break

        phase = step * sim.get_physics_dt()
        joint_target = robot.data.default_joint_pos.clone()
        offsets = torch.tensor(
            [
                0.25 * math.sin(phase),
                0.20 * math.sin(phase + 0.7),
                0.30 * math.sin(phase + 1.4),
                0.20 * math.sin(phase + 2.1),
            ],
            device=robot.device,
        ).unsqueeze(0)
        joint_target[:, arm_ids] = offsets
        robot.set_joint_position_target(joint_target[:, arm_ids], arm_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

        if step % 60 == 0 or step == 1 or step == args_cli.steps:
            print(f"[STEP {step:04d}]")
            print(f"  arm_target[0] = {fmt_tensor_row(joint_target[0, arm_ids])}")
            print(f"  arm_pos[0]    = {fmt_tensor_row(robot.data.joint_pos[0, arm_ids])}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
