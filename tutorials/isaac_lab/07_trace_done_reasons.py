# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trace which internal conditions contribute to terminated and truncated flags."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Trace done reasons in a direct Isaac Lab env.")
parser.add_argument("--task", type=str, required=True, help="Registered Gym task name.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=300, help="Number of environment steps.")
parser.add_argument("--print_every", type=int, default=30, help="How often to print status.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import aerial_lab.tasks  # noqa: F401
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab_tasks.utils import parse_env_cfg


def mask_count(mask: torch.Tensor) -> int:
    return int(mask.sum().item())


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        obs, info = env.reset()
        unwrapped = env.unwrapped
        print(f"[INFO] task={args_cli.task}")
        print(f"[INFO] env_type={unwrapped.__class__.__name__}")
        print("[INFO] tracing time_out, crash, drift, terminated, truncated")

        for step in range(1, args_cli.steps + 1):
            if not simulation_app.is_running():
                break

            with torch.inference_mode():
                actions = 2 * torch.rand(env.action_space.shape, device=unwrapped.device) - 1
                obs, reward, terminated, truncated, info = env.step(actions)

            contact_force = torch.linalg.norm(unwrapped._contact_sensor.data.net_forces_w.squeeze(1), dim=-1)
            crash = contact_force > unwrapped.cfg.contact_force_threshold
            drift = torch.logical_or(unwrapped._robot.data.root_pos_w[:, 2] < 0.1, unwrapped._robot.data.root_pos_w[:, 2] > 5.0)
            time_out = unwrapped.episode_length_buf >= unwrapped.max_episode_length - 1

            if step % args_cli.print_every == 0 or step == 1 or (terminated | truncated).any():
                print(f"[STEP {step:04d}]")
                print(f"  crash_count       = {mask_count(crash)}")
                print(f"  drift_count       = {mask_count(drift)}")
                print(f"  time_out_count    = {mask_count(time_out)}")
                print(f"  terminated_count  = {mask_count(terminated)}")
                print(f"  truncated_count   = {mask_count(truncated)}")
                print(f"  max_contact_force = {float(contact_force.max().item()):.4f}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
