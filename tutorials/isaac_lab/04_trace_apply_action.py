# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trace how policy actions become simulator commands inside a direct Isaac Lab env."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Trace how actions are transformed inside the env.")
parser.add_argument("--task", type=str, required=True, help="Registered Gym task name.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=8, help="Number of environment steps to trace.")
parser.add_argument("--print_every", type=int, default=1, help="How often to print the trace.")
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


def fmt_tensor(tensor: torch.Tensor, max_items: int = 6, decimals: int = 4) -> str:
    row = tensor.detach().cpu().reshape(-1).tolist()
    head = row[:max_items]
    suffix = ", ..." if len(row) > max_items else ""
    return "[" + ", ".join(f"{value:.{decimals}f}" for value in head) + suffix + "]"


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
        print(f"[INFO] action_dim={env.action_space.shape[-1]}")
        print("[INFO] tracing env._actions -> env._action_thrust_force -> env._target_thrust_force / torque")

        for step in range(1, args_cli.steps + 1):
            if not simulation_app.is_running():
                break

            phase = step / max(args_cli.steps, 1) * torch.pi
            action_dim = env.action_space.shape[-1]
            base = torch.linspace(-0.75, 0.75, action_dim, device=unwrapped.device)
            actions = base.unsqueeze(0).repeat(args_cli.num_envs, 1)
            actions = actions + 0.25 * torch.sin(torch.ones_like(actions) * phase)

            with torch.inference_mode():
                obs, reward, terminated, truncated, info = env.step(actions)

            if step % args_cli.print_every == 0 or step == 1 or step == args_cli.steps:
                print(f"[STEP {step:04d}]")
                print(f"  sampled_action[0]        = {fmt_tensor(actions[0])}")
                print(f"  clipped_env_action[0]    = {fmt_tensor(unwrapped._actions[0])}")
                print(f"  thrust_cmd[0]            = {fmt_tensor(unwrapped._action_thrust_force[0])}")
                print(f"  target_force[0]          = {fmt_tensor(unwrapped._target_thrust_force[0])}")
                print(f"  target_torque[0]         = {fmt_tensor(unwrapped._target_rotor_torque[0])}")
                print(f"  root_lin_vel_b[0]        = {fmt_tensor(unwrapped._robot.data.root_lin_vel_b[0])}")
                print(f"  reward[0]                = {float(reward[0].item()):.4f}")
                print(f"  done_count               = {int((terminated | truncated).sum().item())}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
