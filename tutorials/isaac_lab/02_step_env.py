# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create an Isaac Lab Gym environment and step it for a fixed number of steps."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step an Isaac Lab environment.")
parser.add_argument("--task", type=str, required=True, help="Registered Gym task name.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=240, help="Number of environment steps.")
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


def obs_summary(obs) -> str:
    if hasattr(obs, "keys"):
        return ", ".join(f"{key}:{getattr(value, 'shape', None)}" for key, value in obs.items())
    return str(getattr(obs, "shape", None))


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        obs, info = env.reset()
        print(f"[INFO] task={args_cli.task}")
        print(f"[INFO] env_type={env.unwrapped.__class__.__name__}")
        print(f"[INFO] observation={obs_summary(obs)}")
        print(f"[INFO] action_space={env.action_space}")
        print(f"[INFO] device={env.unwrapped.device}")

        for step in range(1, args_cli.steps + 1):
            if not simulation_app.is_running():
                break

            with torch.inference_mode():
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                obs, reward, terminated, truncated, info = env.step(actions)

            if step % 60 == 0 or step == 1 or step == args_cli.steps:
                resets = int((terminated | truncated).sum().item())
                reward_std = float(reward.std(unbiased=False).item()) if reward.numel() > 0 else 0.0
                print(
                    f"[STEP {step:04d}] reward_mean={float(reward.mean().item()):.4f} "
                    f"reward_std={reward_std:.4f} resets={resets}"
                )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
