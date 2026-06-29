# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trace observation, reward, and done signals from a running Isaac Lab env."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Trace observation, reward, and done signals.")
parser.add_argument("--task", type=str, required=True, help="Registered Gym task name.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=120, help="Number of environment steps.")
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


def fmt_row(tensor: torch.Tensor, decimals: int = 4) -> str:
    values = tensor.detach().cpu().tolist()
    return "[" + ", ".join(f"{value:.{decimals}f}" for value in values) + "]"


def first_policy_obs(obs):
    if hasattr(obs, "keys"):
        if "policy" in obs:
            return obs["policy"][0]
        first_key = next(iter(obs))
        return obs[first_key][0]
    return obs[0]


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        obs, info = env.reset()
        print(f"[INFO] task={args_cli.task}")
        print(f"[INFO] env_type={env.unwrapped.__class__.__name__}")
        print(f"[INFO] first_obs_shape={getattr(first_policy_obs(obs), 'shape', None)}")

        for step in range(1, args_cli.steps + 1):
            if not simulation_app.is_running():
                break

            with torch.inference_mode():
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                obs, reward, terminated, truncated, info = env.step(actions)

            if step % 30 == 0 or step == 1 or step == args_cli.steps:
                done = terminated | truncated
                policy_obs = first_policy_obs(obs)
                head = policy_obs[: min(8, policy_obs.shape[0])]
                print(f"[STEP {step:04d}]")
                print(f"  obs_head           = {fmt_row(head)}")
                print(f"  reward[0]          = {float(reward[0].item()):.4f}")
                print(f"  terminated_count   = {int(terminated.sum().item())}")
                print(f"  truncated_count    = {int(truncated.sum().item())}")
                print(f"  done_count         = {int(done.sum().item())}")
                if isinstance(info, dict) and info:
                    print(f"  info_keys          = {sorted(info.keys())}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
