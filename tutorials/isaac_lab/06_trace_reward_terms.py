# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trace per-step reward terms by observing the env's internal episode-sum accumulators."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Trace reward terms inside the env.")
parser.add_argument("--task", type=str, required=True, help="Registered Gym task name.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=120, help="Number of environment steps.")
parser.add_argument("--print_every", type=int, default=20, help="How often to print reward terms.")
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


def format_terms(term_map: dict[str, torch.Tensor], env_id: int = 0, max_terms: int = 8) -> str:
    items = []
    for index, (name, value) in enumerate(term_map.items()):
        if index >= max_terms:
            items.append("...")
            break
        items.append(f"{name}={float(value[env_id].item()):.5f}")
    return ", ".join(items)


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
        print(f"[INFO] reward_keys={list(unwrapped._episode_sums.keys())}")
        print("[INFO] per-step terms are estimated from changes in env._episode_sums before reset zeroing")

        previous_sums = {name: value.clone() for name, value in unwrapped._episode_sums.items()}

        for step in range(1, args_cli.steps + 1):
            if not simulation_app.is_running():
                break

            with torch.inference_mode():
                actions = 2 * torch.rand(env.action_space.shape, device=unwrapped.device) - 1
                obs, reward, terminated, truncated, info = env.step(actions)

            current_sums = {name: value.clone() for name, value in unwrapped._episode_sums.items()}
            step_terms = {}
            for name in current_sums:
                delta = current_sums[name] - previous_sums[name]
                done = terminated | truncated
                # When reset zeroes the episode sum, the delta is not meaningful for that env in this step.
                delta = torch.where(done, torch.full_like(delta, float("nan")), delta)
                step_terms[name] = delta
            previous_sums = current_sums

            if step % args_cli.print_every == 0 or step == 1 or step == args_cli.steps:
                done_count = int((terminated | truncated).sum().item())
                print(f"[STEP {step:04d}] reward[0]={float(reward[0].item()):.5f} done_count={done_count}")
                print(f"  terms[env0] = {format_terms(step_terms, env_id=0)}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
