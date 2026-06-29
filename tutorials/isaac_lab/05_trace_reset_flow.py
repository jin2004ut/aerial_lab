# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trace how done signals lead into per-env resets in Isaac Lab."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Trace done signals and reset flow.")
parser.add_argument("--task", type=str, required=True, help="Registered Gym task name.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=400, help="Number of environment steps.")
parser.add_argument("--print_every", type=int, default=40, help="How often to print heartbeat logs.")
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


def fmt_ids(mask: torch.Tensor, max_ids: int = 8) -> str:
    env_ids = torch.nonzero(mask, as_tuple=False).reshape(-1).detach().cpu().tolist()
    head = env_ids[:max_ids]
    suffix = ", ..." if len(env_ids) > max_ids else ""
    return "[" + ", ".join(str(idx) for idx in head) + suffix + "]"


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
        print("[INFO] tracing terminated/truncated masks and post-reset state")

        for step in range(1, args_cli.steps + 1):
            if not simulation_app.is_running():
                break

            # Push the system a bit harder than pure random actions so resets show up sooner.
            actions = 2 * torch.rand(env.action_space.shape, device=unwrapped.device) - 1
            actions = actions * 1.5

            with torch.inference_mode():
                obs, reward, terminated, truncated, info = env.step(actions)

            done = terminated | truncated
            if done.any():
                print(f"[STEP {step:04d}] reset happened")
                print(f"  terminated_envs  = {fmt_ids(terminated)}")
                print(f"  truncated_envs   = {fmt_ids(truncated)}")
                print(f"  done_envs        = {fmt_ids(done)}")
                print(f"  episode_len_done = {unwrapped.episode_length_buf[done].detach().cpu().tolist()[:8]}")
                print(f"  root_pos_z_done  = {unwrapped._robot.data.root_pos_w[done, 2].detach().cpu().tolist()[:8]}")
                print(f"  goal_pos_done    = {unwrapped._desired_pos_w[done].detach().cpu().tolist()[:2]}")
                if isinstance(getattr(unwrapped, "extras", None), dict):
                    log_info = unwrapped.extras.get("log", {})
                    if log_info:
                        keys = sorted(log_info.keys())
                        print(f"  extras.log keys  = {keys[:8]}{' ...' if len(keys) > 8 else ''}")

            if step % args_cli.print_every == 0 or step == 1 or step == args_cli.steps:
                print(
                    f"[HEARTBEAT {step:04d}] mean_episode_len={float(unwrapped.episode_length_buf.float().mean().item()):.2f} "
                    f"active_root_z_mean={float(unwrapped._robot.data.root_pos_w[:, 2].mean().item()):.3f}"
                )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
