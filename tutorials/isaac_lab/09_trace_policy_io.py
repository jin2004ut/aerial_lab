# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trace the observation/action interface that the policy network sees."""

import argparse
import importlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Trace policy input and output shapes.")
parser.add_argument("--task", type=str, required=True, help="Registered Gym task name.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=5, help="Number of environment steps.")
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


def import_from_entry_point(entry_point: str):
    module_name, attr_name = entry_point.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def get_policy_obs(obs):
    if hasattr(obs, "keys"):
        if "policy" in obs:
            return obs["policy"]
        first_key = next(iter(obs))
        return obs[first_key]
    return obs


def main():
    spec = gym.spec(args_cli.task)
    runner_cfg_cls = import_from_entry_point(spec.kwargs["rsl_rl_cfg_entry_point"])
    runner_cfg = runner_cfg_cls()
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        obs, info = env.reset()
        policy_obs = get_policy_obs(obs)
        print(f"[INFO] task={args_cli.task}")
        print(f"[INFO] policy_obs_shape={tuple(policy_obs.shape)}")
        print(f"[INFO] action_space_shape={env.action_space.shape}")
        print(f"[INFO] actor_hidden_dims={runner_cfg.policy.actor_hidden_dims}")
        print(f"[INFO] critic_hidden_dims={runner_cfg.policy.critic_hidden_dims}")
        print("[INFO] this tutorial uses a tiny dummy MLP only to illustrate I/O shapes")

        obs_dim = policy_obs.shape[-1]
        action_dim = env.action_space.shape[-1]
        dummy_actor = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, runner_cfg.policy.actor_hidden_dims[0]),
            torch.nn.ELU(),
            torch.nn.Linear(runner_cfg.policy.actor_hidden_dims[0], action_dim),
        ).to(env.unwrapped.device)

        for step in range(1, args_cli.steps + 1):
            if not simulation_app.is_running():
                break

            with torch.inference_mode():
                policy_obs = get_policy_obs(obs)
                action_mean = dummy_actor(policy_obs)
                clipped_actions = torch.tanh(action_mean)
                obs, reward, terminated, truncated, info = env.step(clipped_actions)

            print(f"[STEP {step:04d}]")
            print(f"  obs_batch_shape     = {tuple(policy_obs.shape)}")
            print(f"  action_mean_shape   = {tuple(action_mean.shape)}")
            print(f"  action_sent_shape   = {tuple(clipped_actions.shape)}")
            print(f"  first_action        = {clipped_actions[0].detach().cpu().tolist()}")
            print(f"  reward_mean         = {float(reward.mean().item()):.4f}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
