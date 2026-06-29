# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trace how a task name becomes env cfg, runner cfg, env, wrapper, and learn loop."""

import argparse
import importlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Trace the high-level RSL-RL training pipeline.")
parser.add_argument("--task", type=str, required=True, help="Registered Gym task name.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments.")
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
from isaaclab_tasks.utils import parse_env_cfg


def import_from_entry_point(entry_point: str):
    module_name, attr_name = entry_point.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def main():
    spec = gym.spec(args_cli.task)
    env_cfg_entry = spec.kwargs["env_cfg_entry_point"]
    runner_cfg_entry = spec.kwargs["rsl_rl_cfg_entry_point"]

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    runner_cfg_cls = import_from_entry_point(runner_cfg_entry)
    runner_cfg = runner_cfg_cls()
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        print(f"[INFO] task                    = {args_cli.task}")
        print(f"[INFO] gym entry_point         = {spec.entry_point}")
        print(f"[INFO] env_cfg_entry_point     = {env_cfg_entry}")
        print(f"[INFO] rsl_rl_cfg_entry_point  = {runner_cfg_entry}")
        print(f"[INFO] env_type                = {env.unwrapped.__class__.__name__}")
        print(f"[INFO] runner_cfg_type         = {runner_cfg.__class__.__name__}")
        print()
        print("[PIPELINE]")
        print("  1. import aerial_lab.tasks")
        print("     -> registers the Gym task name")
        print("  2. parse_env_cfg(task, ...)")
        print("     -> resolves MiniQuadcopterEnvCfg and applies CLI overrides")
        print("  3. gym.make(task, cfg=env_cfg)")
        print("     -> instantiates MiniQuadcopterEnv")
        print("  4. RslRlVecEnvWrapper(env)")
        print("     -> adapts Isaac Lab env to rsl_rl")
        print("  5. OnPolicyRunner(env, runner_cfg)")
        print("     -> creates policy, rollout storage, optimizer")
        print("  6. runner.learn(...)")
        print("     -> repeats rollout collection and PPO updates")
        print()
        print("[RUNNER CFG]")
        print(f"  num_steps_per_env        = {runner_cfg.num_steps_per_env}")
        print(f"  max_iterations           = {runner_cfg.max_iterations}")
        print(f"  num_learning_epochs      = {runner_cfg.algorithm.num_learning_epochs}")
        print(f"  num_mini_batches         = {runner_cfg.algorithm.num_mini_batches}")
        print(f"  actor_hidden_dims        = {runner_cfg.policy.actor_hidden_dims}")
        print(f"  critic_hidden_dims       = {runner_cfg.policy.critic_hidden_dims}")
        print()
        rollout_size = env_cfg.scene.num_envs * runner_cfg.num_steps_per_env
        mini_batch_size = rollout_size // runner_cfg.algorithm.num_mini_batches
        updates_per_iteration = (
            runner_cfg.algorithm.num_learning_epochs * runner_cfg.algorithm.num_mini_batches
        )
        print("[DERIVED NUMBERS]")
        print(f"  rollout_size             = num_envs * num_steps_per_env = {rollout_size}")
        print(f"  mini_batch_size          = rollout_size / num_mini_batches = {mini_batch_size}")
        print(f"  updates_per_iteration    = epochs * mini_batches = {updates_per_iteration}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
