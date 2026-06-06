# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compute the rollout/iteration/epoch/minibatch numbers for a task's PPO config."""

import argparse
import importlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Trace rollout buffer sizing from the task's PPO config.")
parser.add_argument("--task", type=str, required=True, help="Registered Gym task name.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments to assume.")
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
    runner_cfg_cls = import_from_entry_point(spec.kwargs["rsl_rl_cfg_entry_point"])
    runner_cfg = runner_cfg_cls()
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    env_step_dt = env_cfg.sim.dt * getattr(env_cfg, "decimation", 1)
    rollout_size = env_cfg.scene.num_envs * runner_cfg.num_steps_per_env
    mini_batch_size = rollout_size // runner_cfg.algorithm.num_mini_batches
    env_seconds_per_iteration = runner_cfg.num_steps_per_env * env_step_dt
    updates_per_iteration = runner_cfg.algorithm.num_learning_epochs * runner_cfg.algorithm.num_mini_batches

    print(f"[INFO] task={args_cli.task}")
    print(f"[INFO] num_envs={env_cfg.scene.num_envs}")
    print(f"[INFO] sim.dt={env_cfg.sim.dt}")
    print(f"[INFO] decimation={getattr(env_cfg, 'decimation', 'n/a')}")
    print(f"[INFO] env_step_dt={env_step_dt}")
    print(f"[INFO] episode_length_s={env_cfg.episode_length_s}")
    print()
    print("[ROLLOUT]")
    print(f"  num_steps_per_env         = {runner_cfg.num_steps_per_env}")
    print(f"  rollout_size              = {rollout_size}")
    print(f"  env_seconds_per_iteration = {env_seconds_per_iteration}")
    print()
    print("[PPO UPDATE]")
    print(f"  num_learning_epochs       = {runner_cfg.algorithm.num_learning_epochs}")
    print(f"  num_mini_batches          = {runner_cfg.algorithm.num_mini_batches}")
    print(f"  mini_batch_size           = {mini_batch_size}")
    print(f"  optimizer_updates/iter    = {updates_per_iteration}")
    print()
    print("[MENTAL MODEL]")
    print("  step       = one env.step(action)")
    print("  iteration  = collect rollout_size samples, then update")
    print("  epoch      = one full pass over the collected rollout")
    print("  minibatch  = one chunk used for one optimizer update")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
