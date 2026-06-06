# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve an Isaac Lab task into its environment config and inspect key fields."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect an Isaac Lab environment config.")
parser.add_argument("--task", type=str, required=True, help="Registered Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import aerial_lab.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    print(f"[INFO] task={args_cli.task}")
    print(f"[INFO] cfg_type={env_cfg.__class__.__name__}")
    print(f"[INFO] sim.device={env_cfg.sim.device}")
    print(f"[INFO] sim.dt={env_cfg.sim.dt}")
    print(f"[INFO] decimation={getattr(env_cfg, 'decimation', 'n/a')}")
    print(f"[INFO] scene.num_envs={env_cfg.scene.num_envs}")
    print(f"[INFO] scene.env_spacing={env_cfg.scene.env_spacing}")
    print(f"[INFO] action_space={getattr(env_cfg, 'action_space', 'n/a')}")
    print(f"[INFO] observation_space={getattr(env_cfg, 'observation_space', 'n/a')}")
    print(f"[INFO] episode_length_s={getattr(env_cfg, 'episode_length_s', 'n/a')}")
    print(f"[INFO] add_noise={getattr(env_cfg, 'add_noise', 'n/a')}")
    print(f"[INFO] add_randomization={getattr(env_cfg, 'add_randomization', 'n/a')}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
