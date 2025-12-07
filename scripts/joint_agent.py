# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
from operator import imod

import ipdb
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import aerial_lab.tasks  # noqa: F401
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab_tasks.utils import parse_env_cfg
from aerial_lab.utility.plotLogger import ObservationLogger  # isort: skip
import math
import numpy as np
import os


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    # downsample number of environments for random agent
    args_cli.num_envs = 4
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.scene.env_spacing = 1.0
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    counter = 0
    plot_logger = ObservationLogger(save_dir=os.path.join("/home/wentao/", "plots"), max_samples=100 * 50, plot_interval=5.0)
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            counter += 1
            gimbal_target = 2 * torch.sin(torch.tensor(counter * env.unwrapped.step_dt / 4.0 * math.pi * 2))  # oscillate between -2 and 2
            actions[:, 0] = gimbal_target  # set gimbal 1 target
            actions[:, 1] = -gimbal_target  # set gimbal 2 target
            actions[:, 2] = gimbal_target  # set gimbal 3 target
            actions[:, 3] = -gimbal_target  # set gimbal 4 target
            # thrust_target = 200 * torch.sin(torch.tensor(counter / 50.0))  # constant thrust
            # thrust_target = torch.full((1, 4), 0, device=env.unwrapped.device)
            # actions[:, 4:8] = thrust_target  # set thrust targets
            # actions[:, 4] = 1.0
            # actions[:, 5] = 1.0
            # actions[:, 6] = 1.0
            # actions[:, 7] = 1.0
            # apply actions
            obs, _, _, _, _ = env.step(actions)
            # import ipdb; ipdb.set_trace()
            obs_np = obs["policy"][0].cpu().numpy()
            obs_39 = np.concatenate([
                obs_np[:12],  # lin_vel(3) + ang_vel(3) + gravity(3) + goal_pos(3)
                np.zeros(3, dtype=np.float32),  # placeholder for angular_error
                obs_np[12:],  # gimbal(4) + root_rot(6) + goal_rot(6) + last_action(8)
            ])
            actions_np = actions[0].cpu().numpy()
            plot_logger.log(obs_39, actions_np)
            if counter * env.unwrapped.step_dt > 100.0:
                plot_logger.save_to_csv(os.path.join("joint_resoponse_200Hz"))
                plot_logger.save_final_plot()
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
