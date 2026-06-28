# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a minimal PPO policy for fixed-base kinikun arm joint target tracking."""

import argparse
import random
from collections import deque

from isaaclab.app import AppLauncher

# PPO / reward hyperparameters kept in-code on purpose to keep the CLI short.
# Strongly recommend technical blog of PPO algorithm:
# [Policy Gradients RL](https://towardsdatascience.com/policy-gradients-in-reinforcement-learning-explained-ecec7df94245/)
# [Natural Policy Gradients RL](https://towardsdatascience.com/natural-policy-gradients-in-reinforcement-learning-explained-2265864cf43c/)
# [Trust Region Policy Optimization (TRPO)](https://towardsdatascience.com/trust-region-policy-optimization-trpo-explained-4b56bd206fc2/)
# [Proximal Policy Optimization (PPO)](https://towardsdatascience.com/proximal-policy-optimization-ppo-explained-abed1952457b/)
TRAIN_ITERS_DEFAULT = 200
NUM_ENVS = 32
ENV_SPACING = 2.5
ROLLOUT_STEPS = 256
EPISODE_LENGTH = 120
PRINT_EVERY = 100
SEED = 7
SIM_DT = 0.01
TARGET_SCALE = 0.7
ENABLE_GRAVITY = False

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
UPDATE_EPOCHS = 10
MINI_BATCH_SIZE = 128
ACTION_DELTA_WEIGHT = 0.01
JOINT_VEL_WEIGHT = 0.05
ENTROPY_WEIGHT = 0.0

parser = argparse.ArgumentParser(description="Train a minimal PPO policy for kinikun arm joint tracking.")
parser.add_argument("--train_iters", type=int, default=TRAIN_ITERS_DEFAULT, help="Number of PPO iterations.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import numpy as np
import torch
import torch.nn as nn
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from torch.distributions import Independent, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

try:
    from .common import resolve_camera_view, resolve_kinikun_cfg, spawn_ground_and_light
except ImportError:
    from common import resolve_camera_view, resolve_kinikun_cfg, spawn_ground_and_light


ARM_LOWER_LIMITS = torch.tensor(
    [-0.7853981633974483, -0.7853981633974483, -1.0471975511965976, -0.7853981633974483],
    dtype=torch.float32,
)
ARM_UPPER_LIMITS = torch.tensor(
    [0.7853981633974483, 0.7853981633974483, 1.0471975511965976, 0.7853981633974483],
    dtype=torch.float32,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers.extend((nn.Linear(dims[i], dims[i + 1]), nn.Tanh()))
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.backbone = build_mlp([obs_dim, 128, 128, 128])
        self.mean_head = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.8))

    def forward(self, obs: torch.Tensor):
        hidden = self.backbone(obs)
        mean = self.mean_head(hidden)
        std = torch.exp(self.log_std).expand_as(mean)
        base_dist = Normal(mean, std)
        squashed_dist = TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])
        return Independent(squashed_dist, 1)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = build_mlp([obs_dim, 128, 128, 128, 1])

    def forward(self, obs: torch.Tensor):
        return self.net(obs).squeeze(-1)


def resolve_fixed_base_cfg(enable_gravity: bool | None = None):
    cfg = resolve_kinikun_cfg(enable_gravity=enable_gravity)
    cfg.spawn.fix_base = True
    return cfg.replace(prim_path="/World/envs/env_.*/Robot")


def sample_target_positions(device: torch.device, lower: torch.Tensor, upper: torch.Tensor, scale: float, num_envs: int):
    center = 0.5 * (lower + upper)
    half_range = 0.5 * (upper - lower) * scale
    noise = 2.0 * torch.rand((num_envs, lower.numel()), device=device) - 1.0
    return center.unsqueeze(0) + noise * half_range.unsqueeze(0)


def action_to_joint_targets(action: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor):
    center = 0.5 * (lower + upper)
    half_range = 0.5 * (upper - lower)
    return center.unsqueeze(0) + action * half_range.unsqueeze(0)


def build_observation(robot: Articulation, arm_ids: torch.Tensor, target_joint_pos: torch.Tensor):
    joint_pos = robot.data.joint_pos[:, arm_ids]
    joint_vel = robot.data.joint_vel[:, arm_ids]
    position_error = target_joint_pos - joint_pos
    return torch.cat((joint_pos, joint_vel, target_joint_pos, position_error), dim=-1)


def compute_reward(
    robot: Articulation,
    arm_ids: torch.Tensor,
    target_joint_pos: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
):
    joint_pos = robot.data.joint_pos[:, arm_ids]
    joint_vel = robot.data.joint_vel[:, arm_ids]
    position_error = target_joint_pos - joint_pos

    track_penalty = position_error.square().mean(dim=-1)
    joint_vel_penalty = joint_vel.square().mean(dim=-1)
    action_delta_penalty = (action - prev_action).square().mean(dim=-1)

    reward = (
        -track_penalty
        - JOINT_VEL_WEIGHT * joint_vel_penalty
        - ACTION_DELTA_WEIGHT * action_delta_penalty
    )
    return reward, track_penalty, joint_vel_penalty, action_delta_penalty


def reset_envs(
    robot: Articulation,
    scene: InteractiveScene,
    env_ids: torch.Tensor,
    target_joint_pos: torch.Tensor,
    prev_action: torch.Tensor,
    episode_return: torch.Tensor,
    episode_track_error: torch.Tensor,
    episode_steps: torch.Tensor,
    target_limits,
    sim_dt: float,
):
    if env_ids.numel() == 0:
        return

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    default_root_state = robot.data.default_root_state[env_ids].clone()
    default_root_state[:, :3] += scene.env_origins[env_ids]

    robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    scene.reset(env_ids)
    scene.update(sim_dt)

    lower, upper = target_limits
    target_joint_pos[env_ids] = sample_target_positions(robot.device, lower, upper, TARGET_SCALE, env_ids.numel())
    prev_action[env_ids].zero_()
    episode_return[env_ids].zero_()
    episode_track_error[env_ids].zero_()
    episode_steps[env_ids].zero_()


def compute_gae(rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor, last_value: torch.Tensor):
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    for step in reversed(range(rewards.shape[0])):
        if step == rewards.shape[0] - 1:
            next_value = last_value
        else:
            next_value = values[step + 1]
        non_terminal = 1.0 - dones[step]
        delta = rewards[step] + GAMMA * next_value * non_terminal - values[step]
        gae = delta + GAMMA * GAE_LAMBDA * non_terminal * gae
        advantages[step] = gae
    returns = advantages + values
    return advantages, returns


def ppo_update(
    policy: PolicyNet,
    value_net: ValueNet,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
):
    obs = obs.reshape(-1, obs.shape[-1])
    actions = actions.reshape(-1, actions.shape[-1])
    old_log_probs = old_log_probs.reshape(-1)
    advantages = advantages.reshape(-1)
    returns = returns.reshape(-1)

    batch_size = obs.shape[0]
    mini_batch_size = min(MINI_BATCH_SIZE, batch_size)

    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0

    for _ in range(UPDATE_EPOCHS):
        permutation = torch.randperm(batch_size, device=obs.device)
        for start in range(0, batch_size, mini_batch_size):
            batch_ids = permutation[start : start + mini_batch_size]

            batch_obs = obs[batch_ids]
            batch_actions = actions[batch_ids]
            batch_old_log_probs = old_log_probs[batch_ids]
            batch_advantages = advantages[batch_ids]
            batch_returns = returns[batch_ids]

            dist = policy(batch_obs)
            new_log_probs = dist.log_prob(batch_actions)
            entropy_estimate = -new_log_probs.mean()

            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
            policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
            policy_loss -= ENTROPY_WEIGHT * entropy_estimate

            value_loss = torch.mean((value_net(batch_obs) - batch_returns) ** 2)

            policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            policy_optimizer.step()

            value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
            value_optimizer.step()

            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_entropy = float(entropy_estimate.item())

    return last_policy_loss, last_value_loss, last_entropy


def main():
    set_seed(SEED)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=SIM_DT)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim_dt = sim.get_physics_dt()
    camera_eye, camera_target = resolve_camera_view()
    sim.set_camera_view(camera_eye, camera_target)
    spawn_ground_and_light()

    scene = InteractiveScene(InteractiveSceneCfg(num_envs=NUM_ENVS, env_spacing=ENV_SPACING, replicate_physics=True))
    robot = Articulation(resolve_fixed_base_cfg(enable_gravity=ENABLE_GRAVITY))
    scene.articulations["robot"] = robot
    scene.clone_environments(copy_from_source=False)
    if args_cli.device.startswith("cpu"):
        scene.filter_collisions(global_prim_paths=[])

    sim.reset()
    scene.update(sim_dt)

    arm_id_list, arm_names = robot.find_joints("arm.*_joint")
    arm_ids = torch.tensor(arm_id_list, device=robot.device, dtype=torch.long)
    arm_lower = ARM_LOWER_LIMITS.to(robot.device)
    arm_upper = ARM_UPPER_LIMITS.to(robot.device)
    obs_dim = len(arm_names) * 4
    action_dim = len(arm_names)
    num_envs = robot.num_instances

    policy = PolicyNet(obs_dim=obs_dim, action_dim=action_dim).to(robot.device)
    value_net = ValueNet(obs_dim=obs_dim).to(robot.device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=ACTOR_LR)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=CRITIC_LR)

    print(f"{'iter':>6} {'ep':>6} {'ret':>10} {'mae':>10} {'pi_loss':>12} {'v_loss':>12} {'entropy':>10}")

    episode_returns = deque(maxlen=20)
    episode_mae = deque(maxlen=20)
    total_episodes = 0

    target_joint_pos = torch.zeros((num_envs, action_dim), device=robot.device)
    prev_action = torch.zeros_like(target_joint_pos)
    episode_return = torch.zeros(num_envs, device=robot.device)
    episode_track_error = torch.zeros(num_envs, device=robot.device)
    episode_steps = torch.zeros(num_envs, device=robot.device, dtype=torch.long)
    all_env_ids = torch.arange(num_envs, device=robot.device, dtype=torch.long)

    reset_envs(
        robot,
        scene,
        all_env_ids,
        target_joint_pos,
        prev_action,
        episode_return,
        episode_track_error,
        episode_steps,
        (arm_lower, arm_upper),
        sim_dt,
    )

    for iteration in range(1, args_cli.train_iters + 1):
        if not simulation_app.is_running():
            break

        obs_buffer = []
        action_buffer = []
        log_prob_buffer = []
        reward_buffer = []
        done_buffer = []
        value_buffer = []

        for _ in range(ROLLOUT_STEPS):
            obs = build_observation(robot, arm_ids, target_joint_pos)

            with torch.no_grad():
                dist = policy(obs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = value_net(obs)

            commanded_joint_pos = action_to_joint_targets(action, arm_lower, arm_upper)
            robot.set_joint_position_target(commanded_joint_pos, arm_id_list)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            reward, _, _, _ = compute_reward(
                robot, arm_ids, target_joint_pos, action, prev_action
            )
            episode_steps += 1
            done = episode_steps >= EPISODE_LENGTH

            obs_buffer.append(obs)
            action_buffer.append(action)
            log_prob_buffer.append(log_prob)
            reward_buffer.append(reward)
            done_buffer.append(done.float())
            value_buffer.append(value)

            position_error = (target_joint_pos - robot.data.joint_pos[:, arm_ids]).abs().mean(dim=-1)
            episode_return += reward
            episode_track_error += position_error
            prev_action = action.detach()

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if done_env_ids.numel() > 0:
                total_episodes += done_env_ids.numel()
                episode_returns.extend(episode_return[done_env_ids].detach().cpu().tolist())
                episode_mae.extend(
                    (episode_track_error[done_env_ids] / episode_steps[done_env_ids].float()).detach().cpu().tolist()
                )
                reset_envs(
                    robot,
                    scene,
                    done_env_ids,
                    target_joint_pos,
                    prev_action,
                    episode_return,
                    episode_track_error,
                    episode_steps,
                    (arm_lower, arm_upper),
                    sim_dt,
                )

        obs_batch = torch.stack(obs_buffer)
        action_batch = torch.stack(action_buffer)
        old_log_prob_batch = torch.stack(log_prob_buffer)
        reward_batch = torch.stack(reward_buffer)
        done_batch = torch.stack(done_buffer)
        value_batch = torch.stack(value_buffer)

        with torch.no_grad():
            last_obs = build_observation(robot, arm_ids, target_joint_pos)
            last_value = value_net(last_obs) * (1.0 - done_batch[-1])

        advantages, returns = compute_gae(reward_batch, done_batch, value_batch, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss, value_loss, entropy_estimate = ppo_update(
            policy,
            value_net,
            policy_optimizer,
            value_optimizer,
            obs_batch,
            action_batch,
            old_log_prob_batch,
            advantages,
            returns,
        )

        if iteration % PRINT_EVERY == 0 or iteration == 1 or iteration == args_cli.train_iters:
            mean_return = float(np.mean(episode_returns)) if episode_returns else float("nan")
            mean_mae = float(np.mean(episode_mae)) if episode_mae else float("nan")
            print(
                f"{iteration:6d} {total_episodes:6d} "
                f"{mean_return:10.4f} {mean_mae:10.4f} "
                f"{policy_loss:12.6f} {value_loss:12.6f} {entropy_estimate:10.6f}"
            )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
