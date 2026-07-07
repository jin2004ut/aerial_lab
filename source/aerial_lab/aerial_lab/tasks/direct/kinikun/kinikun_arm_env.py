# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-base kinikun arm1 joint-tracking environment driven through a NARX model."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from aerial_lab.assets import ISAACLAB_ASSETS_DATA_DIR

from aerial_lab.assets.aerialrobot import KINIKUN_CFG  # isort: skip
from .narx_model import load_narx_model  # isort: skip

# Fixed-base copy of the kinikun asset (deepcopy so we don't mutate the shared global cfg).
_KINIKUN_ARM_CFG = copy.deepcopy(KINIKUN_CFG)
_KINIKUN_ARM_CFG.spawn.fix_base = True
_KINIKUN_ARM_CFG.actuators.pop("arm", None)


@configclass
class KinikunArmEnvCfg(DirectRLEnvCfg):
    # --- timing ---
    # sim_dt matches the NARX training sample rate (narx_meta.json dt_est ~= 0.005s / 200Hz)
    # so the history advances one slice per _apply_action at the rate the model was trained on.
    # decimation=1 -> the policy commands pressure every NARX step, keeping the dp/dt feature
    # continuous and in-distribution (the divisor in _apply_action is sim_dt).
    sim_dt = 1 / 200.0
    decimation = 1
    episode_length_s = 10.0

    # --- play mode ---  (set True by scripts/rsl_rl/play.py)
    # In play mode the target is a sine wave instead of a random per-episode target.
    play_mode = False
    play_sine_period_s = 4.0  # period of the sine-wave target

    # --- spaces ---
    controlled_joint_name = "arm1_joint"
    num_arm_joints = 1
    pressure_channels = 2
    action_space = num_arm_joints * pressure_channels
    observation_space = num_arm_joints * 7
    state_space = observation_space

    # --- task / reward ---
    target_scale = 0.7  # fraction of the joint range used when sampling targets
    track_reward_scale = 1.0
    joint_vel_reward_scale = 0.05
    action_rate_reward_scale = 0.01
    pressure_limit_mpa = 0.6
    narx_model_path = (
        Path(ISAACLAB_ASSETS_DATA_DIR) / "Robots/kinikun/models/out_narx2/narx_model.pt"
    )
    narx_meta_path = (
        Path(ISAACLAB_ASSETS_DATA_DIR)
        / "Robots/kinikun/models/out_narx2/narx_meta.json"
    )

    # --- simulation ---
    sim: SimulationCfg = SimulationCfg(dt=sim_dt, render_interval=decimation)

    # --- scene & robot ---
    robot_cfg: ArticulationCfg = _KINIKUN_ARM_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.0, replicate_physics=True
    )


class KinikunArmEnv(DirectRLEnv):
    cfg: KinikunArmEnvCfg

    def __init__(self, cfg: KinikunArmEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Arm joints and their position limits (read from the asset, not hard-coded).
        self._arm_ids, arm_names = self._robot.find_joints(
            self.cfg.controlled_joint_name
        )
        arm_limits = self._robot.data.soft_joint_pos_limits[
            0, self._arm_ids
        ]  # (num_arm_joints, 2)
        self._arm_lower = arm_limits[:, 0]
        self._arm_upper = arm_limits[:, 1]
        self._arm_center = 0.5 * (self._arm_lower + self._arm_upper)
        self._arm_half_range = 0.5 * (self._arm_upper - self._arm_lower)
        self._narx_model, self._narx_meta = load_narx_model(
            self.cfg.narx_model_path,
            self.cfg.narx_meta_path,
            self.device,
        )

        # The NARX history advances one slice per physics step, so sim_dt must equal the
        # model's training sample rate (dt_est). A mismatch silently distorts the lag window
        # and the dp/dt feature scale, so fail loudly instead.
        if self._narx_meta.dt_est > 0.0 and not math.isclose(
            self.cfg.sim_dt, self._narx_meta.dt_est, rel_tol=0.05
        ):
            raise ValueError(
                f"sim_dt ({self.cfg.sim_dt:.6f}s) does not match the NARX training "
                f"dt_est ({self._narx_meta.dt_est:.6f}s). Set cfg.sim_dt to the model's "
                "sample rate so the history/derivative timing matches training."
            )

        self._sine_phase = torch.zeros(self.cfg.num_arm_joints, device=self.device)

        # Per-env buffers.
        n = self.num_envs
        self._actions = torch.zeros(n, self.cfg.action_space, device=self.device)
        self._last_actions = torch.zeros(n, self.cfg.action_space, device=self.device)
        self._target_joint_pos = torch.zeros(
            n, self.cfg.num_arm_joints, device=self.device
        )
        self._pressure_cmd = torch.zeros(
            n, self.cfg.num_arm_joints, self.cfg.pressure_channels, device=self.device
        )
        self._prev_pressure_cmd = torch.zeros_like(self._pressure_cmd)
        self._narx_history = torch.zeros(
            n,
            self.cfg.num_arm_joints,
            self._narx_meta.lags,
            self._narx_meta.feature_dim,
            device=self.device,
        )

        self._episode_sums = {
            key: torch.zeros(n, device=self.device)
            for key in ["track", "joint_vel", "action_rate"]
        }

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Kinikun arm env | fixed_base=True | arm joint:", arm_names[0])
        print("Arm lower limit:", self._arm_lower.tolist())
        print("Arm upper limit:", self._arm_upper.tolist())
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        spawn_ground_plane("/World/ground", GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # In play mode the tracking target is a per-joint sine wave that moves every step.
        if self.cfg.play_mode:
            t = self.episode_length_buf.unsqueeze(-1) * self.step_dt  # (num_envs, 1)
            sine = torch.sin(
                2.0 * math.pi * t / self.cfg.play_sine_period_s + self._sine_phase
            )
            self._target_joint_pos = (
                self._arm_center + self.cfg.target_scale * self._arm_half_range * sine
            )

        # Store action history, then map normalized action [-1, 1] -> pressure commands.
        self._last_actions = self._actions.clone()
        self._actions = actions.clamp(-1.0, 1.0).to(self.device)
        self._pressure_cmd = 0.5 * (
            self._actions.view(self.num_envs, self.cfg.num_arm_joints, 2) + 1.0
        )
        self._pressure_cmd *= self.cfg.pressure_limit_mpa

    def _apply_action(self) -> None:
        joint_pos = self._robot.data.joint_pos[:, self._arm_ids]
        joint_vel = self._robot.data.joint_vel[:, self._arm_ids]
        pressure_rate = (self._pressure_cmd - self._prev_pressure_cmd) / self.cfg.sim_dt
        current_features = torch.stack(
            (
                joint_pos,
                self._pressure_cmd[:, :, 0],
                self._pressure_cmd[:, :, 1],
                pressure_rate[:, :, 0],
                pressure_rate[:, :, 1],
            ),
            dim=-1,
        )
        self._narx_history = torch.roll(self._narx_history, shifts=-1, dims=2)
        self._narx_history[:, :, -1, :] = current_features

        narx_input = self._narx_history.view(
            self.num_envs * self.cfg.num_arm_joints, -1
        )
        narx_input = (
            narx_input - self._narx_meta.mu.unsqueeze(0)
        ) / self._narx_meta.std.unsqueeze(0)
        with torch.no_grad():
            predicted_joint_pos = self._narx_model(narx_input).view(
                self.num_envs, self.cfg.num_arm_joints
            )
        predicted_joint_pos = torch.clamp(
            predicted_joint_pos, self._arm_lower, self._arm_upper
        )
        predicted_joint_vel = (
            0.5 * ((predicted_joint_pos - joint_pos) / self.cfg.sim_dt)
            + 0.5 * joint_vel
        )

        self._robot.write_joint_state_to_sim(
            predicted_joint_pos, predicted_joint_vel, self._arm_ids
        )
        self._prev_pressure_cmd.copy_(self._pressure_cmd)

    def _get_observations(self) -> dict:
        joint_pos = self._robot.data.joint_pos[:, self._arm_ids]
        joint_vel = self._robot.data.joint_vel[:, self._arm_ids]
        position_error = self._target_joint_pos - joint_pos
        last_pressure = self._last_actions
        pressure_delta = self._pressure_cmd[:, :, 0] - self._pressure_cmd[:, :, 1]
        obs = torch.cat(
            (
                joint_pos,
                joint_vel,
                self._target_joint_pos,
                position_error,
                last_pressure,
                pressure_delta,
            ),
            dim=-1,
        )
        return {"policy": obs, "critic": obs}

    def _get_rewards(self) -> torch.Tensor:
        joint_pos = self._robot.data.joint_pos[:, self._arm_ids]
        joint_vel = self._robot.data.joint_vel[:, self._arm_ids]
        position_error = self._target_joint_pos - joint_pos

        track = position_error.square().mean(dim=-1)
        joint_vel_pen = joint_vel.square().mean(dim=-1)
        action_rate = (self._actions - self._last_actions).square().mean(dim=-1)

        rewards = {
            "track": -self.cfg.track_reward_scale * track,
            "joint_vel": -self.cfg.joint_vel_reward_scale * joint_vel_pen,
            "action_rate": -self.cfg.action_rate_reward_scale * action_rate,
        }
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return torch.sum(torch.stack(list(rewards.values())), dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros_like(time_out)  # fixed base -> never terminates early
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # Logging: average tracking error (MAE) and reward sums over the finished episodes.
        joint_pos = self._robot.data.joint_pos[env_ids][:, self._arm_ids]
        final_mae = (self._target_joint_pos[env_ids] - joint_pos).abs().mean()
        extras = {}
        for key in self._episode_sums.keys():
            extras["Episode_Reward/" + key] = torch.mean(
                self._episode_sums[key][env_ids]
            ).item()
            self._episode_sums[key][env_ids] = 0.0
        extras["Metrics/final_tracking_mae"] = final_mae.item()
        self.extras["log"] = extras

        super()._reset_idx(env_ids)

        # Reset the arm to its default joint state.
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Sample a new joint-position target inside target_scale * joint range.
        # (In play mode the target is a sine wave driven in _pre_physics_step, so skip sampling.)
        if not self.cfg.play_mode:
            noise = (
                2.0
                * torch.rand(len(env_ids), self.cfg.num_arm_joints, device=self.device)
                - 1.0
            )
            self._target_joint_pos[env_ids] = (
                self._arm_center + noise * self._arm_half_range * self.cfg.target_scale
            )

        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._pressure_cmd[env_ids] = 0.0
        self._prev_pressure_cmd[env_ids] = 0.0
        initial_features = torch.stack(
            (
                joint_pos[:, self._arm_ids],
                torch.zeros(len(env_ids), self.cfg.num_arm_joints, device=self.device),
                torch.zeros(len(env_ids), self.cfg.num_arm_joints, device=self.device),
                torch.zeros(len(env_ids), self.cfg.num_arm_joints, device=self.device),
                torch.zeros(len(env_ids), self.cfg.num_arm_joints, device=self.device),
            ),
            dim=-1,
        )
        self._narx_history[env_ids] = initial_features.unsqueeze(2).repeat(
            1, 1, self._narx_meta.lags, 1
        )
