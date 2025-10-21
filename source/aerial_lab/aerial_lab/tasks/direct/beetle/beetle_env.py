# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip
from aerial_lab.assets.aerialrobot import *  # isort: skip
from isaaclab.sensors import ContactSensorCfg, ContactSensor  # isort: skip


class PoseTrackingEnvWindow(BaseEnvWindow):
    """Window manager for the Beetle environment."""

    def __init__(self, env: BeetleEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class BeetleEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    rotor_num = 4
    gimbal_num = 4
    # 4 gimbals, 4 rotors
    action_space = 8
    # linear velocity (3)
    # angular velocity (3)
    # projected gravity (3)
    # distance_to_goal (local frame) (3)
    # servo positions (4)
    # last action (8)
    observation_space = 24
    thrust_to_torque_ratio = 0.05
    rotor_direction = [1, -1, 1, -1]
    contact_force_threshold = 0.1
    state_space = 0
    debug_vis = True

    ui_window_class_type = PoseTrackingEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot: ArticulationCfg = BEETLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot: ArticulationCfg = BEETLE_OMNI_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )
    # import ipdb; ipdb.set_trace()
    # robot

    # contact_sensor: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/root",   # Bind to the robot root link
    #     history_length=1,
    #     update_period=0,                   # Update every physics step
    #     track_air_time=True,
    #     debug_vis=False,
    #     filter_prim_paths_expr=["/World/ground"],  # Only track contacts with the ground
    #     # filter_prim_paths_expr=[terrain.prim_path],  # Only track contacts with the ground
    # )

    thrust_to_weight = 1.9
    moment_scale = 0.01

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0


class BeetleEnv(DirectRLEnv):
    cfg: BeetleEnvCfg

    def __init__(self, cfg: BeetleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print("BeeetleEnv created num_envs:", self.num_envs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, self.cfg.rotor_num, device=self.device)
        self._gimbal_pos = torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device)
        self._rotor_torque = torch.zeros(self.num_envs, self.cfg.rotor_num, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("root")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        self._thrust_ids = self._robot.find_bodies("thrust.*")
        self._gimbal_ids = self._robot.find_joints("gimbal.*")
        self._rotor_ids = self._robot.find_joints("rotor.*")

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Beetle robot link list: ")
        for link in self._robot.body_names:
            print(f" - {link}")
        print("Beetle robot joint list: ")
        for joint in self._robot.joint_names:
            print(f" - {joint}")
        print("Thrust IDs: ", self._thrust_ids)
        print("Gimbal IDs: ", self._gimbal_ids)
        print("Rotor IDs: ", self._rotor_ids)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        self._thrust_ids = self._thrust_ids[0]
        self._gimbal_ids = self._gimbal_ids[0]
        self._rotor_ids = self._rotor_ids[0]
        # self._undesired_contact_body_ids = self._contact_sensor.find_bodies("root")

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)

        # self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.articulations["robot"] = self._robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._gimbal_pos = self._actions[:, :self.cfg.gimbal_num] * 1.57  # scale to [-1.57, 1.57] rad
        self._thrust = self._actions[:, self.cfg.gimbal_num :]  # shape: (N, 4)
        self._rotor_torque = (
            -self.cfg.thrust_to_torque_ratio
            * self._thrust
            * torch.tensor(self.cfg.rotor_direction, device=self.device)
        )

    def _apply_action(self):
        self._robot.set_joint_position_target(self._gimbal_pos, self._gimbal_ids)
        self._robot.set_joint_effort_target(self._rotor_torque, self._rotor_ids)
        target_thrust = torch.zeros(self.num_envs, self.cfg.rotor_num, 3, device=self.device)
        target_thrust[:, :, 2] = self._thrust
        target_torque = torch.zeros_like(target_thrust)
        self._robot.set_external_force_and_torque(forces=target_thrust, torques=target_torque, body_ids=self._thrust_ids)

    def _get_observations(self) -> dict:
        # import ipdb; ipdb.set_trace()
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # net_contact_forces_l2m = torch.linalg.norm(self._contact_sensor.data.net_forces_w_history, dim=-1)
        # died = torch.logical_or(
        #     self._robot.data.root_pos_w[:, 2] < 0.2,
        #     net_contact_forces_l2m > self.cfg.contact_force_threshold
        # )
        died = self._robot.data.root_pos_w[:, 2] < 0.2
        # died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
