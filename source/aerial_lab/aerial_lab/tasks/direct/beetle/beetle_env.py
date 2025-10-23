# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, Imu, ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform, subtract_frame_transforms

from aerial_lab.assets.aerialrobot import BEETLE_CFG, MINI_QUADROTOR_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG  # isort: skip


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
                    # self._create_debug_vis_ui_element("contact_sensor", self.env)


@configclass
class BeetleEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 5.0
    # - spaces definition
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
    thrust_to_torque_ratio = 0.0165
    # rotor_direction = [1, -1, 1, -1]
    rotor_direction = [1, 1, 1, 1]
    contact_force_threshold = 0.1

    thrust_limit = 16.0  # N
    gimbal_limit = math.pi / 2  # rad
    state_space = 0

    # custom parameters/scales
    debug_vis = True

    ui_window_class_type = PoseTrackingEnvWindow

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    reach_lin_vel_reward_scale = -0.05
    reach_ang_vel_reward_scale = -0.1
    thrust_power_reward_scale = -1.0e-4
    distance_to_goal_reward_scale = 5.0

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

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # robot_cfg: ArticulationCfg = MINI_QUADROTOR_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg: ArticulationCfg = BEETLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/root",  # Bind to the robot root link
        history_length=1,
        update_period=0,  # Update every physics step
        track_air_time=True,
        debug_vis=True,
        # filter_prim_paths_expr=["/World/ground"],  # Only track contacts with the ground
        filter_prim_paths_expr=[terrain.prim_path],  # Only track contacts with the ground
    )

    # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#inertia-measurement-unit
    imu_sensor: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/root",
        update_period=0,
        history_length=1,
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),  # w, x, y, z
        ),
        debug_vis=True,
    )


class BeetleEnv(DirectRLEnv):
    cfg: BeetleEnvCfg

    def __init__(self, cfg: BeetleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._last_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._target_thrust_force = torch.zeros(self.num_envs, self.cfg.rotor_num, device=self.device)
        self._target_gimbal_pos = torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device)
        self._target_rotor_torque = torch.zeros(self.num_envs, self.cfg.rotor_num, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "reach_lin_vel",
                "reach_ang_vel",
                "thrust_power",
            ]
        }

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

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        # # # add articulation to scene
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot
        # # # setup IMU sensor
        self._imu_sensor = Imu(self.cfg.imu_sensor)
        self.scene.sensors["imu_sensor"] = self._imu_sensor
        # # # setup contact sensor
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        # # # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._last_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)  # TODO: check action limits
        self._target_gimbal_pos = (
            self._actions[:, : self.cfg.gimbal_num] * self.cfg.gimbal_limit
        )  # scale to [-1.57, 1.57] rad
        self._target_thrust_force = (
            (self._actions[:, self.cfg.gimbal_num :] + 1.0) / 2.0 * self.cfg.thrust_limit
        )  # shape: (N, 4)
        self._target_rotor_torque = (
            self.cfg.thrust_to_torque_ratio
            * self._target_thrust_force
            * torch.tensor(self.cfg.rotor_direction, device=self.device)
        )
        # self._target_thrust_force[:, :] = 0
        # print("Rotor velocities: ", self._rotor_vel[0])
        # print("Rotor velocities: ", self._robot.data.joint_vel[:, self._rotor_ids[0]][0])
        # print("Rotor positions: ", self._robot.data.joint_pos[:, self._rotor_ids[0]][0])
        #################################################
        # print("Action debug info:")
        # print(" Gimbal pos targets: ", self._target_gimbal_pos[0])
        # print(" Rotor thrust targets: ", self._target_thrust_force[0])
        # print(" Rotor torque targets: ", self._target_rotor_torque[0])
        #################################################

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._target_gimbal_pos, self._gimbal_ids[0])
        # self._robot.set_joint_effort_target(self._target_rotor_torque, self._rotor_ids[0])
        # self._robot.set_joint_velocity_target(self._target_rotor_torque, self._rotor_ids[0])
        # target_position = torch.zeros_like(self._target_rotor_torque)
        # self._robot.set_joint_position_target(target_position, self._rotor_ids[0])
        # print("Applying rotor torques: ", self._target_rotor_torque[0])
        # print("Applying rotor thrusts: ", self._target_thrust_force[0])
        target_thrust = torch.zeros(self.num_envs, self.cfg.rotor_num, 3, device=self.device)
        target_thrust[:, :, 2] = self._target_thrust_force
        target_torque = torch.zeros(self.num_envs, self.cfg.rotor_num, 3, device=self.device)
        # target_torque[:, :, 2] = self._target_rotor_torque
        # self._robot.set_external_force_and_torque(forces=target_thrust, body_ids=self._thrust_ids)
        # self._robot.set_external_force_and_torque(forces=target_thrust, torques=target_torque, body_ids=self._thrust_ids[0])
        # print("Thrust sequences applied: ", self._target_thrust_force[0])
        # print("Thrust sequences applied: ", self._thrust_ids)
        body_force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        body_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)
        body_torque[:, 0, 2] = torch.sum(self._target_rotor_torque, dim=1)
        # self._robot.set_external_force_and_torque(forces=body_force, torques=body_torque, body_ids=self._body_id)
        # self._robot.set_external_force_and_torque(
        #     forces=target_thrust, torques=target_torque, body_ids=self._thrust_ids[0]
        # )
        applied_thrust = torch.cat([target_thrust, body_force], dim=1)
        applied_torque = torch.cat([target_torque, body_torque], dim=1)
        applied_ids = self._thrust_ids[0] + self._body_id
        self._robot.set_external_force_and_torque(forces=applied_thrust, torques=applied_torque, body_ids=applied_ids)

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        obs = torch.cat(
            (
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
                self._robot.data.joint_pos[:, self._gimbal_ids[0]],
                # self._robot.data.joint_vel[:, self._rotor_ids[0]],
                self._last_actions,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_states(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        states = torch.cat(
            (
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
                self._robot.data.joint_pos[:, self._gimbal_ids[0]],
                # self._robot.data.joint_vel[:, self._rotor_ids[0]],
                self._last_actions,
            ),
            dim=-1,
        )
        return states

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        distance_to_goal_weight = torch.exp(-torch.square(distance_to_goal))
        reach_lin_vel = torch.linalg.norm(self._robot.data.root_lin_vel_b, dim=-1) * distance_to_goal_weight
        reach_ang_vel = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=-1) * distance_to_goal_weight
        thrust_power = torch.sum(torch.square(self._target_thrust_force), dim=1)

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "reach_lin_vel": reach_lin_vel * self.cfg.reach_lin_vel_reward_scale * self.step_dt,
            "reach_ang_vel": reach_ang_vel * self.cfg.reach_ang_vel_reward_scale * self.step_dt,
            "thrust_power": thrust_power * self.cfg.thrust_power_reward_scale * self.step_dt,
        }
        total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        # total_reward = torch.zeros(self.num_envs, device=self.device)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # import ipdb; ipdb.set_trace()
        # died = torch.linalg.norm(self._contact_sensor.data.force_matrix_w.squeeze(1).squeeze(1), dim=-1) > 0.1
        crash = torch.linalg.norm(self._contact_sensor.data.net_forces_w.squeeze(1), dim=-1) > 0.1
        drift = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 5.0)
        died = torch.logical_or(crash, drift)
        #################################################
        # print(torch.linalg.norm(self._contact_sensor.data.force_matrix_w.squeeze(1).squeeze(1), dim=-1))  # die if in contact with the ground
        # print(self._contact_sensor.data.force_matrix_w.squeeze(1).squeeze(1))
        # print(self._contact_sensor.data.net_forces_w.squeeze(1))
        # died = torch.zeros_like(time_out, dtype=torch.bool)
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
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
        self._last_actions[env_ids] = 0.0
        self._target_gimbal_pos[env_ids] = 0.0
        self._target_thrust_force[env_ids] = 0.0
        self._target_rotor_torque[env_ids] = 0.0
        # self._gimbal_pos[env_ids] = 0.0
        # self._gimbal_vel[env_ids] = 0.0
        # self._rotor_vel[env_ids] = 0.0
        # self._rotor_force[env_ids] = 0.0
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
                marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.1)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 1.0)
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
