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
from aerial_lab.actuators.rotor import Rotor
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, Imu, ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    compute_pose_error,
    matrix_from_euler,
    matrix_from_quat,
    normalize,
    quat_apply,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
)

from aerial_lab.assets.aerialrobot import BEETLE_CFG, MINI_QUADROTOR_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG  # isort: skip

from aerial_lab.actuators.rotorgroup import RotorGroup  # isort: skip
from aerial_lab.utility.noisemodel import NoiseModel  # isort: skip


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
                    self._create_debug_vis_ui_element("thrusts", self.env)


@configclass
class BeetleEnvCfg(DirectRLEnvCfg):
    # env
    sim_dt = 1 / 200.0
    decimation = 4
    episode_length_s = 10.0
    max_curricular_steps = 8000.0
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
    observation_space = 9 + 6 + 3 + 6 + 3 + gimbal_num + action_space
    thrust_to_torque_ratio = 0.0165
    rotor_direction = [1, -1, -1, 1]
    # rotor_direction = [1, 1, 1, 1]
    contact_force_threshold = 0.1

    thrust_limit = 16.0  # N
    # gimbal_limit = math.pi * 3 / 4  # rad
    gimbal_limit = math.pi * 0.5  # rad
    state_space = observation_space + action_space

    # custom parameters/scales
    debug_vis = True

    ui_window_class_type = PoseTrackingEnvWindow

    class normalization:
        class obs_scales:
            ang_vel = 0.2
            lin_vel = 1.0

    clip_observations = 100.0
    clip_actions = 100.0
    # class control:
    gimbal_action_scale = 0.25
    thrust_action_scale = 1.25

    # reward scales
    lin_vel_reward_scale = -0.05
    lin_vel_th = 3.0
    ang_vel_reward_scale = -0.01  # -0.01
    ang_vel_th = 6.0
    # reach_lin_vel_reward_scale = -0.05
    # reach_ang_vel_reward_scale = -0.1
    thrust_power_reward_scale = -2.0e-6  # -1.0e-4
    # goal_orientation_reward_scale = -0.001
    distance_to_goal_reward_scale = 5.0
    # quat_error_to_goal_reward_scale = -6.0
    # angular_to_goal_reward_scale = 3.0
    angular_error_to_goal_reward_scale = -5.0

    # # # # Noise Configuration
    noiseCfg = {
        # "root_pos": {
        #     "type": "uniform",
        #     "dim": 3,
        #     "mean": 0.0,
        #     "std": 0.02,
        #     "clip": 0.3,
        # },
        # "lin_vel": {
        #     "type": "uniform",
        #     "dim": 3,
        #     "mean": 0.0,
        #     "std": 0.1,
        #     "clip": 0.3,
        # },
        # "ang_vel": {
        #     "type": "uniform",
        #     "dim": 3,
        #     "mean": 0.0,
        #     "std": 0.3,
        #     "clip": 0.3,
        # },
        # "gravity": {
        #     "type": "uniform",
        #     "dim": 3,
        #     "mean": 0.0,
        #     "std": 0.05,
        #     "clip": 0.1,
        # },
        # "dof_pos": {
        #     "type": "uniform",
        #     "dim": gimbal_num,
        #     "mean": 0.0,
        #     "std": 0.02,
        #     "clip": 0.1,
        # },
    }

    rotorCfg = {
        "rotor_num": 4,
        "dt": sim_dt,
        "mode": "foc",
        "thrust_coeff": 1.0,
        "torque_coeff": thrust_to_torque_ratio,
        "max_vel": 200.0,
        "max_foc": thrust_limit,
        "vel_wn": 1.0,
        "vel_zeta": 0.8,
        "foc_wn": 1.0,
        "foc_zeta": 0.8,
    }

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
        dt=sim_dt,
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
        prim_path="/World/envs/env_.*/Robot/base_link",  # Bind to the robot root link
        history_length=1,
        update_period=0,  # Update every physics step
        track_air_time=True,
        # track_contact_points=True,
        debug_vis=True,
        # filter_prim_paths_expr=["/World/ground"],  # Only track contacts with the ground
        # filter_prim_paths_expr=[terrain.prim_path],  # Only track contacts with the ground
    )

    thrust_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rotor_parent.*",  # Bind to all rotor parent links
        history_length=1,
        update_period=0,
        track_air_time=True,
        track_pose=True,
        debug_vis=False,
    )

    # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#inertia-measurement-unit
    imu_sensor: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
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
        # Basic cfgs
        self.obsScales = cfg.normalization.obs_scales
        self.noiseModel = NoiseModel(cfg.noiseCfg, device=self.device, num_envs=self.num_envs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._last_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._action_thrust_force = torch.zeros(self.num_envs, self.cfg.rotor_num, device=self.device)
        self._action_gimbal_pos = torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device)
        self._target_thrust_force = torch.zeros(self.num_envs, self.cfg.rotor_num, 3, device=self.device)
        self._target_rotor_torque = torch.zeros(self.num_envs, self.cfg.rotor_num, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_zyx_euler_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._position_error = torch.zeros(self.num_envs, 3, device=self.device)
        self._angle_error = torch.zeros(self.num_envs, 3, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "reach_ang_vel",
                "thrust_power",
                # "goal_orientation",
                # "quat_to_goal",
                # "angular_to_goal",
                "angular_error_to_goal",
                "died",
                "reach_goal",
            ]
        }

        self._body_id = self._robot.find_bodies("root")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        self._thrust_ids = self._robot.find_bodies("rotor_parent.*")
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

        self._rotors = RotorGroup(
            cfg=cfg.rotorCfg,
            devices=self.device,
            num_envs=self.num_envs,
            rotor_ids=self._thrust_ids[0],
            rotor_names=self._thrust_ids[1],
            rotor_directions=self.cfg.rotor_direction,
        )

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
        self._thrusts_sensor = ContactSensor(self.cfg.thrust_sensor)
        self.scene.sensors["thrust_sensor"] = self._thrusts_sensor
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
        clip_actions = self.cfg.clip_actions
        self._actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)  # TODO: check action limits
        self._action_gimbal_pos = (
            self._actions[:, : self.cfg.gimbal_num] * self.cfg.gimbal_action_scale
        )  # scale to [-1.57, 1.57] rad
        self._action_thrust_force = (
            self._actions[:, self.cfg.gimbal_num :] * self.cfg.thrust_action_scale
        )  # shape: (N, 4)
        # print("Gimbal Positions: ", self._action_gimbal_pos[0])
        # print("Thrust Forces: ", action_thrust_force[0])
        # self._action_gimbal_pos[:, 0] = -math.pi / 2
        # self._action_gimbal_pos[:, 2] = math.pi / 2
        # action_thrust_force[:, 0] = 1.0
        # action_thrust_force[:, 1] = 0.0
        # action_thrust_force[:, 2] = 1.0
        # action_thrust_force[:, 3] = 0.0
        self._target_thrust_force, self._target_rotor_torque = self._rotors.forward(self._action_thrust_force)
        # #################################################
        # self._target_rotor_torque = (
        #     self.cfg.thrust_to_torque_ratio
        #     * self._target_thrust_force
        #     * torch.tensor(self.cfg.rotor_direction, device=self.device)
        # )
        # self._target_thrust_force[:, :] = 0
        # print("Rotor velocities: ", self._rotor_vel[0])
        # print("Rotor velocities: ", self._robot.data.joint_vel[:, self._rotor_ids[0]][0])
        # print("Rotor positions: ", self._robot.data.joint_pos[:, self._rotor_ids[0]][0])
        #################################################
        # print("Action debug info:")
        # print(" Gimbal pos targets: ", self._action_gimbal_pos[0])
        # print(" Rotor thrust targets: ", self._target_thrust_force[0])
        # print(" Rotor torque targets: ", self._target_rotor_torque[0])
        #################################################

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._action_gimbal_pos, self._gimbal_ids[0])

        self._robot.set_external_force_and_torque(
            forces=self._target_thrust_force,
            torques=self._target_rotor_torque,
            body_ids=self._thrust_ids[0],
        )

    def _get_observations(self) -> dict:
        goal_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        # desired_ang_bias = self._angle_error
        root_rot_mat = matrix_from_quat(self._robot.data.root_quat_w)
        root_rot_vec = root_rot_mat[:, :2, :].reshape(self.num_envs, 6)

        goal_rot_mat = matrix_from_quat(self._desired_quat_w)
        goal_rot_vec = goal_rot_mat[:, :2, :].reshape(self.num_envs, 6)

        angular_error = self._angle_error

        obs = torch.cat(
            (
                self._robot.data.root_lin_vel_b * self.obsScales.lin_vel,
                self._robot.data.root_ang_vel_b * self.obsScales.ang_vel,
                self._robot.data.projected_gravity_b,
                goal_pos_b,
                angular_error,
                self._robot.data.joint_pos[:, self._gimbal_ids[0]],
                root_rot_vec,
                goal_rot_vec,
                self._last_actions,
            ),
            dim=-1,
        )
        if "lin_vel" in self.noiseModel.params:
            obs[:, 0:3] = self.noiseModel.apply(obs[:, 0:3], "lin_vel")
        if "ang_vel" in self.noiseModel.params:
            obs[:, 3:6] = self.noiseModel.apply(obs[:, 3:6], "ang_vel")
        if "gravity" in self.noiseModel.params:
            obs[:, 6:9] = self.noiseModel.apply(obs[:, 6:9], "gravity")
        if "root_pos" in self.noiseModel.params:
            obs[:, 9:12] = self.noiseModel.apply(obs[:, 9:12], "root_pos")
        if "root_ang" in self.noiseModel.params:
            obs[:, 15:18] = self.noiseModel.apply(obs[:, 15:18], "root_ang")
        if "dof_pos" in self.noiseModel.params:
            obs[:, 18 : 18 + self.cfg.gimbal_num] = self.noiseModel.apply(
                obs[:, :, 18 : 18 + self.cfg.gimbal_num], "dof_pos"
            )
        clip_obs = self.cfg.clip_observations
        obs = torch.clamp(obs, -clip_obs, clip_obs)
        states = self._get_states()
        observations = {"policy": obs, "critic": states}
        return observations

    def _get_states(self) -> torch.Tensor:
        goal_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        # desired_ang_bias = self._angle_error
        root_rot_mat = matrix_from_quat(self._robot.data.root_quat_w)
        root_rot_vec = root_rot_mat[:, :2, :].reshape(self.num_envs, 6)

        goal_rot_mat = matrix_from_quat(self._desired_quat_w)
        goal_rot_vec = goal_rot_mat[:, :2, :].reshape(self.num_envs, 6)

        angular_error = self._angle_error

        states = torch.cat(
            (
                self._robot.data.root_lin_vel_b * self.obsScales.lin_vel,
                self._robot.data.root_ang_vel_b * self.obsScales.ang_vel,
                self._robot.data.projected_gravity_b,
                goal_pos_b,
                angular_error,
                self._robot.data.joint_pos[:, self._gimbal_ids[0]],
                root_rot_vec,
                goal_rot_vec,
                self._last_actions,
            ),
            dim=-1,
        )
        return states

    def _get_rewards(self) -> torch.Tensor:
        pos_err, rot_err = compute_pose_error(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
            self._desired_quat_w,
        )
        self._position_error = pos_err
        self._angle_error = rot_err

        lin_vel_norm = torch.linalg.norm(self._robot.data.root_lin_vel_b, dim=1)
        lin_vel_over = torch.clamp(lin_vel_norm - self.cfg.lin_vel_th, min=0.0)
        lin_vel = torch.square(lin_vel_over)

        ang_vel_norm = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=1)
        ang_vel_over = torch.clamp(ang_vel_norm - self.cfg.ang_vel_th, min=0.0)
        ang_vel = torch.square(ang_vel_over)
        # distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal = torch.linalg.norm(self._position_error, dim=1)
        # distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        distance_to_goal_mapped = torch.exp(-2 * distance_to_goal)
        # distance_to_goal_weight = torch.exp(-torch.square(3.0 * distance_to_goal))
        # reach_lin_vel = torch.linalg.norm(self._robot.data.root_lin_vel_b, dim=-1) * distance_to_goal_weight
        # reach_ang_vel = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=-1) * distance_to_goal_weight
        thrust_power = torch.sum(torch.square(self._action_thrust_force), dim=1)

        # goal_orientation = torch.linalg.norm(self._robot.data.projected_gravity_b, dim=-1) * distance_to_goal_weight
        # thrust_power = torch.sum(torch.square(self._target_thrust_force), dim=1)
        # desired_ang_bias = quat_error_magnitude(
        #     self._desired_quat_w,
        #     self._robot.data.root_quat_w
        # )
        # quat_to_goal = torch.linalg.norm(self._angle_error, dim=1) * distance_to_goal_weight
        # quat_to_goal = torch.sum(torch.square(self._angle_error), dim=1) * distance_to_goal_weight
        angular_to_goal = torch.linalg.norm(self._angle_error, dim=1)
        # angular_to_goal_mapped = torch.tanh(angular_to_goal)
        # angular_to_goal_mapped = 1 - torch.exp(-2 * angular_to_goal)
        angular_to_goal_mapped = 1 - torch.exp(-angular_to_goal / 0.8)
        # angular_error_to_goal_mapped = 1 - torch.exp(-angular_to_goal)

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            # "reach_lin_vel": reach_lin_vel * self.cfg.reach_lin_vel_reward_scale * self.step_dt,
            # "reach_ang_vel": reach_ang_vel * self.cfg.reach_ang_vel_reward_scale * self.step_dt,
            "thrust_power": thrust_power * self.cfg.thrust_power_reward_scale * self.step_dt,
            # "goal_orientation": goal_orientation * self.cfg.goal_orientation_reward_scale * self.step_dt,
            # "quat_to_goal": quat_to_goal * self.cfg.quat_error_to_goal_reward_scale * self.step_dt,
            # "angular_to_goal": angular_to_goal_mapped * self.cfg.angular_to_goal_reward_scale * self.step_dt,
            "angular_error_to_goal": (
                angular_to_goal_mapped * self.cfg.angular_error_to_goal_reward_scale * self.step_dt
            ),
        }
        total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Early termination penalty
        die_reward = self.reset_terminated.to(torch.float32) * -30.0
        total_reward += die_reward
        rewards["died"] = die_reward

        # # timeout reward
        # time_out_reward = self.reset_time_outs.to(torch.float32) * 20.0
        # total_reward += time_out_reward
        # rewards["time_out"] = time_out_reward

        # timeout reward with reaching goal
        ang_vel_norm = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=-1)
        lin_vel_norm = torch.linalg.norm(self._robot.data.root_lin_vel_b, dim=-1)

        # reach_goal_precise = torch.logical_and(
        #     torch.logical_and(ang_vel_norm < 0.01, lin_vel_norm < 0.05),
        #     torch.logical_and(distance_to_goal < 0.01, angular_to_goal < 0.05),
        # )
        # reach_goal = torch.logical_and(
        #     torch.logical_and(ang_vel_norm < 0.02, lin_vel_norm < 0.01),
        #     torch.logical_and(angular_to_goal < 0.1, distance_to_goal < 0.05),
        # )
        # reach_goal_rough = torch.logical_and(
        #     torch.logical_and(ang_vel_norm < 0.05, lin_vel_norm < 0.2),
        #     torch.logical_and(distance_to_goal < 0.1, angular_to_goal < 0.5),
        # )
        ANG_VEL_TH = 0.02
        LIN_VEL_TH = 0.01
        POS_TH = 0.05
        ANG_TH = 0.1
        reach_goal = torch.logical_and(
            torch.logical_and(ang_vel_norm < ANG_VEL_TH, lin_vel_norm < LIN_VEL_TH),
            torch.logical_and(angular_to_goal < ANG_TH, distance_to_goal < POS_TH),
        )
        reach_goal_reward = self.reset_time_outs.to(torch.float32) * reach_goal.to(torch.float32) * 10.0
        reach_goal_reward = (
            torch.exp((POS_TH - distance_to_goal) / POS_TH)
            * self.reset_time_outs.to(torch.float32)
            * reach_goal.to(torch.float32)
            * 20.0
        )
        reach_goal_reward = (
            torch.exp((ANG_TH - angular_to_goal) / ANG_TH)
            * self.reset_time_outs.to(torch.float32)
            * reach_goal.to(torch.float32)
            * 20.0
        )
        reach_goal_reward = (
            torch.exp((LIN_VEL_TH - lin_vel_norm) / LIN_VEL_TH)
            * self.reset_time_outs.to(torch.float32)
            * reach_goal.to(torch.float32)
            * 20.0
        )
        reach_goal_reward = (
            torch.exp((ANG_VEL_TH - ang_vel_norm) / ANG_VEL_TH)
            * self.reset_time_outs.to(torch.float32)
            * reach_goal.to(torch.float32)
            * 20.0
        )
        reach_goal_reward = torch.clamp(reach_goal_reward, min=0.0, max=40.0)
        total_reward += reach_goal_reward
        rewards["reach_goal"] = reach_goal_reward
        # reach_goal_precise_reward = self.reset_time_outs.to(torch.float32) * reach_goal_precise.to(torch.float32) * 30.0
        # reach_goal_rough_reward = self.reset_time_outs.to(torch.float32) * reach_goal_rough.to(torch.float32) * 30.0
        # total_reward += reach_goal_reward + reach_goal_precise_reward + reach_goal_rough_reward
        # rewards["reach_goal"] = reach_goal_reward + reach_goal_precise_reward + reach_goal_rough_reward

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        # total_reward = torch.zeros(self.num_envs, device=self.device)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # import ipdb; ipdb.set_trace()
        # died = torch.linalg.norm(self._contact_sensor.data.force_matrix_w.squeeze(1).squeeze(1), dim=-1) > 0.1
        crash = (
            torch.linalg.norm(self._contact_sensor.data.net_forces_w.squeeze(1), dim=-1)
            > self.cfg.contact_force_threshold
        )
        drift = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 5.0)
        died = torch.logical_or(crash, drift)
        # died = crash
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
        final_angle_error = torch.linalg.norm(self._angle_error[env_ids], dim=1).mean()
        thrust_average = torch.mean(self._action_thrust_force, dim=1).mean()

        goal_lin_vel_avg = torch.mean(torch.abs(self._robot.data.root_lin_vel_b[env_ids]), dim=0)
        goal_ang_vel_avg = torch.mean(torch.abs(self._robot.data.root_ang_vel_b[env_ids]), dim=0)
        # print("Resetting envs: ", env_ids)

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
        extras["Metrics/final_angular_to_goal"] = final_angle_error.item()
        extras["Metrics/avg_thrusts"] = thrust_average.item()
        extras["Metrics/avg_goal_lin_vel_x"] = goal_lin_vel_avg[0].item()
        extras["Metrics/avg_goal_lin_vel_y"] = goal_lin_vel_avg[1].item()
        extras["Metrics/avg_goal_lin_vel_z"] = goal_lin_vel_avg[2].item()
        extras["Metrics/avg_goal_ang_vel_x"] = goal_ang_vel_avg[0].item()
        extras["Metrics/avg_goal_ang_vel_y"] = goal_ang_vel_avg[1].item()
        extras["Metrics/avg_goal_ang_vel_z"] = goal_ang_vel_avg[2].item()
        extras["Metrics/sample_rate"] = self.common_step_counter / self.cfg.max_curricular_steps
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)

        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._action_gimbal_pos[env_ids] = 0.0
        self._action_thrust_force[env_ids] = 0.0
        # self._target_thrust_force[env_ids] = 0.0
        # self._target_rotor_torque[env_ids] = 0.0

        self._position_error[env_ids] = 0.0
        self._angle_error[env_ids] = 0.0

        # self._gimbal_pos[env_ids] = 0.0
        # self._gimbal_vel[env_ids] = 0.0
        # self._rotor_vel[env_ids] = 0.0
        # self._rotor_force[env_ids] = 0.0
        # Sample new commands

        # self._desired_zyx_euler_w[env_ids] = torch.empty_like(self._desired_zyx_euler_w[env_ids]).uniform_(-math.pi, math.pi)
        # self._desired_quat_w[env_ids] = quat_from_euler_xyz(
        #     self._desired_zyx_euler_w[env_ids][:, 0],
        #     self._desired_zyx_euler_w[env_ids][:, 1],
        #     self._desired_zyx_euler_w[env_ids][:, 2],
        # )

        # quat_sample_rate = max(self._sim_step_counter / self.max_episode_length * 2, 0.6)  # start from 0.3, reach 0.8
        # pos_sample_rate = max(self._sim_step_counter / self.max_episode_length * 2 , 0.2)  # start from 0.1, reach 0.6

        # quat_mask = torch.rand_like(env_ids, dtype=torch.float32, device=self.device) > (1.6 - quat_sample_rate)
        # sample_quat_env_ids = env_ids[quat_mask]
        # unsampled_quat_env_ids = env_ids[~quat_mask]
        # has_sampled_quat = quat_mask.any()
        # has_unsampled_quat = (~quat_mask).any()

        # if has_sampled_quat:
        #     self._desired_zyx_euler_w[sample_quat_env_ids, :2] = torch.empty_like(self._desired_zyx_euler_w[sample_quat_env_ids, :2]).uniform_(-math.pi * 0.3, math.pi * 0.3)
        #     self._desired_zyx_euler_w[sample_quat_env_ids, 2] = torch.empty_like(self._desired_zyx_euler_w[sample_quat_env_ids, 2]).uniform_(-math.pi, math.pi)
        #     self._desired_quat_w[sample_quat_env_ids] = quat_from_euler_xyz(
        #         self._desired_zyx_euler_w[sample_quat_env_ids][:, 0],
        #         self._desired_zyx_euler_w[sample_quat_env_ids][:, 1],
        #         self._desired_zyx_euler_w[sample_quat_env_ids][:, 2],
        #     )

        # pos_mask = torch.rand_like(env_ids, dtype=torch.float32, device=self.device) > -0.1
        # sample_pos_env_ids = env_ids[pos_mask]
        # unsampled_pos_env_ids = env_ids[~pos_mask]
        # has_sampled_pos = pos_mask.any()
        # has_unsampled_pos = (~pos_mask).any()

        # if has_sampled_pos:
        #     self._desired_pos_w[sample_pos_env_ids, :2] = torch.empty_like(
        #         self._desired_pos_w[sample_pos_env_ids, :2]
        #     ).uniform_(-3.0, 3.0)
        #     self._desired_pos_w[sample_pos_env_ids, :2] += self._terrain.env_origins[sample_pos_env_ids, :2]
        #     self._desired_pos_w[sample_pos_env_ids, 2] = torch.empty_like(
        #         self._desired_pos_w[sample_pos_env_ids, 2]
        #     ).uniform_(0.5, 2.5)
        # quat_sample_rate = self._sim_step_counter / self.max_episode_length * 2  # start from 0.3, reach 0.8
        # pos_sample_rate = self._sim_step_counter / self.max_episode_length * 2  # start from 0.1, reach 0.6
        quat_sample_rate = self.common_step_counter / self.cfg.max_curricular_steps * 2  # start from 0.3, reach 0.8
        pos_sample_rate = self.common_step_counter / self.cfg.max_curricular_steps * 2  # start from 0.1, reach 0.6
        quat_sample_rate = max(quat_sample_rate - 0.20, 0.0)
        pos_sample_rate = max(pos_sample_rate - 0.06, 0.0)
        ang_range = min(math.pi * 0.5 * quat_sample_rate, math.pi * 0.45)
        pos_range = min(5.0 * pos_sample_rate, 5.0)
        pos_range_z = min(1.0 * pos_sample_rate, 1.0)

        # # # # Debug
        ang_range = 0.0
        pos_range = 0.0
        pos_range_z = 0.0

        # Euler ZYX[a,b,c] = RPY[c,b,a]
        # self._desired_zyx_euler_w[env_ids, 0] = torch.empty_like(self._desired_zyx_euler_w[env_ids, 0]).uniform_(
        #     -math.pi, math.pi
        # )
        self._desired_zyx_euler_w[env_ids, 0] = torch.empty_like(self._desired_zyx_euler_w[env_ids, 0]).uniform_(
            -0.0, 0.0
        )
        self._desired_zyx_euler_w[env_ids, 1] = torch.empty_like(self._desired_zyx_euler_w[env_ids, 1]).uniform_(
            -ang_range, ang_range
        )
        self._desired_zyx_euler_w[env_ids, 2] = torch.empty_like(self._desired_zyx_euler_w[env_ids, 2]).uniform_(
            -ang_range, ang_range
        )
        desired_matrix = matrix_from_euler(self._desired_zyx_euler_w[env_ids], "ZYX")
        self._desired_quat_w[env_ids] = quat_from_matrix(desired_matrix)
        # self._desired_quat_w[env_ids] = quat_from_euler_xyz(
        #     self._desired_zyx_euler_w[env_ids][:, 0],
        #     self._desired_zyx_euler_w[env_ids][:, 1],
        #     self._desired_zyx_euler_w[env_ids][:, 2],
        # )
        self._desired_pos_w[env_ids, :2] = torch.empty_like(self._desired_pos_w[env_ids, :2]).uniform_(
            -pos_range, pos_range
        )
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.empty_like(self._desired_pos_w[env_ids, 2]).uniform_(
            max(0.5, 1 - pos_range_z), min(1 + pos_range_z, 3.0)
        )

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        init_rp = torch.empty_like(self._desired_zyx_euler_w[env_ids, :2]).uniform_(-math.pi * 0.5, math.pi * 0.5)
        init_y = torch.empty_like(self._desired_zyx_euler_w[env_ids, 2]).uniform_(-math.pi, math.pi)
        init_quat = quat_from_euler_xyz(
            init_rp[:, 0],
            init_rp[:, 1],
            init_y,
        )
        default_root_state[:, 3:7] = init_quat
        default_root_state[:, 7:10] = torch.randn_like(default_root_state[:, 7:10]) * self.obsScales.lin_vel
        default_root_state[:, 10:13] = torch.randn_like(default_root_state[:, 10:13]) * self.obsScales.ang_vel
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # if has_unsampled_pos:
        #     self._desired_pos_w[unsampled_pos_env_ids] = (
        #         self._robot.data.default_root_state[unsampled_pos_env_ids, :3]
        #         + self._terrain.env_origins[unsampled_pos_env_ids]
        #     )
        # if has_unsampled_quat:
        #     self._desired_quat_w[unsampled_quat_env_ids] = self._robot.data.default_root_state[
        #         unsampled_quat_env_ids, 3:7
        #     ]

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = VisualizationMarkersCfg(
                    markers={
                        "frame": sim_utils.UsdFileCfg(
                            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                            scale=(0.25, 0.25, 0.25),
                        ),
                    }
                )
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_pose"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "thrusts_visualizer"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.02, 0.02, 0.02)
                marker_cfg.prim_path = "/Visuals/Command/thrusts_visualization"
                self.thrusts_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
            self.thrusts_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "thrusts_visualizer"):
                self.thrusts_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w, self._desired_quat_w)

        thrusts = self._target_thrust_force
        thrusts_pos = self._thrusts_sensor.data.pos_w
        thrusts_quat = self._thrusts_sensor.data.quat_w

        if thrusts_pos is None:
            print("Thrust positions not available yet for debug visualization.")
            return

        thrusts = thrusts.reshape(-1, 3)
        thrusts_pos = thrusts_pos.reshape(-1, 3)
        thrusts_quat = thrusts_quat.reshape(-1, 4)
        local_offset = torch.zeros_like(thrusts_pos)
        local_offset[:, 2] = 0.03
        thrusts_pos += quat_apply(thrusts_quat, local_offset)

        thrusts_mag = torch.linalg.norm(thrusts, dim=-1)

        unit_dir = normalize(thrusts)

        orientations = quat_from_angle_axis(torch.zeros(thrusts_mag.shape[0], device=self.device), unit_dir)
        x2z_quat = quat_from_euler_xyz(
            torch.zeros(unit_dir.shape[0], device=self.device, dtype=unit_dir.dtype),
            -math.pi / 2 * torch.ones(unit_dir.shape[0], device=self.device, dtype=unit_dir.dtype),
            torch.zeros(unit_dir.shape[0], device=self.device, dtype=unit_dir.dtype),
        )  # (N,4)
        orientations = quat_mul(thrusts_quat, orientations)
        orientations = quat_mul(orientations, x2z_quat)

        scales = torch.ones_like(thrusts)
        scales[:, 0] = thrusts_mag * 0.5
        self.thrusts_visualizer.visualize(translations=thrusts_pos, orientations=orientations, scales=scales)
