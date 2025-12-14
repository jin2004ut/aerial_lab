# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import gymnasium as gym
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from aerial_lab.utility.utilitymath import (
    sampleCenterQuatwithTilt,
    sampleUniformQuatwithTilt,
)
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
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
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
)

from aerial_lab.assets.aerialrobot import BEETLE_CFG, BEETLE_OMNI_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG  # isort: skip

from aerial_lab.actuators.rotorgroup import RotorGroup  # isort: skip
from aerial_lab.utility.noisemodel import NoiseModel  # isort: skip
from traitlets import default  # isort: skip

PUSH_LIN_VEL = 1.0  # m/s
PUSH_ANG_VEL = 1.0  # rad/s


class PoseTrackingEnvWindow(BaseEnvWindow):
    """Window manager for the Beetle environment."""

    def __init__(self, env: BeetleOmniEnv, window_name: str = "IsaacLab"):
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
class EventCfg:
    """Configuration for randomization."""

    # Always got error: TypeError: randomize_rigid_body_mass.__init__() got an unexpected keyword argument 'asset_cfg'
    scale_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.90, 1.10),
            "operation": "scale",
        },
    )
    # # # # So we define a new function in events.py
    # scale_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass_hand,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "mass_distribution_params": (0.1, 2.0),
    #         "operation": "scale",
    #     },
    # )

    noise_com_pos = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 15.0),
        params={
            "velocity_range": {
                "x": (-PUSH_LIN_VEL, PUSH_LIN_VEL),
                "y": (-PUSH_LIN_VEL, PUSH_LIN_VEL),
                "z": (-PUSH_LIN_VEL, PUSH_LIN_VEL),
                "roll": (-PUSH_ANG_VEL, PUSH_ANG_VEL),
                "pitch": (-PUSH_ANG_VEL, PUSH_ANG_VEL),
                "yaw": (-PUSH_ANG_VEL, PUSH_ANG_VEL),
            }
        },
    )


@configclass
class BeetleOmniEnvCfg(DirectRLEnvCfg):
    # env
    sim_dt = 1 / 400.0
    decimation = 4
    num_steps_per_env = 48
    play_mode = False
    evaluate_mode = False
    add_noise = True
    add_randomization = True
    episode_length_s = 15.0
    max_curricular_steps = 8000.0 * num_steps_per_env  # num_steps_per_env * max_iterations
    ang_curricular_steps = 6000.0 * num_steps_per_env  # num_steps_per_env * max_iterations
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
    observation_space = 9 + 6 + 3 + 6 + gimbal_num + action_space
    obs_vel_delay_steps = 4
    thrust_to_torque_ratio = -0.0165  # -0.0165
    rotor_direction = [1, -1, 1, -1]  # beetle_hyper, joint urdf configuration
    contact_force_threshold = 0.1

    state_space = observation_space

    # custom parameters/scales
    debug_vis = True

    ui_window_class_type = PoseTrackingEnvWindow

    class randomization:
        lin_vel = 1.0
        ang_vel = 1.0
        dof_pos = math.pi / 180.0 * 3.0  # 5 degrees
        body_ang = math.pi  # body tilt angle for init orientation sampling

    class normalization:
        class obs_scales:
            ang_vel = 0.2
            lin_vel = 1.0

    class control:
        body_ang = math.pi  # body tilt angle for desired orientation sampling
        ref_lin_vel = 1.5  # m/s
        ang_reset_min_episode_s = 8  # reset orientation target after N episodes
        ang_reset_max_episode_s = 10  # reset orientation target after N episodes
        ang_reset_rate = 0.0  # reset orientation target after
        clip_observations = 100.0
        clip_actions = 100.0
        # class control:
        gimbal_action_scale = 0.25
        thrust_action_scale = 1.25

        thrust_limit = 20.0  # N
        hover_thrust = 6.0  # N
        # gimbal_limit = math.pi * 3 / 4  # rad
        default_gimbal_pos = {
            "gimbal": 0.0,
        }
        limit_gimbal_pos = {
            "gimbal": math.pi * 1.25,
        }

    # # # # reward scales # # # # # # # #
    lin_vel_reward_scale = -0.02
    lin_vel_th = 3.0
    ang_vel_reward_scale = -0.02  # -0.01
    ang_vel_th = 6.0
    lin_vel_static_reward_scale = -0.0
    ang_vel_static_reward_scale = -0.0
    # reach_lin_vel_reward_scale = -0.05
    # reach_ang_vel_reward_scale = -0.1
    thrust_power_reward_scale = -1.0e-5  # -1.0e-4
    # goal_orientation_reward_scale = -0.001
    distance_to_goal_reward_scale = -1.0
    # quat_error_to_goal_reward_scale = -6.0
    # angular_to_goal_reward_scale = 3.0
    # angular_error_to_goal_reward_scale = -5.0
    # angular_error_to_goal_reward_scale = 3.0
    angular_to_goal_reward_scale = 3.0

    died_reward_scale = -1.0
    reach_goal_reward_timeout_scale = 0.0
    reach_goal_reward_scale = 0.5
    reach_pos_reward_scale = 0.5

    # smoothing reward scales
    gimbal_action_rate_reward_scale = -1.0e-4  # -1.0e-3
    thrust_action_rate_reward_scale = -1.0e-5  # -1.0e-4
    gimbal_acc_reward_scale = -1.5e-7  # -1.5e-7
    gimbal_limit_reward_scale = -0.01  # -0.01
    gimbal_limit_scale = math.pi * 1.25
    gimbal_vel_reward_scale = 0.0  # -1.0e-5
    thrust_limit_reward_scale = -0.01  # -0.01
    thrust_limit = control.thrust_limit * 0.9
    thrust_uneven_reward_scale = -5.0e-5  # -1.0e-9

    # # # # Noise Configuration
    noiseCfg = {
        "root_pos": {
            "type": "uniform",
            "dim": 3,
            "mean": 0.005,
            "std": 0.005,
            "clip": 0.3,
        },
        "root_quat": {
            "type": "uniform",
            "dim": 3,
            "mean": math.pi * (0.5 / 180.0),
            "std": math.pi * (1.0 / 180.0),
            "clip": 0.3,
        },
        "lin_vel": {
            "type": "uniform",
            "dim": 3,
            "mean": 0.005,
            "std": 0.005,
            "clip": 3.0,
        },
        "ang_vel": {
            "type": "uniform",
            "dim": 3,
            "mean": 0.005,
            "std": 0.005,
            "clip": 3.0,
        },
        # "gravity": {
        #     "type": "uniform",
        #     "dim": 3,
        #     "mean": 0.0,
        #     "std": 0.05,
        #     "clip": 0.1,
        # },
        "dof_pos": {
            "type": "uniform",
            "dim": gimbal_num,
            "mean": 0.0,
            "std": math.pi * (1.0 / 180.0),
            "clip": 0.1,
        },
    }

    rotorCfg = {
        "rotor_num": 4,
        "dt": sim_dt,
        "mode": "foc",
        "thrust_coeff": 1.0,
        "torque_coeff": thrust_to_torque_ratio,
        "randomize_ratio": 0.0,
        "max_vel": 1000.0,
        "max_foc": control.thrust_limit,
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
    robot_cfg: ArticulationCfg = BEETLE_OMNI_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",  # Bind to the robot root link
        history_length=4,
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
    imu_link_name = "fc"
    imu_sensor: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/fc",
        update_period=0,
        history_length=1,
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),  # w, x, y, z
        ),
        debug_vis=True,
    )

    if add_randomization:
        events: EventCfg = EventCfg()


class BeetleOmniEnv(DirectRLEnv):
    cfg: BeetleOmniEnvCfg

    def __init__(self, cfg: BeetleOmniEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        try:
            print(f"BeetleEnv init: cfg class = {cfg.__class__.__name__}")
            print(f"cfg.robot_cfg.asset_path = {getattr(cfg.robot_cfg.spawn, 'asset_path', None)}")
        except Exception as e:
            print("BeetleEnv init: failed to print cfg.robot_cfg:", e)
        # Basic cfgs
        self.obsScales = cfg.normalization.obs_scales
        self.noiseModel = NoiseModel(cfg.noiseCfg, device=self.device, num_envs=self.num_envs)
        self.randomCfg = cfg.randomization

        self._quat_sample_rate = 0.0
        self._pos_sample_rate = 0.0
        self._reach_goal_count = torch.zeros(self.num_envs, device=self.device)
        self._reach_goal = torch.zeros(self.num_envs, device=self.device)
        self._reach_goal_state = torch.zeros(self.num_envs, device=self.device)
        self._success_rate = torch.zeros(0, device=self.device)
        self._success_window_size = 100
        self._success_window = torch.zeros(0, device=self.device)
        self._desired_time_to_goal_s = torch.full(
            (self.num_envs,), cfg.episode_length_s, device=self.device
        )  # desired time to reach the goal position

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
        self._init_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._init_zyx_euler_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._init_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._position_error = torch.zeros(self.num_envs, 3, device=self.device)
        self._angle_error = torch.zeros(self.num_envs, 3, device=self.device)

        self._gimbal_pos = torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device)
        self._gimbal_vel = torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device)
        self._gimbal_vel_last = torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "thrust_power",
                "angular_to_goal",
                "died",
                "reach_goal",
                "reach_pos",
                "reach_goal_timeout",
                "gimbal_action_rate",
                "thrust_action_rate",
                "gimbal_acc",
                "gimbal_vel",
                "gimbal_limit",
                "thrust_limit",
                "thrust_uneven",
            ]
        }

        self._body_id = self._robot.find_bodies("base_link")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_default_com = self._robot.data.body_com_pose_b[0, self._body_id].clone().cpu().numpy()

        self._imu_id = self._robot.find_bodies(self.cfg.imu_link_name)[0]
        _body_pos_w = self._robot.data.body_pos_w[:, self._body_id]
        _body_quat_w = self._robot.data.body_quat_w[:, self._body_id]
        _imu_pos_w = self._robot.data.body_pos_w[:, self._imu_id]
        _imu_quat_w = self._robot.data.body_quat_w[:, self._imu_id]
        self._imu_pos_body, self._imu_quat_body = subtract_frame_transforms(
            _body_pos_w,
            _body_quat_w,
            _imu_pos_w,
            _imu_quat_w,
        )

        self.cfg.imu_sensor.offset.pos = self._imu_pos_body[0, 0, :].cpu().numpy().tolist()
        self.cfg.imu_sensor.offset.rot = self._imu_quat_body[0, 0, :].cpu().numpy().tolist()

        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        self._thrust_ids = self._robot.find_bodies("rotor_parent.*")
        self._gimbal_ids = self._robot.find_joints("gimbal.*")
        self._rotor_ids = self._robot.find_joints("rotor.*")

        self.ctrlCfg = self.cfg.control

        self._reset_ang_pose_mask = torch.empty(self.num_envs, dtype=torch.bool, device=self.device).fill_(False)
        self._reset_ang_episode_counts = (
            torch.empty(self.num_envs, device=self.device)
            .uniform_(
                self.cfg.control.ang_reset_min_episode_s / (self.cfg.sim.dt * self.cfg.decimation),
                self.cfg.control.ang_reset_max_episode_s / (self.cfg.sim.dt * self.cfg.decimation),
            )
            .floor_()
            .to(torch.int64)
        )

        gimbal_default_pos = torch.tensor(
            [
                self.ctrlCfg.default_gimbal_pos.get(
                    joint_name,
                    self.ctrlCfg.default_gimbal_pos.get("gimbal", 0.0),
                )
                for joint_name in self._gimbal_ids[1]
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self._initial_gimbal_pos = gimbal_default_pos.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._gimbal_default_pos = gimbal_default_pos.unsqueeze(0).expand(self.num_envs, -1).clone()
        if self.cfg.add_randomization:
            self._gimbal_default_pos = self._initial_gimbal_pos + torch.empty_like(self._initial_gimbal_pos).uniform_(
                -self.randomCfg.dof_pos, self.randomCfg.dof_pos
            )
        self._last_gimbal_pos = (
            torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device) - self._gimbal_default_pos
        )
        self._last_last_gimbal_pos = (
            torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device) - self._gimbal_default_pos
        )
        self._body_ang_vel_buffer = torch.zeros(self.num_envs, self.cfg.obs_vel_delay_steps, 3, device=self.device)
        self._body_lin_vel_buffer = torch.zeros(self.num_envs, self.cfg.obs_vel_delay_steps, 3, device=self.device)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("Beetle robot link list: ")
        # for link in self._robot.body_names:
        #     print(f" - {link}")
        # print("Beetle robot joint list: ")
        # for joint in self._robot.joint_names:
        #     print(f" - {joint}")
        print("Root Body ID: ", self._body_id, " Name: ", "base_link")
        print("Robot Mass: ", self._robot_mass)
        print("Gravity Magnitude: ", self._gravity_magnitude)
        print("IMU Body ID: ", self._imu_id, " Name: ", self.cfg.imu_link_name)
        print("IMU Offset Position (body frame): ", [round(x, 4) for x in self.cfg.imu_sensor.offset.pos])
        print("IMU Offset Rotation (body frame): ", [round(x, 4) for x in self.cfg.imu_sensor.offset.rot])
        print("IMU Position Offset Shape: ", self._imu_pos_body.shape)
        print("IMU Rotation Offset Shape: ", self._imu_quat_body.shape)
        print("Thrust Names: ", self._thrust_ids[1])
        print("Gimbal IDs: ", self._gimbal_ids[0])
        print("Gimbal Names: ", self._gimbal_ids[1])
        print("Gimbal Default Positions: ", self._gimbal_default_pos[0])
        print("Rotor IDs: ", self._rotor_ids[0])
        print("Rotor Names: ", self._rotor_ids[1])
        # masses = self._robot.root_physx_view.get_masses()[0]
        # body_ids = torch.arange(self._robot.num_bodies, dtype=torch.int)
        # print("Robot link masses: ")
        # for i, body_id in enumerate(body_ids):
        #     body_name = self._robot.body_names[body_id]
        #     body_mass = masses[i].item()
        #     print(f" - Body ID: {body_id}, Name: {body_name}, Mass: {body_mass:.4f} kg")
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
        clip_actions = self.ctrlCfg.clip_actions
        self._actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)  # TODO: check action limits
        self._action_gimbal_pos = (
            self._actions[:, : self.cfg.gimbal_num] * self.ctrlCfg.gimbal_action_scale + self._gimbal_default_pos
        )  # scale to [-1.57, 1.57] rad
        self._action_thrust_force = (
            self._actions[:, self.cfg.gimbal_num :] * self.ctrlCfg.thrust_action_scale + self.ctrlCfg.hover_thrust
        )  # shape: (N, 4)
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
        """Apply the action to the robot. Every dt step"""
        if self.cfg.evaluate_mode:
            if self.common_step_counter < 15:
                return
        if self.common_step_counter % 2 == 0:
            self._robot.set_joint_position_target(self._action_gimbal_pos, self._gimbal_ids[0])
            self._robot.set_external_force_and_torque(
                forces=self._target_thrust_force,
                torques=self._target_rotor_torque,
                body_ids=self._thrust_ids[0],
            )

        # env_ids = 1
        # if self.common_step_counter % 200 == 0:
        #     print(f"Debug Envent {env_ids}  Reset: ==========================================================")
        #     robot_mass = self._robot.root_physx_view.get_masses()[env_ids].sum()
        #     print(f"Robot [{env_ids}] mass: {robot_mass:.4f}, default mass: {self._robot_mass:.4f}")
        #     root_com = self._robot.data.body_com_pose_w[env_ids, self._body_id].clone().cpu().numpy()
        #     root_com_str = ", ".join(f"{x:.4f}" for x in root_com.flatten())
        #     default_com_str = ", ".join(f"{x:.4f}" for x in self._robot_default_com.flatten())
        #     print(f"Robot [{env_ids}] root com: [{root_com_str}], default com: [{default_com_str}]")

    def _get_observations(self) -> dict:
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        root_lin_vel_w = self._body_lin_vel_buffer[:, 0, :].clone()
        self._body_lin_vel_buffer = torch.roll(self._body_lin_vel_buffer, shifts=-1, dims=1)
        self._body_lin_vel_buffer[:, -1, :] = self._robot.data.root_lin_vel_w

        root_ang_vel_b = self._body_ang_vel_buffer[:, 0, :].clone()
        self._body_ang_vel_buffer = torch.roll(self._body_ang_vel_buffer, shifts=-1, dims=1)
        self._body_ang_vel_buffer[:, -1, :] = self._robot.data.root_ang_vel_b

        projected_gravity_w = self._robot.data.GRAVITY_VEC_W
        gimbal_pos = self._last_gimbal_pos.clone()
        self._last_last_gimbal_pos = self._last_gimbal_pos.clone()
        self._last_gimbal_pos = self._robot.data.joint_pos[:, self._gimbal_ids[0]] - self._gimbal_default_pos

        if self.cfg.add_noise:
            if "lin_vel" in self.noiseModel.params:
                root_lin_vel_w = self.noiseModel.apply(root_lin_vel_w, "lin_vel")
            if "ang_vel" in self.noiseModel.params:
                root_ang_vel_b = self.noiseModel.apply(root_ang_vel_b, "ang_vel")
            if "gravity" in self.noiseModel.params:
                noise = torch.empty(self.num_envs, 3, device=self.device).uniform_(
                    -self.cfg.noiseCfg["gravity"]["std"], self.cfg.noiseCfg["gravity"]["std"]
                )
                projected_gravity_w += noise
                projected_gravity_w = normalize(projected_gravity_w)
            if "root_pos" in self.noiseModel.params:
                root_pos_w = self.noiseModel.apply(root_pos_w, "root_pos")
            if "root_quat" in self.noiseModel.params:
                axis = torch.rand(self.num_envs, 3, device=self.device)
                axis = axis / axis.norm(dim=-1, keepdim=True)
                angle = torch.empty(self.num_envs, 1, device=self.device).uniform_(
                    -self.cfg.noiseCfg["root_quat"]["std"], self.cfg.noiseCfg["root_quat"]["std"]
                )
                box_quat = quat_from_angle_axis(angle.squeeze(1), axis)
                root_quat_w = quat_mul(box_quat, root_quat_w)
            if "dof_pos" in self.noiseModel.params:
                gimbal_pos = self.noiseModel.apply(gimbal_pos, "dof_pos")

        root_lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
        projected_gravity_b = quat_apply_inverse(root_quat_w, projected_gravity_w)

        goal_pos_b, goal_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, self._desired_pos_w, self._desired_quat_w
        )
        root_rot_mat = matrix_from_quat(root_quat_w)
        root_rot_vec = root_rot_mat[:, :2, :].reshape(self.num_envs, 6)

        goal_rot_mat = matrix_from_quat(goal_quat_b)
        goal_rot_vec = goal_rot_mat[:, :2, :].reshape(self.num_envs, 6)

        # IMU Debug Info
        # imu_ang_vel_b = quat_apply(self._imu_quat_body, self._robot.data.body_ang_vel_w[:, self._imu_id])
        # print("IMU Link   Ang Vel (body frame): ", imu_ang_vel_b[0])
        # print("Root       Ang Vel (body frame): ", self._robot.data.root_ang_vel_b[0])
        # print("IMU Sensor Ang Vel (body frame): ", self._imu_sensor.data.ang_vel_b[0])
        # # _robot.data.projected_gravity_b = quat_apply_inverse(self.root_link_quat_w, self.GRAVITY_VEC_W)
        # imu_gravity_b = quat_apply(self._imu_quat_body, self._robot.data.GRAVITY_VEC_W)
        # print("IMU Link   Prj Gra (body frame): ", imu_gravity_b[0])
        # print("Root       Prj Gra (body frame): ", self._robot.data.projected_gravity_b[0])
        # print("IMU Sensor Prj Gra (body frame): ", self._imu_sensor.data.projected_gravity_b[0])

        obs = torch.cat(
            (
                root_lin_vel_b * self.obsScales.lin_vel,
                root_ang_vel_b * self.obsScales.ang_vel,
                projected_gravity_b,
                goal_pos_b,
                gimbal_pos,
                root_rot_vec,
                goal_rot_vec,
                self._last_actions,
            ),
            dim=-1,
        )
        clip_obs = self.ctrlCfg.clip_observations
        obs = torch.clamp(obs, -clip_obs, clip_obs)
        states = self._get_states()
        states = torch.clamp(states, -clip_obs, clip_obs)
        observations = {"policy": obs, "critic": states}
        return observations

    def _get_states(self) -> torch.Tensor:
        goal_pos_b, goal_quat_b = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w, self._desired_quat_w
        )
        root_rot_mat = matrix_from_quat(self._robot.data.root_quat_w)
        root_rot_vec = root_rot_mat[:, :2, :].reshape(self.num_envs, 6)

        goal_rot_mat = matrix_from_quat(goal_quat_b)
        goal_rot_vec = goal_rot_mat[:, :2, :].reshape(self.num_envs, 6)

        states = torch.cat(
            (
                self._robot.data.root_lin_vel_b * self.obsScales.lin_vel,
                self._robot.data.root_ang_vel_b * self.obsScales.ang_vel,
                self._robot.data.projected_gravity_b,
                goal_pos_b,
                self._robot.data.joint_pos[:, self._gimbal_ids[0]] - self._gimbal_default_pos,
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
        # lin_vel_over = torch.clamp(lin_vel_norm - self.cfg.lin_vel_th, min=0.0)
        # lin_vel = torch.square(lin_vel_over)
        # exp(2.3984) - 1.0 = 10.0
        lin_vel_norm = torch.clamp(lin_vel_norm, max=4.0)
        lin_vel = torch.square(torch.exp(0.6 * lin_vel_norm) - 1.0)

        ang_vel_norm = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=1)
        # ang_vel_over = torch.clamp(ang_vel_norm - self.cfg.ang_vel_th, min=0.0)
        # ang_vel = torch.square(ang_vel_over)
        ang_vel_norm = torch.clamp(ang_vel_norm, max=6.0)
        ang_vel = torch.square(torch.exp(0.4 * ang_vel_norm) - 1.0)

        # distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal = torch.linalg.norm(self._position_error, dim=1)
        # distance_to_goal_mapped = torch.square(torch.tanh(distance_to_goal / 0.6))
        # distance_to_goal_mapped = torch.square(torch.tanh(distance_to_goal))
        # distance_to_goal_mapped = torch.square(torch.tanh(0.5 * distance_to_goal))
        # distance_to_goal_mapped = torch.tanh(distance_to_goal / 0.6)
        distance_to_goal_mapped = torch.tanh(distance_to_goal / 1.5) + torch.tanh(distance_to_goal / 0.6)
        # distance_to_goal_mapped = torch.tanh(distance_to_goal * 0.6)
        # distance_to_goal_mapped = torch.square(distance_to_goal / 2.0)
        # distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        # distance_to_goal_mapped = torch.exp(-2 * distance_to_goal)
        # distance_to_goal_weight = torch.exp(-torch.square(3.0 * distance_to_goal))
        # distance_to_goal_weight = 1 - torch.tanh(distance_to_goal / 0.2)
        distance_to_goal_weight = 1 - torch.tanh(distance_to_goal)
        distance_to_goal_weight = 1 - torch.tanh(0.5 * distance_to_goal)
        thrust_power = torch.sum(torch.square(self._action_thrust_force), dim=1)

        angular_to_goal = torch.linalg.norm(self._angle_error, dim=1)
        # angular_to_goal_mapped = (1 - torch.tanh(angular_to_goal)) * (distance_to_goal_weight)
        angular_to_goal_mapped = (1 - torch.tanh(0.5 * angular_to_goal)) * distance_to_goal_weight
        # angular_to_goal_mapped = torch.exp(-2 * angular_to_goal) * distance_to_sgoal_weight
        # angular_to_goal_mapped = (1 - torch.exp(-angular_to_goal / 0.8))
        # angular_error_to_goal_mapped = 1 - torch.exp(-angular_to_goal)

        # current_time_s = self.episode_length_buf * self.cfg.sim.dt * self.cfg.decimation
        # distance_reward_map = current_time_s / self._desired_time_to_goal_s
        # distance_reward_map = torch.clamp(distance_reward_map, max=1.0)

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "thrust_power": thrust_power * self.cfg.thrust_power_reward_scale * self.step_dt,
            "angular_to_goal": angular_to_goal_mapped * self.cfg.angular_to_goal_reward_scale * self.step_dt,
        }
        total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Early termination penalty
        die_reward = self.reset_terminated.to(torch.float32) * self.cfg.died_reward_scale * self.max_episode_length_s
        total_reward += die_reward
        rewards["died"] = die_reward

        # timeout reward with reaching goal
        ang_vel_norm = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=-1)
        lin_vel_norm = torch.linalg.norm(self._robot.data.root_lin_vel_b, dim=-1)

        ANG_VEL_TH = 0.02
        LIN_VEL_TH = 0.02
        POS_TH = 0.02
        ANG_TH = 0.05
        reach_goal = torch.logical_and(
            torch.logical_and(ang_vel_norm < ANG_VEL_TH, lin_vel_norm < LIN_VEL_TH),
            torch.logical_and(angular_to_goal < ANG_TH, distance_to_goal < POS_TH),
        )
        reach_pos = torch.logical_and(distance_to_goal < POS_TH, lin_vel_norm < LIN_VEL_TH)
        # reach_goal = torch.logical_and(
        #     reach_goal,
        #     self.episode_length_buf > (self.max_episode_length - (2.0 / (self.cfg.sim.dt * self.cfg.decimation))),
        # )
        self._reach_goal_state = reach_goal
        self._reach_goal = reach_goal.to(torch.float32) * self.reset_time_outs.to(torch.float32)
        self._reach_goal_count += self._reach_goal
        reach_goal_reward_timeout = (
            self.reset_time_outs.to(torch.float32) * reach_goal.to(torch.float32) * self.max_episode_length_s
        ) * self.cfg.reach_goal_reward_timeout_scale
        reach_goal_reward = torch.zeros_like(reach_goal_reward_timeout)
        # reach_goal_reward += reach_goal.to(torch.float32) * self.step_dt * 1.0
        reach_goal_reward += torch.exp(-2 * distance_to_goal / POS_TH) * reach_goal.to(torch.float32) * self.step_dt
        reach_goal_reward += torch.exp(-2 * angular_to_goal / ANG_TH) * reach_goal.to(torch.float32) * self.step_dt
        reach_goal_reward += torch.exp(-2 * lin_vel_norm / LIN_VEL_TH) * reach_goal.to(torch.float32) * self.step_dt
        reach_goal_reward += torch.exp(-2 * ang_vel_norm / ANG_VEL_TH) * reach_goal.to(torch.float32) * self.step_dt
        reach_goal_reward = reach_goal_reward * self.cfg.reach_goal_reward_scale

        reach_pos_reward = torch.zeros_like(reach_goal_reward_timeout)
        reach_pos_reward += torch.exp(-2 * distance_to_goal / POS_TH) * reach_pos.to(torch.float32) * self.step_dt
        reach_pos_reward += torch.exp(-2 * lin_vel_norm / LIN_VEL_TH) * reach_pos.to(torch.float32) * self.step_dt
        reach_pos_reward = reach_pos_reward * self.cfg.reach_pos_reward_scale

        total_reward += reach_goal_reward
        rewards["reach_goal"] = reach_goal_reward
        total_reward += reach_pos_reward
        rewards["reach_pos"] = reach_pos_reward
        total_reward += reach_goal_reward_timeout
        rewards["reach_goal_timeout"] = reach_goal_reward_timeout

        # functional smoothness penalties

        gimbal_action_rate = torch.sum(
            torch.square(self._actions[:, : self.cfg.gimbal_num] - self._last_actions[:, : self.cfg.gimbal_num]), dim=1
        )
        rewards["gimbal_action_rate"] = gimbal_action_rate * self.cfg.gimbal_action_rate_reward_scale * self.step_dt

        thrust_action_rate = torch.sum(
            torch.square(self._actions[:, self.cfg.gimbal_num :] - self._last_actions[:, self.cfg.gimbal_num :]), dim=1
        )
        rewards["thrust_action_rate"] = thrust_action_rate * self.cfg.thrust_action_rate_reward_scale * self.step_dt

        self._gimbal_vel = (self._robot.data.joint_vel[:, self._gimbal_ids[0]]).clone()
        gimbal_acc = torch.sum(torch.square(self._gimbal_vel - self._gimbal_vel_last), dim=1) / self.step_dt
        rewards["gimbal_acc"] = gimbal_acc * self.cfg.gimbal_acc_reward_scale * self.step_dt
        self._gimbal_vel_last = self._gimbal_vel.clone()

        gimbal_vel = torch.sum(torch.square(self._gimbal_vel), dim=1)
        rewards["gimbal_vel"] = gimbal_vel * self.cfg.gimbal_vel_reward_scale * self.step_dt

        self._gimbal_pos = (self._robot.data.joint_pos[:, self._gimbal_ids[0]]).clone()
        gimbal_limit = -(self._gimbal_pos - (-self.cfg.gimbal_limit_scale)).clip(max=0.0)
        gimbal_limit += (self._gimbal_pos - (self.cfg.gimbal_limit_scale)).clip(min=0.0)
        gimbal_limit = torch.sum(gimbal_limit, dim=1)
        rewards["gimbal_limit"] = gimbal_limit * self.cfg.gimbal_limit_reward_scale * self.step_dt

        thrust_limit = (self._action_thrust_force - self.cfg.thrust_limit).clip(min=0.0)
        thrust_limit = torch.sum(torch.square(thrust_limit), dim=1)
        rewards["thrust_limit"] = thrust_limit * self.cfg.thrust_limit_reward_scale * self.step_dt

        thrust_average = torch.mean(self._action_thrust_force, dim=1)
        thrust_uneven = torch.sum(torch.square(self._action_thrust_force - thrust_average.unsqueeze(-1)), dim=1)
        thrust_uneven_refer = torch.abs(self._robot.data.projected_gravity_b[:, 2])
        rewards["thrust_uneven"] = (
            thrust_uneven * thrust_uneven_refer * self.cfg.thrust_uneven_reward_scale * self.step_dt
        )

        total_reward += gimbal_action_rate * self.cfg.gimbal_action_rate_reward_scale * self.step_dt
        total_reward += thrust_action_rate * self.cfg.thrust_action_rate_reward_scale * self.step_dt
        total_reward += gimbal_acc * self.cfg.gimbal_acc_reward_scale * self.step_dt
        total_reward += gimbal_vel * self.cfg.gimbal_vel_reward_scale * self.step_dt
        total_reward += gimbal_limit * self.cfg.gimbal_limit_reward_scale * self.step_dt
        total_reward += thrust_limit * self.cfg.thrust_limit_reward_scale * self.step_dt
        total_reward += thrust_uneven * self.cfg.thrust_uneven_reward_scale * self.step_dt

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        # total_reward = torch.zeros(self.num_envs, device=self.device)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.ctrlCfg.ang_reset_rate > 0.0 and self.common_step_counter > self.cfg.ang_curricular_steps:
            reset_mask = (self.episode_length_buf > self._reset_ang_episode_counts) & ~self._reset_ang_pose_mask
            reset_ang_mask = reset_mask & (torch.rand(self.num_envs, device=self.device) < self.ctrlCfg.ang_reset_rate)
            reset_env_ids = torch.nonzero(reset_ang_mask, as_tuple=False).squeeze(-1)
            ang_range = self.ctrlCfg.body_ang * self._quat_sample_rate
            self._desired_quat_w[reset_env_ids] = sampleCenterQuatwithTilt(
                torch.tensor(ang_range), len(reset_env_ids)
            ).to(self.device)
            self._reset_ang_pose_mask |= reset_mask
        # import ipdb; ipdb.set_trace()
        # died = torch.linalg.norm(self._contact_sensor.data.force_matrix_w.squeeze(1).squeeze(1), dim=-1) > 0.1
        force_norm = torch.linalg.norm(self._contact_sensor.data.net_forces_w_history, dim=-1)
        crash = (force_norm > self.cfg.contact_force_threshold).any(dim=(1, 2))
        # crash = (
        #     torch.linalg.norm(self._contact_sensor.data.net_forces_w.squeeze(1), dim=-1)
        #     > self.cfg.contact_force_threshold
        # )
        drift = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 6.0)
        died = torch.logical_or(crash, drift)
        # # # DEBUG
        if self.cfg.evaluate_mode:
            died = torch.zeros_like(time_out, dtype=torch.bool)
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        quat_sample_rate = (
            (self.common_step_counter + 300 * self.cfg.num_steps_per_env) / self.cfg.max_curricular_steps * 6
        )  # start from 0.3, reach 0.8
        pos_sample_rate = (
            (self.common_step_counter + 100 * self.cfg.num_steps_per_env) / self.cfg.max_curricular_steps * 6
        )  # start from 0.1, reach 0.6
        success_flags = self._reach_goal  # success flags
        self._success_window = torch.cat([self._success_window, success_flags])[-self._success_window_size :]
        # Calculate sliding window success_rate
        success_rate = self._success_window.mean().item()
        # quat_sample_rate = self._quat_sample_rate
        # if success_rate > 0.85:
        #     quat_sample_rate = min(1.0, quat_sample_rate + 0.1)
        # elif success_rate < 0.60:
        #     quat_sample_rate = max(0.2, quat_sample_rate - 0.1)
        self._success_rate = success_rate
        quat_sample_rate = max(quat_sample_rate, 0.0)
        pos_sample_rate = max(pos_sample_rate, 0.0)

        if self.cfg.play_mode or self.cfg.evaluate_mode:
            quat_sample_rate = 1.0
            pos_sample_rate = 1.0
            if self.ctrlCfg.ang_reset_rate > 0.0:
                self.ctrlCfg.ang_reset_rate = 1.0

        quat_sample_rate = min(quat_sample_rate, 1.0)
        pos_sample_rate = min(pos_sample_rate, 1.0)
        # quat_sample_rate = 1.0
        # pos_sample_rate = 1.0
        self._quat_sample_rate = quat_sample_rate
        self._pos_sample_rate = pos_sample_rate

        # ### Logging before reset
        final_distance_to_goal = torch.mean(torch.linalg.norm(self._position_error[env_ids], dim=1))
        final_angle_error = torch.mean(torch.linalg.norm(self._angle_error[env_ids], dim=1))

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
        extras["Metrics/quat_sample_rate"] = self._quat_sample_rate
        extras["Metrics/pos_sample_rate"] = self._pos_sample_rate
        extras["Metrics/reach_goal_reset"] = torch.mean(self._reach_goal).item()
        extras["Metrics/reach_goal_count"] = torch.mean(self._reach_goal_count).item()
        extras["Metrics/success_rate"] = success_rate
        extras["Metrics/timeouts"] = torch.sum(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        self._contact_sensor.reset(env_ids)
        self._rotors.reset(env_ids)
        self.noiseModel.reset(env_ids)
        self._reset_ang_pose_mask[env_ids] = False
        if self.cfg.add_randomization:
            self._gimbal_default_pos[env_ids] = self._initial_gimbal_pos[env_ids] + torch.empty_like(
                self._initial_gimbal_pos[env_ids]
            ).uniform_(-self.randomCfg.dof_pos, self.randomCfg.dof_pos)

        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._action_gimbal_pos[env_ids] = 0.0
        self._action_thrust_force[env_ids] = 0.0

        self._position_error[env_ids] = 0.0
        self._angle_error[env_ids] = 0.0

        self._gimbal_pos[env_ids] = 0.0
        self._gimbal_vel[env_ids] = 0.0
        self._gimbal_vel_last[env_ids] = 0.0
        self._last_gimbal_pos[env_ids] = 0.0
        self._last_last_gimbal_pos[env_ids] = 0.0
        self._body_ang_vel_buffer[env_ids] = 0.0
        self._body_lin_vel_buffer[env_ids] = 0.0

        # ######################### Sample new commands #############################
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

        ang_range = self.ctrlCfg.body_ang * quat_sample_rate
        pos_range = 5.0 * pos_sample_rate
        pos_range_z = 2.5 * pos_sample_rate

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        if self.cfg.evaluate_mode:
            # self._desired_zyx_euler_w[env_ids] = torch.zeros_like(self._desired_zyx_euler_w[env_ids])
            # self._desired_zyx_euler_w[env_ids, 0] = torch.pi * 0.2
            # self._desired_zyx_euler_w[env_ids, 2] = torch.pi * 0.5
            self._desired_zyx_euler_w[env_ids] = torch.empty_like(self._desired_zyx_euler_w[env_ids]).uniform_(
                -ang_range, ang_range
            )
            self._desired_quat_w[env_ids] = quat_from_euler_xyz(
                self._desired_zyx_euler_w[env_ids, 0],
                self._desired_zyx_euler_w[env_ids, 1],
                self._desired_zyx_euler_w[env_ids, 2],
            )
            self._desired_pos_w[env_ids, :3] = torch.zeros_like(self._desired_pos_w[env_ids, :3])
            self._desired_pos_w[env_ids, :3] += self._terrain.env_origins[env_ids]
            # self._desired_pos_w[env_ids, 0] += 0.5
            self._desired_pos_w[env_ids, 2] += 1.5
            default_root_state[:, :2] = default_root_state[:, :2] + torch.empty_like(
                default_root_state[:, :2]
            ).uniform_(-2.0, 2.0)
            default_root_state[:, 3:7] = sampleUniformQuatwithTilt(
                torch.tensor(self.randomCfg.body_ang), len(env_ids)
            ).to(self.device)
            default_root_state[:, 7:10] = (
                torch.empty_like(default_root_state[:, 7:10]).uniform_(-1, 1) * self.randomCfg.lin_vel
            )
            # Angular velocity
            default_root_state[:, 10:13] = (
                torch.empty_like(default_root_state[:, 10:13]).uniform_(-1, 1) * self.randomCfg.ang_vel
            )
            # default_root_quat = quat_from_euler_xyz(
            #     torch.ones_like(self._desired_zyx_euler_w[env_ids, 0]) * 0.0,
            #     torch.ones_like(self._desired_zyx_euler_w[env_ids, 1]) * 0.0,
            #     torch.ones_like(self._desired_zyx_euler_w[env_ids, 2]) * (math.pi * 0.5),
            # )
            # default_root_state[:, 3:7] = default_root_quat
            # default_root_state[:, 2] = 0.1
        else:  # random reset ##########################################################################################
            self._desired_quat_w[env_ids] = sampleCenterQuatwithTilt(torch.tensor(ang_range), len(env_ids)).to(
                self.device
            )
            self._desired_pos_w[env_ids, :2] = torch.empty_like(self._desired_pos_w[env_ids, :2]).uniform_(
                -pos_range, pos_range
            )
            self._desired_pos_w[env_ids, 2] = torch.empty_like(self._desired_pos_w[env_ids, 2]).uniform_(
                max(0.6, 2.5 - pos_range_z), min(2.5 + pos_range_z, 5.0)
            )
            self._desired_pos_w[env_ids, :3] += self._terrain.env_origins[env_ids, :3]
            default_root_state[:, 3:7] = sampleUniformQuatwithTilt(
                torch.tensor(self.randomCfg.body_ang), len(env_ids)
            ).to(self.device)
            # Linear velocity
            default_root_state[:, 7:10] = (
                torch.empty_like(default_root_state[:, 7:10]).uniform_(-1, 1) * self.randomCfg.lin_vel
            )
            # Angular velocity
            default_root_state[:, 10:13] = (
                torch.empty_like(default_root_state[:, 10:13]).uniform_(-1, 1) * self.randomCfg.ang_vel
            )

        desired_distance = torch.linalg.norm(self._desired_pos_w[env_ids] - default_root_state[:, :3], dim=1)
        self._desired_time_to_goal_s[env_ids] = desired_distance / self.ctrlCfg.ref_lin_vel
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

        # print(f"Debug Envent {env_ids}  Reset: ==========================================================")
        # robot_mass = self._robot.root_physx_view.get_masses()[env_ids].sum()
        # print(f"Robot [{env_ids}] mass: {robot_mass:.4f}, default mass: {self._robot_mass:.4f}")
        # root_com = self._robot.data.body_com_pose_w[env_ids, self._body_id].clone().cpu().numpy()
        # root_com_str = ", ".join(f"{x:.4f}" for x in root_com.flatten())
        # default_com_str = ", ".join(f"{x:.4f}" for x in self._robot_default_com.flatten())
        # print(f"Robot [{env_ids}] root com: [{root_com_str}], default com: [{default_com_str}]")

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

        if thrusts_pos is None or thrusts_quat is None:
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
        # orientations = quat_mul(thrusts_quat, orientations)
        # orientations = quat_mul(orientations, x2z_quat)

        orientations = quat_mul(thrusts_quat, x2z_quat)

        scales = torch.ones_like(thrusts)
        scales[:, 0] = thrusts_mag * 0.5
        self.thrusts_visualizer.visualize(translations=thrusts_pos, orientations=orientations, scales=scales)
