# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the quadcopters"""

from __future__ import annotations

import isaaclab.sim as sim_utils

# from aerial_lab.actuators import RotorActuatorCfg
from aerial_lab.assets import ISAACLAB_ASSETS_DATA_DIR
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

MINI_QUADROTOR_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        root_link_name="base_link",
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/mini_quadrotor/mini_quadrotor.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "rotor.*": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # ImplicitActuatorCfg: stiffness and damping are set into simulation engine directly
        "rotor": ImplicitActuatorCfg(
            joint_names_expr=["rotor.*"],
            velocity_limit=100.0,
            stiffness=0.0,
            damping=150.0,
        )
    },
)


KINIKUN_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        root_link_name="main_body",
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/kinikun/kinikun.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={"arm.*_joint": 0.0, "rotor.*": 0.0},
        joint_vel={"arm.*_joint": 0.0, "rotor.*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "rotor": ImplicitActuatorCfg(
            joint_names_expr=["rotor.*"],
            effort_limit=100.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=0.5,
            friction=0.0,
            dynamic_friction=0.0,
        ),
        "arm": DelayedPDActuatorCfg(
            joint_names_expr=["arm.*_joint"],
            effort_limit=1.0,
            velocity_limit=2.0,
            velocity_limit_sim=2.0,
            effort_limit_sim=1.0,
            stiffness=0.285,
            damping=0.019,
            friction=0.0,
            min_delay=4,
            max_delay=4,
        ),
    },
)


BEETLE_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        root_link_name="base_link",
        collider_type="convex_decomposition",  # convex_decomposition
        merge_fixed_joints=False,
        collision_from_visuals=False,
        self_collision=False,
        replace_cylinders_with_capsules=False,
        # TODO: fix the base_link.dae visual mesh
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/beetle_hyper/beetle_hyper.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        # TODO: change joint driver according to the real robot
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        # joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        #     gains={
        #         "rotor.*": sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        #         "gimbal.*": sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        #     },
        #     target_type={"rotor.*": "velocity", "gimbal.*": "position"},
        #     drive_type={"rotor.*": "force", "gimbal.*": "force"},
        # ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # ImplicitActuatorCfg: stiffness and damping are set into simulation engine directly
        # "rotor": DCMotorCfg(  # test for parameter setting
        #     joint_names_expr=["rotor.*"],
        #     effort_limit=10.0,
        #     velocity_limit=100.0,
        #     saturation_effort=10.0,
        #     stiffness=1.0,
        #     damping=0.1,
        # ),
        "rotor": ImplicitActuatorCfg(  # test for parameter setting
            joint_names_expr=["rotor.*"],
            effort_limit=100.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=0.5,
            friction=0.0,
            dynamic_friction=0.0,
        ),
        # "gimbal": DCMotorCfg(
        #     joint_names_expr=["gimbal.*"],
        #     effort_limit=6.6,
        #     saturation_effort=10.0,
        #     velocity_limit=3.0,
        #     velocity_limit_sim=3.0,
        #     effort_limit_sim=6.6,
        #     stiffness=5.0,
        #     damping=0.1,
        #     friction=0.0,
        # ),
        "gimbal": DelayedPDActuatorCfg(
            joint_names_expr=["gimbal.*"],
            effort_limit=3.0,
            velocity_limit=10.0,
            velocity_limit_sim=10.0,
            effort_limit_sim=3.0,
            # stiffness=5.0,
            # damping=0.1,
            # friction=0.0,
            # min_delay=0,
            # max_delay=0,
            stiffness=0.285,
            damping=0.019,
            friction=0.0,
            min_delay=4,
            max_delay=4,
        ),
    },
)

BEETLE_OMNI_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        root_link_name="base_link",
        collider_type="convex_decomposition",  # convex_decomposition
        merge_fixed_joints=False,
        collision_from_visuals=False,
        self_collision=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/beetle_omni/beetle_omni.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),
        joint_pos={
            "rotor.*": 5.0,
            "gimbal.*": 0.0,
        },
        joint_vel={
            "rotor.*": 0.0,
            "gimbal.*": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # ImplicitActuatorCfg: stiffness and damping are set into simulation engine directly
        "rotor": ImplicitActuatorCfg(
            joint_names_expr=["rotor.*"],
            velocity_limit=100.0,
            stiffness=0.0,
            damping=150.0,
        ),
        # "servos": DCMotorCfg(
        #     joint_names_expr=["gimbal.*"],
        #     effort_limit=1.0,
        #     saturation_effort=1.0,
        #     velocity_limit=3.0,
        #     velocity_limit_sim=3.0,
        #     effort_limit_sim=6.6,
        #     stiffness=5.0,
        #     damping=0.1,
        #     friction=0.0,
        # ),
        "gimbal": DelayedPDActuatorCfg(
            joint_names_expr=["gimbal.*"],
            effort_limit=3.0,
            velocity_limit=10.0,
            velocity_limit_sim=10.0,
            effort_limit_sim=3.0,
            # stiffness=5.0,
            # damping=0.1,
            # friction=0.0,
            # min_delay=1,
            # max_delay=3,
            stiffness=0.344964875,
            damping=0.0094725,
            friction=0.0,
            min_delay=2,
            max_delay=6,
        ),
    },
)

DRAGON_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        root_link_name="base_link",
        collider_type="convex_decomposition",  # convex_decomposition
        merge_fixed_joints=False,
        collision_from_visuals=False,
        self_collision=False,
        replace_cylinders_with_capsules=False,
        # TODO: fix the base_link.dae visual mesh
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/dragon/dragon.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        # TODO: change joint driver according to the real robot
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        # joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        #     gains={
        #         "rotor.*": sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        #         "gimbal.*": sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        #     },
        #     target_type={"rotor.*": "velocity", "gimbal.*": "position"},
        #     drive_type={"rotor.*": "force", "gimbal.*": "force"},
        # ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            "rotor.*": 0.0,
            "gimbal.*": 0.0,
            "joint.*_yaw": 1.0,
            "joint.*_pitch": 0.0,
        },
        joint_vel={
            "rotor.*": 0.0,
            "gimbal.*": 0.0,
            "joint.*_yaw": 0.0,
            "joint.*_pitch": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # ImplicitActuatorCfg: stiffness and damping are set into simulation engine directly
        # "rotor": DCMotorCfg(  # test for parameter setting
        #     joint_names_expr=["rotor.*"],
        #     effort_limit=10.0,
        #     velocity_limit=100.0,
        #     saturation_effort=10.0,
        #     stiffness=1.0,
        #     damping=0.1,
        # ),
        "rotor": ImplicitActuatorCfg(  # test for parameter setting
            joint_names_expr=["rotor.*"],
            effort_limit=100.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=0.5,
            friction=0.0,
            dynamic_friction=0.0,
        ),
        "gimbal": DCMotorCfg(
            joint_names_expr=["gimbal.*"],
            effort_limit=1.0,
            saturation_effort=1.0,
            velocity_limit=10.0,
            stiffness=5.0,
            damping=0.1,
            friction=0.0,
        ),
        "joint": DCMotorCfg(
            joint_names_expr=["joint.*"],
            effort_limit=1.0,
            saturation_effort=1.0,
            velocity_limit=10.0,
            stiffness=5.0,
            damping=0.1,
            friction=0.0,
        ),
    },
)

SPIDAR_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        root_link_name="center_link",
        collider_type="convex_decomposition",  # convex_decomposition
        merge_fixed_joints=False,
        collision_from_visuals=False,
        self_collision=False,
        replace_cylinders_with_capsules=False,
        # TODO: fix the base_link.dae visual mesh
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/spidar/spidar_v1_leg.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        # TODO: change joint driver according to the real robot
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        # joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        #     gains={
        #         "rotor.*": sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        #         "gimbal.*": sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        #     },
        #     target_type={"rotor.*": "velocity", "gimbal.*": "position"},
        #     drive_type={"rotor.*": "force", "gimbal.*": "force"},
        # ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            "^joint(?!.*_pitch$).*": 0.0,
            "rotor.*": 0.0,
            "gimbal.*": 0.0,
            "joint1_pitch": -0.45,
            "joint2_pitch": 1.45,
            "joint3_pitch": -0.45,
            "joint4_pitch": 1.45,
            "joint5_pitch": -0.45,
            "joint6_pitch": 1.45,
            "joint7_pitch": -0.45,
            "joint8_pitch": 1.45,
        },
        joint_vel={
            "rotor.*": 0.0,
            "gimbal.*": 0.0,
            "joint.*": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # ImplicitActuatorCfg: stiffness and damping are set into simulation engine directly
        # "rotor": DCMotorCfg(  # test for parameter setting
        #     joint_names_expr=["rotor.*"],
        #     effort_limit=10.0,
        #     velocity_limit=100.0,
        #     saturation_effort=10.0,
        #     stiffness=1.0,
        #     damping=0.1,
        # ),
        "rotor": ImplicitActuatorCfg(  # test for parameter setting
            joint_names_expr=["rotor.*"],
            effort_limit=100.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=0.5,
            friction=0.0,
            dynamic_friction=0.0,
        ),
        "gimbal": DCMotorCfg(
            joint_names_expr=["gimbal.*"],
            effort_limit=1.0,
            saturation_effort=1.0,
            velocity_limit=10.0,
            stiffness=5.0,
            damping=0.1,
            friction=0.0,
        ),
        "joint": DCMotorCfg(
            joint_names_expr=["joint.*"],
            effort_limit=1.0,
            saturation_effort=1.0,
            velocity_limit=10.0,
            stiffness=5.0,
            damping=0.1,
            friction=0.0,
        ),
    },
)
"""Configuration for the Crazyflie quadcopter."""
