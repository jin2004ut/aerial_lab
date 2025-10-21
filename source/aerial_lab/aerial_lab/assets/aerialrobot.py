# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the quadcopters"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg,DCMotorCfg
# from aerial_lab.actuators import RotorActuatorCfg
from aerial_lab.assets import ISAACLAB_ASSETS_DATA_DIR
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

MINI_QUADROTOR_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
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
        pos=(0.0, 0.0, 0.5),
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


BEETLE_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/beetle/beetle.urdf",
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
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            ".*": 0.0,
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
        "servos": DCMotorCfg(
            joint_names_expr=["gimbal.*"],
            effort_limit=1.0,
            saturation_effort=1.0,
            velocity_limit=10.0,
            stiffness=5.0,
            damping=0.1,
            friction=0.0,
        )
    },
)

BEETLE_OMNI_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/beetle_omni/beetle_art_omni.urdf",
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
        pos=(0.0, 0.0, 1.2),
        joint_pos={
            ".*": 0.0,
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
        "servos": DCMotorCfg(
            joint_names_expr=["gimbal.*"],
            effort_limit=1.0,
            saturation_effort=1.0,
            velocity_limit=10.0,
            stiffness=5.0,
            damping=0.1,
            friction=0.0,
        )
    },
)
"""Configuration for the Crazyflie quadcopter."""
