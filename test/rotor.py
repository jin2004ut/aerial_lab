
from isaaclab.app import AppLauncher

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app

from aerial_lab.actuators.rotorgroup import RotorGroup
from aerial_lab.actuators.rotor import Rotor
import math
import matplotlib.pyplot as plt
import torch


if __name__ == "__main__":
    """Main function."""
    # run the main function
    thrust_to_torque_ratio = 0.01
    cfg = {
        "rotor_num": 4,
        "dt": 0.01,
        "mode": "foc",
        "thrust_coeff": 1.0,
        "torque_coeff": thrust_to_torque_ratio,
        "max_vel": 200.0,
        "max_foc": 6.0,
        "vel_wn": 1.0,
        "vel_zeta": 0.8,
        "foc_wn": 1.0,
        "foc_zeta": 0.8,
    }
    env_num = 3
    ids = [0, 1, 2, 3]

    rotor_directions = torch.tensor([1.0, -1.0, -1.0, 1.0])
    group = RotorGroup(
        cfg=cfg,
        devices=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_envs=env_num,
        rotor_ids=ids,
        rotor_names=["FL", "FR", "RL", "RR"],
        rotor_directions=rotor_directions,
    )
    group.reset()

    rotors = Rotor(
        env_num=env_num,
        directions=rotor_directions,
        thrust_coeff=1.0,
        torque_coeff=thrust_to_torque_ratio,
        rotor_ids=ids,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    input = torch.rand((env_num, cfg["rotor_num"]), device=group.device)
    print("Input thrusts/commands:", input)
    forces, torques = group.forward(input)
    # forces, torques = group(input)
    print("Group dir", group.dir)
    print("")
    print("Forces shape from RotorGroup:", forces.shape)
    print("Forces from RotorGroup:", forces)
    print("Torques shape from RotorGroup:", torques.shape)
    print("Torques from RotorGroup:", torques)

    forces, torques = rotors.compute_dynamics(input)
    print("Forces shape from Rotor:", forces.shape)
    print("Forces from Rotor:", forces)
    print("Torques shape from Rotor:", torques.shape)
    print("Torques from Rotor:", torques)
