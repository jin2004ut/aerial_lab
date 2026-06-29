# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create a minimal Isaac Sim scene with ground and a dome light."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Create a minimal Isaac Sim scene.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.8])

    spawn_ground_plane("/World/ground", GroundPlaneCfg())

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    sim.reset()

    print("[INFO] Added ground plane and dome light.")
    print("[INFO] This is the minimal scene setup used by many Isaac Lab tasks.")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
