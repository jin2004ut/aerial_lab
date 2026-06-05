# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim with an empty stage and step the app loop."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Launch an empty Isaac Sim stage.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    sim.reset()

    print("[INFO] Isaac Sim launched successfully.")
    print(f"[INFO] Device: {args_cli.device}")
    print("[INFO] This is an empty stage. Close the window or press Ctrl+C to exit.")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
