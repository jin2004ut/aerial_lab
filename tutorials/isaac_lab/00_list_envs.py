# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""List the Isaac Lab tasks registered by aerial_lab."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import aerial_lab.tasks  # noqa: F401
import gymnasium as gym
from prettytable import PrettyTable


def main():
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Aerial Lab Isaac Lab Tasks"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 1
    for task_spec in sorted(gym.registry.values(), key=lambda spec: spec.id):
        if "Aerial-" in task_spec.id or "Beetle-" in task_spec.id:
            table.add_row([index, task_spec.id, task_spec.entry_point, task_spec.kwargs.get("env_cfg_entry_point", "-")])
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
