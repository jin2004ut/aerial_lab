# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read the done logic with line numbers and a condition summary."""

from pathlib import Path


TARGET = Path(
    "/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/quadcopter/mini_quadcopter_env.py"
)
START = 784
END = 798


def main():
    print("[INFO] Reading _get_dones() in MiniQuadcopterEnv")
    print("[INFO] Condition summary:")
    print("  terminated = crash OR drift")
    print("  truncated  = time_out")
    print("  crash      = contact force larger than threshold")
    print("  drift      = altitude too low or too high")
    print()

    for lineno, line in enumerate(TARGET.read_text().splitlines(), start=1):
        if START <= lineno <= END:
            print(f"{lineno:04d}: {line}")


if __name__ == "__main__":
    main()
