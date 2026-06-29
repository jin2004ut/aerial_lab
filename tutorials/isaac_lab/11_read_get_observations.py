# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read the observation-building code with line numbers and a short guide."""

from pathlib import Path


TARGET = Path(
    "/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/quadcopter/mini_quadcopter_env.py"
)
START = 574
END = 672


def main():
    print("[INFO] Reading _get_observations() in MiniQuadcopterEnv")
    print("[INFO] Focus points:")
    print("  1. optional observation noise")
    print("  2. world-frame -> body-frame conversion")
    print("  3. goal pose expressed in the robot body frame")
    print("  4. torch.cat(...) that creates the final policy observation")
    print()

    for lineno, line in enumerate(TARGET.read_text().splitlines(), start=1):
        if START <= lineno <= END:
            print(f"{lineno:04d}: {line}")


if __name__ == "__main__":
    main()
