# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read the reset flow with line numbers and a reset checklist."""

from pathlib import Path


TARGET = Path(
    "/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/quadcopter/mini_quadcopter_env.py"
)
START = 799
END = 943


def main():
    print("[INFO] Reading _reset_idx() in MiniQuadcopterEnv")
    print("[INFO] Reset checklist:")
    print("  1. update success-rate curriculum variables")
    print("  2. log episode metrics into self.extras['log']")
    print("  3. reset robot, sensors, and rotor model")
    print("  4. randomize desired goal pose")
    print("  5. randomize root pose / velocity")
    print("  6. write root pose, root velocity, and joint state back to sim")
    print()

    for lineno, line in enumerate(TARGET.read_text().splitlines(), start=1):
        if START <= lineno <= END:
            print(f"{lineno:04d}: {line}")


if __name__ == "__main__":
    main()
