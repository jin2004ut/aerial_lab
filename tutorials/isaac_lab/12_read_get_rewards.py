# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read the reward function with line numbers and a reward-term checklist."""

from pathlib import Path


TARGET = Path(
    "/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/quadcopter/mini_quadcopter_env.py"
)
START = 673
END = 783


def main():
    print("[INFO] Reading _get_rewards() in MiniQuadcopterEnv")
    print("[INFO] Focus points:")
    print("  1. position / attitude error to goal")
    print("  2. velocity penalties")
    print("  3. thrust-power penalty")
    print("  4. reach-goal bonus and died penalty")
    print("  5. per-term logging through self._episode_sums")
    print()

    for lineno, line in enumerate(TARGET.read_text().splitlines(), start=1):
        if START <= lineno <= END:
            print(f"{lineno:04d}: {line}")


if __name__ == "__main__":
    main()
