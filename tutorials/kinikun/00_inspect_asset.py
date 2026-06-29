# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect the kinikun asset files copied into Aerial Lab."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COPIED_ASSET_DIR = REPO_ROOT / "source/aerial_lab/data/Robots/kinikun"
COPIED_URDF = COPIED_ASSET_DIR / "kinikun.urdf"


def find_original_xacro() -> Path | None:
    candidates = [
        REPO_ROOT.parent / "jsk_aerial_robot/robots/kinikun/urdf/robot.urdf.xacro",
        Path("/home/kitagawa/ros/jsk_aerial_robot_ws/src/jsk_aerial_robot/robots/kinikun/urdf/robot.urdf.xacro"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def main():
    original_xacro = find_original_xacro()

    print("[INFO] original xacro")
    if original_xacro is not None:
        print(f"  {original_xacro}")
    else:
        print("  not found from this environment")
    print("[INFO] copied Aerial Lab asset dir")
    print(f"  {COPIED_ASSET_DIR}")
    print("[INFO] generated URDF")
    print(f"  {COPIED_URDF}")
    print()

    print("[INFO] top-level copied files")
    for path in sorted(COPIED_ASSET_DIR.iterdir()):
        print(f"  - {path.name}")
    print()

    print("[INFO] key URDF lines")
    for lineno, line in enumerate(COPIED_URDF.read_text().splitlines(), start=1):
        if any(token in line for token in ["<robot ", 'link name="main_body"', 'link name="fc"', 'joint name="arm1_joint"', 'joint name="rotor1"']):
            print(f"{lineno:04d}: {line}")


if __name__ == "__main__":
    main()
