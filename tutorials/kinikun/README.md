# Kinikun Tutorials for Aerial Lab

This directory is a dedicated learning track for bringing the `kinikun` robot into Aerial Lab.

Unlike `tutorials/isaac_sim/` and `tutorials/isaac_lab/`, this track is focused on one specific robot:

- where its URDF came from
- how it is packaged as an Isaac Lab articulation asset
- how to spawn it
- how to move the arm joints
- how to read state and IMU signals
- how to move arm joints 1-4 explicitly

At this stage, `kinikun` is available as an asset and tutorial target. A full Isaac Lab RL task for `kinikun` is not created yet.

## Learning order

1. `00_inspect_asset.py`
2. `01_spawn_kinikun.py`
3. `02_step_arm_joints.py`
4. `03_read_kinikun_state.py`
5. `04_add_imu_sensor.py`
6. `05_step_each_arm_joint.py`

## Run examples

Run these from `/workspace/aerial_lab` inside the container or from a Python environment where Isaac Sim, Isaac Lab, and `aerial_lab` are installed.

```bash
python tutorials/kinikun/00_inspect_asset.py
python tutorials/kinikun/01_spawn_kinikun.py --headless
python tutorials/kinikun/02_step_arm_joints.py --steps 600 --headless
python tutorials/kinikun/03_read_kinikun_state.py --steps 240 --print_every 60 --headless
python tutorials/kinikun/04_add_imu_sensor.py --steps 240 --print_every 60 --headless
python tutorials/kinikun/05_step_each_arm_joint.py --steps 360 --print_every 30 --headless
```

## What you will learn

- `00_inspect_asset.py`: how the original `kinikun` xacro and the copied Aerial Lab asset files line up.
- `01_spawn_kinikun.py`: how to spawn the `KINIKUN_CFG` articulation into Isaac Sim.
- `02_step_arm_joints.py`: how to command the four arm joints with simple position targets.
- `03_read_kinikun_state.py`: how to inspect root pose, arm joint state, and rotor joint state.
- `04_add_imu_sensor.py`: how to attach an IMU at the `fc` link and read its outputs.
- `05_step_each_arm_joint.py`: how to command `arm1_joint` to `arm4_joint` and compare targets with current joint positions.

## Where This Maps Into Aerial Lab

- asset definition:
  [aerialrobot.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/assets/aerialrobot.py)
- copied asset files:
  [source/aerial_lab/data/Robots/kinikun](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/data/Robots/kinikun)
- original xacro source:
  [robot.urdf.xacro](/home/kitagawa/ros/jsk_aerial_robot_ws/src/jsk_aerial_robot/robots/kinikun/urdf/robot.urdf.xacro)
