# Kinikun Tutorials for Aerial Lab

This directory is a dedicated learning track for bringing the `kinikun` robot into Aerial Lab.

Unlike `tutorials/isaac_sim/` and `tutorials/isaac_lab/`, this track is focused on one specific robot:

- where its URDF came from
- how it is packaged as an Isaac Lab articulation asset
- how to spawn it
- how to move the arm joints
- how to read state and IMU signals
- how to move arm joints 1-4 explicitly

At this stage there is one Isaac Lab RL environment for `kinikun`: `Kinikun-Arm-Direct-v0`, a minimal fixed-base task for arm joint-target tracking (see [Reinforcement learning environment](#reinforcement-learning-environment-kinikun-arm-direct-v0) below). It is only a starting point, not a full robot task.

## Learning order

1. `00_inspect_asset.py`
2. `01_spawn_kinikun.py`
3. `02_step_arm_joints.py`
4. `03_read_kinikun_state.py`
5. `04_add_imu_sensor.py`
6. `05_step_each_arm_joint.py`
7. `06_arm_joint_ppo.py`

## Run examples

Run these from `/workspace/aerial_lab` inside the container or from a Python environment where Isaac Sim, Isaac Lab, and `aerial_lab` are installed.

```bash
python tutorials/kinikun/00_inspect_asset.py
python tutorials/kinikun/01_spawn_kinikun.py --headless
python tutorials/kinikun/02_step_arm_joints.py --steps 600 --headless
python tutorials/kinikun/03_read_kinikun_state.py --steps 240 --print_every 60 --headless
python tutorials/kinikun/04_add_imu_sensor.py --steps 240 --print_every 60 --headless
python tutorials/kinikun/05_step_each_arm_joint.py --steps 360 --print_every 30 --headless
python tutorials/kinikun/06_arm_joint_ppo.py --device cpu --headless --train_iters 50
```

## What you will learn

- `00_inspect_asset.py`: how the original `kinikun` xacro and the copied Aerial Lab asset files line up.
- `01_spawn_kinikun.py`: how to spawn the `KINIKUN_CFG` articulation into Isaac Sim.
- `02_step_arm_joints.py`: how to command the four arm joints with simple position targets.
- `03_read_kinikun_state.py`: how to inspect root pose, arm joint state, and rotor joint state.
- `04_add_imu_sensor.py`: how to attach an IMU at the `fc` link and read its outputs.
- `05_step_each_arm_joint.py`: how to command `arm1_joint` to `arm4_joint` and compare targets with current joint positions.
- `06_arm_joint_ppo.py`: how to train a minimal fixed-base PPO policy for four-arm joint target tracking.

## Reinforcement learning environment (`Kinikun-Arm-Direct-v0`)

A minimal Isaac Lab `DirectRLEnv` version of the `06_arm_joint_ppo.py` task, so it can be trained
with the project's `rsl_rl` pipeline. Scope is intentionally small: fixed base, arm gravity
disabled, the policy commands the 4 arm joints (`arm1_joint`..`arm4_joint`) to track a randomly
sampled joint-position target. Domain randomization (startup) only covers arm-end link mass/inertia
and arm-joint friction.

- env: [kinikun_arm_env.py](../../source/aerial_lab/aerial_lab/tasks/direct/kinikun/kinikun_arm_env.py)
- PPO config: [agents/rsl_rl_ppo_cfg.py](../../source/aerial_lab/aerial_lab/tasks/direct/kinikun/agents/rsl_rl_ppo_cfg.py)

### Train

```bash
# full training run (headless, 4096 parallel envs)
python scripts/rsl_rl/train.py --task Kinikun-Arm-Direct-v0 --headless

# quick smoke test (few envs / iterations)
python scripts/rsl_rl/train.py --task Kinikun-Arm-Direct-v0 --headless --num_envs 64 --max_iterations 3
```

Checkpoints and logs are written to `logs/rsl_rl/kinikun_arm/<timestamp>/`.

### Play (visualize a trained policy)

```bash
# loads the latest checkpoint under logs/rsl_rl/kinikun_arm/ by default
python scripts/rsl_rl/play.py --task Kinikun-Arm-Direct-v0 --num_envs 16

# or point at a specific checkpoint
python scripts/rsl_rl/play.py --task Kinikun-Arm-Direct-v0 --num_envs 16 \
    --checkpoint logs/rsl_rl/kinikun_arm/<timestamp>/model_<iter>.pt
```

### Visualize training curves (TensorBoard)

```bash
tensorboard --logdir logs/rsl_rl/kinikun_arm
```

Then open http://localhost:6006. Useful scalars: `Train/mean_reward`,
`Episode_Reward/track`, and `Metrics/final_tracking_mae` (mean absolute joint-tracking error).

## Where This Maps Into Aerial Lab

- asset definition:
  [aerialrobot.py](../../source/aerial_lab/aerial_lab/assets/aerialrobot.py)
- copied asset files:
  [source/aerial_lab/data/Robots/kinikun](../../source/aerial_lab/data/Robots/kinikun)
- original xacro source (external, in the `jsk_aerial_robot` ROS workspace):
  `jsk_aerial_robot/robots/kinikun/urdf/robot.urdf.xacro`
- arm-tracking RL environment (`Kinikun-Arm-Direct-v0`):
  [source/aerial_lab/aerial_lab/tasks/direct/kinikun](../../source/aerial_lab/aerial_lab/tasks/direct/kinikun)
