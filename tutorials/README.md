# Isaac Sim Tutorials for Aerial Lab

This directory starts with Isaac Sim basics and then moves into Isaac Lab environment structure.

## Learning order

### Isaac Sim
1. `isaac_sim/00_create_empty.py`
2. `isaac_sim/01_add_scene.py`
3. `isaac_sim/02_spawn_robot.py`
4. `isaac_sim/03_step_robot.py`
5. `isaac_sim/04_read_robot_state.py`
6. `isaac_sim/05_add_imu_sensor.py`
7. `isaac_sim/06_compare_root_vs_imu.py`
8. `isaac_sim/07_compare_control_modes.py`
9. `isaac_sim/08_build_observation.py`
10. `isaac_sim/09_reset_and_randomize.py`
11. `isaac_sim/10_add_contact_sensor.py`

### Isaac Lab
12. `isaac_lab/00_list_envs.py`
13. `isaac_lab/01_inspect_env_cfg.py`
14. `isaac_lab/02_step_env.py`
15. `isaac_lab/03_trace_obs_reward_done.py`
16. `isaac_lab/04_trace_apply_action.py`
17. `isaac_lab/05_trace_reset_flow.py`
18. `isaac_lab/06_trace_reward_terms.py`
19. `isaac_lab/07_trace_done_reasons.py`
20. `isaac_lab/08_trace_train_pipeline.py`
21. `isaac_lab/09_trace_policy_io.py`
22. `isaac_lab/10_trace_rollout_buffer.py`
23. `isaac_lab/11_read_get_observations.py`
24. `isaac_lab/12_read_get_rewards.py`
25. `isaac_lab/13_read_get_dones.py`
26. `isaac_lab/14_read_reset_idx.py`

### Kinikun
27. `kinikun/00_inspect_asset.py`
28. `kinikun/01_spawn_kinikun.py`
29. `kinikun/02_step_arm_joints.py`
30. `kinikun/03_read_kinikun_state.py`
31. `kinikun/04_add_imu_sensor.py`
32. `kinikun/05_step_each_arm_joint.py`

## Run examples

Run these from `/workspace/aerial_lab` inside the container or from a Python environment where Isaac Sim, Isaac Lab, and `aerial_lab` are installed.

```bash
python tutorials/isaac_sim/00_create_empty.py --headless
python tutorials/isaac_sim/01_add_scene.py
python tutorials/isaac_sim/02_spawn_robot.py --robot mini_quad
python tutorials/isaac_sim/03_step_robot.py --robot mini_quad --steps 600
python tutorials/isaac_sim/04_read_robot_state.py --robot mini_quad --steps 240 --print_every 60
python tutorials/isaac_sim/05_add_imu_sensor.py --robot mini_quad --steps 240 --print_every 60
python tutorials/isaac_sim/06_compare_root_vs_imu.py --robot mini_quad --steps 240 --print_every 60
python tutorials/isaac_sim/07_compare_control_modes.py --robot beetle --mode position --steps 600
python tutorials/isaac_sim/08_build_observation.py --steps 240 --print_every 60
python tutorials/isaac_sim/09_reset_and_randomize.py --episodes 5
python tutorials/isaac_sim/10_add_contact_sensor.py --robot mini_quad --steps 300 --print_every 30
python tutorials/isaac_lab/00_list_envs.py
python tutorials/isaac_lab/01_inspect_env_cfg.py --task Aerial-MiniQuadcopter-Pose-v0 --headless
python tutorials/isaac_lab/02_step_env.py --task Aerial-MiniQuadcopter-Pose-v0 --num_envs 8 --steps 240 --headless
python tutorials/isaac_lab/03_trace_obs_reward_done.py --task Aerial-MiniQuadcopter-Pose-v0 --num_envs 4 --steps 120 --headless
python tutorials/isaac_lab/04_trace_apply_action.py --task Aerial-MiniQuadcopter-Pose-v0 --num_envs 2 --steps 8 --headless
python tutorials/isaac_lab/05_trace_reset_flow.py --task Aerial-MiniQuadcopter-Pose-v0 --num_envs 8 --steps 400 --headless
python tutorials/isaac_lab/06_trace_reward_terms.py --task Aerial-MiniQuadcopter-Pose-v0 --num_envs 4 --steps 120 --headless
python tutorials/isaac_lab/07_trace_done_reasons.py --task Aerial-MiniQuadcopter-Pose-v0 --num_envs 8 --steps 300 --headless
python tutorials/isaac_lab/08_trace_train_pipeline.py --task Aerial-MiniQuadcopter-Pose-v0 --num_envs 8 --headless
python tutorials/isaac_lab/09_trace_policy_io.py --task Aerial-MiniQuadcopter-Pose-v0 --num_envs 8 --steps 5 --headless
python tutorials/isaac_lab/10_trace_rollout_buffer.py --task Aerial-MiniQuadcopter-Pose-v0 --num_envs 4096 --headless
python tutorials/isaac_lab/11_read_get_observations.py
python tutorials/isaac_lab/12_read_get_rewards.py
python tutorials/isaac_lab/13_read_get_dones.py
python tutorials/isaac_lab/14_read_reset_idx.py
python tutorials/kinikun/00_inspect_asset.py
python tutorials/kinikun/01_spawn_kinikun.py --headless
python tutorials/kinikun/02_step_arm_joints.py --steps 600 --headless
python tutorials/kinikun/03_read_kinikun_state.py --steps 240 --print_every 60 --headless
python tutorials/kinikun/04_add_imu_sensor.py --steps 240 --print_every 60 --headless
python tutorials/kinikun/05_step_each_arm_joint.py --steps 360 --print_every 30 --headless
```

## What you will learn

- `00_create_empty.py`: how to start Isaac Sim from a Python script.
- `01_add_scene.py`: how to create a minimal scene with ground and lighting.
- `02_spawn_robot.py`: how to spawn one of Aerial Lab's robot assets into the scene.
- `03_step_robot.py`: how to step physics, reset state, and send simple joint commands.
- `04_read_robot_state.py`: how to read root pose, body velocity, joint state, and IMU-like quantities.
- `05_add_imu_sensor.py`: how to instantiate a real `Imu` sensor object and compare its outputs with robot state tensors.
- `06_compare_root_vs_imu.py`: how to compare root-body signals, IMU-link signals, and IMU sensor outputs side by side.
- `07_compare_control_modes.py`: how actions enter the simulator through joint targets or external forces.
- `08_build_observation.py`: how an RL observation is assembled from state tensors.
- `09_reset_and_randomize.py`: how reset logic and state randomization are implemented.
- `10_add_contact_sensor.py`: how to instantiate a contact sensor and read collision forces.
- `isaac_lab/00_list_envs.py`: how to see which Gym tasks are registered by `aerial_lab`.
- `isaac_lab/01_inspect_env_cfg.py`: how a task name resolves to an environment config object.
- `isaac_lab/02_step_env.py`: how an Isaac Lab env is created and stepped through Gym.
- `isaac_lab/03_trace_obs_reward_done.py`: how observation, reward, and done flags appear at runtime.
- `isaac_lab/04_trace_apply_action.py`: how action tensors become internal thrust commands.
- `isaac_lab/05_trace_reset_flow.py`: how done flags trigger per-env resets.
- `isaac_lab/06_trace_reward_terms.py`: how reward terms accumulate during a step.
- `isaac_lab/07_trace_done_reasons.py`: how crash, drift, and time-out are detected.
- `isaac_lab/08_trace_train_pipeline.py`: how task, env cfg, runner cfg, and `runner.learn(...)` connect.
- `isaac_lab/09_trace_policy_io.py`: what the policy network sees on input and output.
- `isaac_lab/10_trace_rollout_buffer.py`: how rollout size, minibatch size, and update counts are computed.
- `isaac_lab/11_read_get_observations.py`: how to read the observation-building code with line numbers.
- `isaac_lab/12_read_get_rewards.py`: how to read the reward code with line numbers.
- `isaac_lab/13_read_get_dones.py`: how to read the done logic with line numbers.
- `isaac_lab/14_read_reset_idx.py`: how to read the reset flow with line numbers.
- `kinikun/00_inspect_asset.py`: how the original kinikun xacro maps to the copied Aerial Lab asset.
- `kinikun/01_spawn_kinikun.py`: how to spawn the dedicated kinikun articulation asset.
- `kinikun/02_step_arm_joints.py`: how to command the kinikun arm joints.
- `kinikun/03_read_kinikun_state.py`: how to read kinikun root and joint states.
- `kinikun/04_add_imu_sensor.py`: how to attach an IMU to the `fc` link.
- `kinikun/05_step_each_arm_joint.py`: how to move `arm1_joint` to `arm4_joint` one by one and inspect their targets and current values.

## Reading guide

For `02_spawn_robot.py` and `03_step_robot.py`, the files now include extra inline comments so you can read them almost line by line.

The key mental model for `03_step_robot.py` is:

1. `SimulationCfg(...)`: choose physics settings such as timestep and device.
2. `SimulationContext(...)`: create the object that owns the live physics world.
3. `Articulation(...)`: turn a robot asset config into a runtime robot object.
4. `write_*_to_sim(...)`: push state or commands into the simulator.
5. `sim.step()`: advance physics by one step.
6. `robot.update(...)`: pull the resulting state back into `robot.data.*`.

## Where This Maps Into Aerial Lab

If you want to connect each tutorial back to the actual RL environments, these are the best entry points:

- `02_spawn_robot.py` / `03_step_robot.py`
  - robot asset definitions:
    [source/aerial_lab/aerial_lab/assets/aerialrobot.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/assets/aerialrobot.py)
- `04_read_robot_state.py`
  - root/body state usage in observations:
    [mini_quadcopter_env.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/quadcopter/mini_quadcopter_env.py:576)
  - another direct aerial example:
    [beetle_env.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/beetle/beetle_env.py:611)
- `05_add_imu_sensor.py`
  - IMU config definition:
    [mini_quadcopter_env.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/quadcopter/mini_quadcopter_env.py:338)
  - IMU offset computation:
    [mini_quadcopter_env.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/quadcopter/mini_quadcopter_env.py:423)
  - IMU object creation:
    [mini_quadcopter_env.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/quadcopter/mini_quadcopter_env.py:503)
  - IMU sensor debug comparison:
    [mini_quadcopter_env.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/quadcopter/mini_quadcopter_env.py:620)
  - same pattern for beetle:
    [beetle_env.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/beetle/beetle_env.py:338)
  - same pattern for beetle omni:
    [beetle_omni_env.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/beetle/beetle_omni_env.py:381)
- `06_compare_root_vs_imu.py`
  - direct debug comparison pattern:
    [mini_quadcopter_env.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/quadcopter/mini_quadcopter_env.py:620)
  - same pattern in beetle:
    [beetle_env.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/beetle/beetle_env.py:657)
- `07_compare_control_modes.py`
  - gimbal position targets:
    [beetle_env.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/beetle/beetle_env.py:592)
  - external force application:
    [mini_quadcopter_env.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/quadcopter/mini_quadcopter_env.py:558)
- `08_build_observation.py`
  - observation construction:
    [mini_quadcopter_env.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/quadcopter/mini_quadcopter_env.py:576)
- `09_reset_and_randomize.py`
  - randomized reset logic:
    [mini_quadcopter_env.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/quadcopter/mini_quadcopter_env.py:900)
- `10_add_contact_sensor.py`
  - contact sensor config:
    [mini_quadcopter_env.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/quadcopter/mini_quadcopter_env.py:316)
  - contact force use:
    [mini_quadcopter_env.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/quadcopter/mini_quadcopter_env.py:789)
- `isaac_lab/00_list_envs.py`
  - registered task import entry:
    [source/aerial_lab/aerial_lab/tasks/__init__.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/__init__.py)
- `isaac_lab/01_inspect_env_cfg.py`
  - config resolution is the same path used before env creation in:
    [scripts/random_agent.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/scripts/random_agent.py:37)
- `isaac_lab/02_step_env.py`
  - Gym env creation pattern:
    [scripts/random_agent.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/scripts/random_agent.py:43)
- `isaac_lab/03_trace_obs_reward_done.py`
  - observation / reward / done internals:
    [mini_quadcopter_env.py](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/source/aerial_lab/aerial_lab/tasks/direct/quadcopter/mini_quadcopter_env.py:576)

## After this

Once these are comfortable, move to:

- [isaac_lab/README.md](/home/kitagawa/ros/jsk_aerial_robot_ws/src/aerial_lab/tutorials/isaac_lab/README.md) for the Isaac Lab-focused track
- `python scripts/rsl_rl/train.py --task Aerial-MiniQuadcopter-Pose-v0 --headless` for RL training
