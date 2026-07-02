# Isaac Lab Tutorials for Aerial Lab

This directory is the next learning track after `tutorials/isaac_sim/`.

The Isaac Sim tutorials helped you understand scene setup, sensors, control entry points, observation pieces, reset logic, and contact sensing.
Here, we move one layer up and look at how Isaac Lab wraps those pieces into an RL environment.

## Isaac Lab Basics

Before the scripts, it helps to keep four words straight:

- `task name`
  - a string such as `Aerial-MiniQuadcopter-Pose-v0`
  - this is what you pass to `gym.make(...)`
- `env`
  - the actual Python class that simulates the RL world
  - for example `MiniQuadcopterEnv`
- `cfg`
  - the configuration object used to build and parameterize the env
  - for example `MiniQuadcopterEnvCfg`
- `gym registry`
  - the lookup table that maps a task name to the env class and its config entry point

In RL, an `env` is the world that talks to the policy in a loop:

1. the env returns an `observation`
2. the policy chooses an `action`
3. the env applies that action and steps the world
4. the env returns the next `observation`, a `reward`, and `done` flags

Isaac Lab wraps Isaac Sim into this env interface so that training code can treat the simulator like a standard Gym task.

### What `gym.make(...)` does

When you run:

```python
gym.make("Aerial-MiniQuadcopter-Pose-v0")
```

Gym does not magically know what that string means.
It works only because `aerial_lab.tasks` registers the task ahead of time.

Conceptually, the registry stores something like:

- task name: `Aerial-MiniQuadcopter-Pose-v0`
- env class: `MiniQuadcopterEnv`
- cfg entry point: `MiniQuadcopterEnvCfg`

So `00_list_envs.py` is really showing you the registered task catalog for this repo.

## Learning order

1. `00_list_envs.py`
2. `01_inspect_env_cfg.py`
3. `02_step_env.py`
4. `03_trace_obs_reward_done.py`
5. `04_trace_apply_action.py`
6. `05_trace_reset_flow.py`
7. `06_trace_reward_terms.py`
8. `07_trace_done_reasons.py`
9. `08_trace_train_pipeline.py`
10. `09_trace_policy_io.py`
11. `10_trace_rollout_buffer.py`
12. `11_read_get_observations.py`
13. `12_read_get_rewards.py`
14. `13_read_get_dones.py`
15. `14_read_reset_idx.py`

## Run examples

Run these from `/workspace/aerial_lab` inside the container or from a Python environment where Isaac Sim, Isaac Lab, and `aerial_lab` are installed.

```bash
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
```

## What you will learn

- `00_list_envs.py`: which Gym tasks are registered by `aerial_lab`.
- `01_inspect_env_cfg.py`: how Isaac Lab resolves a task name into an environment config object.
- `02_step_env.py`: how a Gym-style Isaac Lab env is created and stepped.
- `03_trace_obs_reward_done.py`: how observation, reward, and done flags appear at runtime.
- `04_trace_apply_action.py`: how sampled actions become thrust commands inside `_pre_physics_step()` and `_apply_action()`.
- `05_trace_reset_flow.py`: how `terminated` and `truncated` flags lead into per-env resets.
- `06_trace_reward_terms.py`: how the named reward terms accumulate inside `self._episode_sums`.
- `07_trace_done_reasons.py`: how crash, drift, and time-out map into done flags.
- `08_trace_train_pipeline.py`: how task registration, env cfg resolution, runner cfg resolution, and `runner.learn(...)` fit together.
- `09_trace_policy_io.py`: what observation and action tensor shapes the policy sees.
- `10_trace_rollout_buffer.py`: how `step`, `iteration`, `epoch`, and `minibatch` become concrete numbers for PPO.
- `11_read_get_observations.py`: where `_get_observations()` builds the policy tensor.
- `12_read_get_rewards.py`: where `_get_rewards()` builds the reward terms.
- `13_read_get_dones.py`: where `_get_dones()` defines termination and truncation.
- `14_read_reset_idx.py`: where `_reset_idx()` resets robot state, goal state, and logs.

## Where This Maps Into Aerial Lab

- task registration:
  [source/aerial_lab/aerial_lab/tasks/__init__.py](../../source/aerial_lab/aerial_lab/tasks/__init__.py)
- one concrete direct env:
  [mini_quadcopter_env.py](../../source/aerial_lab/aerial_lab/tasks/direct/quadcopter/mini_quadcopter_env.py)
- RL training entry:
  [scripts/rsl_rl/train.py](../../scripts/rsl_rl/train.py)

## After this

Once these are comfortable, the next natural step is to trace one complete training task:

- env cfg selection
- env creation
- action application
- observation construction
- reward / done computation
- runner loop in `scripts/rsl_rl/train.py`
