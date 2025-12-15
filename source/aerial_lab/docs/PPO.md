### RSL RL PPO

Basic Simulation Environment
```python
class BeetleOmniPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48      
    # sample steps for per rollout, it is related to sim_dt and batch size.
    # the long of a sampled trajectory is: sim_dt * num_steps_per_env
    # which is robot specific, we want a normal trajectory duration, such as: 0.48s 
    # the batch size for PPO update is : num_envs * num_steps_per_env
    max_iterations = 20001
    save_interval = 1000
    experiment_name = "beetle_omni"
```