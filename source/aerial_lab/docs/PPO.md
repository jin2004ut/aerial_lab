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
    # maxmum iteration, the whole training process will stop after exceed this number
    save_interval = 1000
    # save policy neural network every interval
    experiment_name = "beetle_omni"
    obs_groups = {
        "policy": ["policy"],
        # actor neural network, deployed in real-world
        "critic": ["critic"],
        # critic neural network, could utilize priviliage information
    }
```

Actor-Critic Training Framework Configuration
```python
policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # initial action noise std, this value will convergent to 0.1~0.3
        # this will influence the early exploration
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        # we can scale the observation in env, to make observation's value in similar range
        actor_hidden_dims=[256, 128, 128],
        # actor MLP size
        critic_hidden_dims=[256, 128, 128],
        # critic MLP size
        # the size of MLP associate to the capability of the MLP to handle knowledge
        activation="elu",
        # activate function for neural network
    )
```

PPO Training Algorithm Configuration
```python
algorithm = RslRlPpoAlgorithmCfg(
        # L = po
        value_loss_coef=1.0,
        # value loss coefficient
        use_clipped_value_loss=True,
        clip_param=0.2,
        # confine the update step of policy
        entropy_coef=0.0005,
        # entropy bouns coefficient,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,  # default 5.0e-4
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```
