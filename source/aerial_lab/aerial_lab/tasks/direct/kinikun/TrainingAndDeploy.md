# RL Policy: Training and Deployment Tutorial

This tutorial explains the general workflow for training an RL policy (Isaac Lab + rsl_rl PPO)
and deploying it, in two parts:

1. **Training** — how to judge that a policy has been trained successfully.
2. **Deployment** — how to take a trained policy to another simulator (sim2sim) and then to
   the real robot (sim2real).

The overall workflow is:

```
train (Isaac Lab + rsl_rl)          logs/rsl_rl/<experiment>/<run>/model_*.pt
        │  watch TensorBoard: has training converged?  (Part 1)
        ▼
play / evaluate in the training sim
        │  play.py also exports  exported/policy.pt (TorchScript) + policy.onnx
        ▼
sim2sim: run the exported policy in a *different* simulator / your deploy code
        │  verify the deploy code reproduces the training-side interface  (Part 2.2)
        ▼
sim2real: run on the real robot, log everything, analyze the gap        (Part 2.3)
        │
        └── feed findings (noise, delay, dynamics mismatch) back into the training env
```

Basic commands (run from the repo root):

```bash
# train (headless is much faster)
python scripts/rsl_rl/train.py --task=<TASK_NAME> --headless

# monitor training
tensorboard --logdir=logs/rsl_rl/<EXPERIMENT_NAME>/

# play the latest checkpoint (also exports policy.pt / policy.onnx)
python scripts/rsl_rl/play.py --task=<TASK_NAME> --num_envs=32
```

---

## Part 1 — When is a policy "trained successfully"?

There is no single number that says "done". You judge it from a small set of TensorBoard
curves **plus** task-specific metrics **plus** actually watching the policy run. Go through
the following checks in order.

### 1.1 `Train/mean_reward` has converged — and converged to a *good* value

The first-order signal is `Train/mean_reward`: when it has plateaued and no longer grows,
the optimization has converged.

**Converged is not the same as good.** The reward value must be judged against the reward
function you designed — its terms, physical units (dimensions), and weights. Compare the
plateau value against the *theoretical maximum* of your reward:

- reward close to the theoretical maximum → the task is actually solved;
- reward converged but far below the maximum → the policy got stuck in a local optimum, or
  the task/reward design makes the maximum unreachable.

For example, if the reward is a pure sum of penalties (negative terms), the theoretical
maximum is 0. It also pays to convert the plateau value back into a physical quantity: from
the episode length and the dominant reward term you can estimate e.g. what average tracking
error (in rad or m) the current reward corresponds to. This back-of-the-envelope conversion
turns an abstract reward number into something you can actually judge.

### 1.2 `Policy/mean_noise_std` must settle low

PPO's actor outputs a Gaussian: mean action + a learned standard deviation (initialized by
`init_noise_std` in the agent config). `Policy/mean_noise_std` in TensorBoard is that learned
std — i.e. **how much the policy is still exploring / how much variance its actions have**.

- A healthy run: `mean_noise_std` decays from its initial value and settles at a small value
  (ideally approaching 0, in practice a small fraction of the action range).
- Why it matters for deployment: at play/deploy time the **mean** action is used (no
  sampling — see Part 2.1). If `mean_noise_std` is still large, the reward you saw during
  training was obtained *with* noise, and the deterministic deployed policy can behave
  noticeably differently.
- A `mean_noise_std` that stays high or **blows up** ("takes off") means training is
  unstable — see 1.3.

### 1.3 Keeping training stable: `entropy_coef` and `learning_rate`

Assuming the reward design is sound, the two main PPO knobs against instability
(`mean_noise_std` blowing up, reward collapsing) are:

- **`entropy_coef`**: exploration strength. The entropy bonus actively pushes the action
  distribution to stay wide, so larger values mean more aggressive exploration at every
  update. If `mean_noise_std` won't come down, lower it; if the policy converges prematurely
  to a bad local optimum, raise it.
- **`learning_rate`**: the *initial* step size. With `schedule="adaptive"`, rsl_rl adjusts it
  during training to keep the KL divergence between policy updates near `desired_kl`. The
  update rule in `rsl_rl/algorithms/ppo.py` is:

  ```python
  if kl_mean > self.desired_kl * 2.0:
      self.learning_rate = max(1e-5, self.learning_rate / 1.5)
  elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
      self.learning_rate = min(1e-2, self.learning_rate * 1.5)
  ```

  Note the floor: the learning rate is clamped to a minimum of `1e-5` (fixed inside rsl_rl).
  **In a stable run the adaptive learning rate should stay above this floor.** If
  `Loss/learning_rate` in TensorBoard is pinned at `1e-5`, the optimizer is constantly making
  updates that are too large (KL keeps exceeding `2 × desired_kl`) — a red flag for the reward
  scaling or the exploration setting.

Because `learning_rate` is managed by the adaptive schedule, in practice **you mainly tune
`entropy_coef`**, and you *monitor* the two curves `Loss/entropy` (exploration strength
actually being applied) and `Loss/learning_rate` (what the schedule settled on).

### 1.4 Task-specific metrics — the reward can lie, physical metrics don't

A converged reward can still hide problems (e.g. the policy trades tracking accuracy for
smoothness because of mis-tuned weights). Therefore, always log extra quantities from your
environment (via `self.extras["log"]` on episode reset):

- **Physical metrics** (e.g. final tracking error in rad/m): these are the numbers to quote
  when someone asks "how good is the policy" — they are physical quantities, independent of
  reward weights.
- **Per-term reward sums** (one scalar per reward component): use these to see *which* term
  dominates. If a regularization penalty (velocity, action rate, ...) is comparable to the
  main task term, the penalties are too strong and the policy is being paid to not do the
  task.

When you add a new task, always log metrics like these — it is the cheapest debugging tool
you will ever add.

### 1.5 Final check: watch it run

Numbers pass ≠ done. Run the policy with `play.py` and look at it. If possible, evaluate on
commands/targets that are *harder or different* from the training distribution (e.g. a
continuously moving target instead of the fixed per-episode targets used in training) — if
the policy still works, it has generalized beyond memorizing the training setup.

**Checklist summary — a policy is "trained successfully" when:**

1. `Train/mean_reward` has plateaued, at a value close to the theoretical max of your reward;
2. `Policy/mean_noise_std` has decayed to a small stable value (and never blew up);
3. `Loss/learning_rate` stayed off the `1e-5` floor and `Loss/entropy` looks stable;
4. physical metrics (tracking error, ...) meet the spec you actually care about;
5. it visibly works in `play.py`, ideally on conditions it never trained on.

---

## Part 2 — Deploying a trained policy: sim2sim, then sim2real

### 2.1 What exactly is "the policy" you deploy?

Running `play.py` once exports the trained actor into two standalone files:

```
logs/rsl_rl/<experiment>/<run>/exported/policy.pt    # TorchScript — load with torch.jit.load, no Isaac/rsl_rl needed
logs/rsl_rl/<experiment>/<run>/exported/policy.onnx  # ONNX — load with onnxruntime, C++/embedded friendly
```

Two properties of the exported policy matter for deployment:

- Its `forward()` is `actor(normalizer(obs))` — the **deterministic mean action**, no
  exploration noise. (If observation normalization was disabled in the agent config, the
  normalizer is an identity and raw observations go straight into the network — the deploy
  side must then feed observations in exactly the training units.)
- It is **just a network**: `obs (float32 vector) → action (float32 vector)`, nothing more.
  Everything else — how the observation vector is built, what the action values mean, how
  often the network is called — is *your* contract to reproduce. That contract is the entire
  content of Section 2.2, and getting it wrong is the #1 reason "the policy works in sim but
  not when deployed".

### 2.2 The interface contract (this is where deployments die)

When moving the policy anywhere — another simulator or the real robot — the following must
match the training environment **exactly**. Compare the training code and the deploy code
side by side, item by item:

- **Observation vector**: the physical quantities, their **order** in the vector, their
  **units/dimensions**, and any **scaling/normalization**. Write out an explicit table
  (index → quantity → unit → range) from the training env's `_get_observations`, and
  implement the deploy side against that table.
- **Action mapping**: the raw network output usually goes through post-processing in the
  training env (clamping, affine mapping from normalized [−1, 1] to physical commands,
  channel ordering). The deploy side must apply the **same** post-processing — reproduce
  what the training code *does*, not what variable names suggest.
- **Control frequency**: the policy step rate in training is `decimation × sim_dt`. The
  deploy loop must call the policy at the same rate. A policy trained at one rate and
  deployed at another sees a world with different effective delays and action-rate dynamics —
  closed-loop behavior changes completely. Measure the achieved loop rate; don't assume it.
- **Initial state**: note what state (joint positions, action history buffers, ...) the
  training env resets to, and start the deployed system from the same condition before
  enabling the policy.

### 2.3 Sim2sim — verify the *deploy code*, not the policy

Sim2sim means running the exported policy inside a second, independent implementation
(another simulator, or your actual robot-side control code running against a simulated
plant). At this stage the policy is a known-good constant; **what you are testing is whether
the deploy code implements the Section 2.2 contract correctly.**

The single most effective check: **feed the same observation vector to both sides and compare
the action outputs.** If the contract is implemented correctly they must agree to float
precision; any systematic difference means a bug in observation construction, ordering,
scaling, or action post-processing.

Concretely:

1. On the training side, run `play.py` and dump a few hundred `(obs, action)` pairs from the
   inference loop (add a `torch.save`/`np.save` around the `actions = policy(obs)` call).
2. On the deploy side, load the exported policy and replay the same observations:

   ```python
   import numpy as np
   import onnxruntime as ort

   sess = ort.InferenceSession("exported/policy.onnx")
   obs, ref_actions = np.load("obs.npy"), np.load("actions.npy")   # dumped from play.py
   for o, a_ref in zip(obs, ref_actions):
       a = sess.run(None, {"obs": o[None].astype(np.float32)})[0][0]
       assert np.allclose(a, a_ref, atol=1e-5), (a, a_ref)
   ```

   (Same idea with `torch.jit.load("exported/policy.pt")` if the deploy stack is PyTorch.)
3. Only after this passes, close the loop in the second simulator and check the *behavior*:
   same task performance as in Part 1, and confirm the loop really runs at the training
   control rate.

If the open-loop action comparison matches but the closed-loop behavior differs, the
difference is in the *plant* (the second simulator's dynamics vs. the training sim), not in
your deploy code — that is already a sim2real-type gap, handled next.

### 2.4 Sim2real — treat it like debugging any other controller

On the real robot, an RL policy is just another closed-loop controller, and you debug it the
same way: **log every state variable involved in the loop, plot the curves, and analyze the
features.** Do not guess from watching the robot.

Log at minimum, all timestamped: the sensor readings, the observation vector actually fed to
the policy, the raw policy output, and the commands after mapping. Then compare these curves
against the same signals recorded in simulation to identify the sim2real gap, and **feed what
you find back into the training environment** (this loop — deploy, measure, adapt the sim,
retrain — *is* the sim2real workflow).

The two gaps that matter most in practice:

- **Delay.** Sensor latency, actuator response, communication, and compute all add loop delay
  that the training sim may not model. Delay in a closed loop erodes phase margin; the
  visible symptom is **oscillation or outright divergence** of a policy that was perfectly
  stable in sim. Diagnose it by measuring the lag between a commanded step and the measured
  response. Fix it on the training side: model the delay in the env (delay the observations
  or actions by the measured number of steps) and retrain.
- **Noise.** Real sensors are noisy; observations derived by differentiation (e.g. velocity
  from positions at high rate) are especially noisy. Noisy observations produce
  **jittery/chattering actions**, which real actuators then amplify or low-pass in
  hardware-specific ways. Diagnose by comparing the spectrum of real vs. sim observations.
  Fix by filtering the real signals and/or adding matching observation noise in training so
  the policy learns to be robust to it (an action-rate penalty in the reward also helps
  suppress chattering — raise its weight if real actions are too jumpy).

Beyond these two, run through the same contract checklist as 2.2 on real hardware: sensor
zero/sign conventions vs. the sim (is the real "zero" the same pose as the sim's default
state?), unit conversions in the driver layer, and the actual achieved loop rate under real
compute load.

**Safety note:** always start with limits — clamp commands conservatively, keep an e-stop,
and run the first trials well inside the workspace. The policy has never seen states or
targets outside its training distribution, so do not command them on hardware either.
