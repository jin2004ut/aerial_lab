from isaaclab.app import AppLauncher

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app

import torch
from aerial_lab.utility.delaymodel import DelayModel, DelayModelCfg


def test_delay_model():
    print("=== Starting DelayModel Test ===")
    device = "cpu" # if torch.cuda.is_available() else "cpu"
    num_envs = 6
    dim = 2

    # --- Test Case 1: No Delay (min=0, max=0, jitter=0) ---
    print("\n--- Test Case 1: No Delay ---")
    cfg_no_delay = DelayModelCfg()
    cfg_no_delay.dim = dim
    cfg_no_delay.min_delay_steps = 0
    cfg_no_delay.max_delay_steps = 0
    cfg_no_delay.jitter_delay_steps = 0

    model_no_delay = DelayModel(cfg_no_delay, device, num_envs)

    # Input sequence: 1, 2, 3, 4, 5
    inputs = torch.arange(1, 6, device=device, dtype=torch.float32).unsqueeze(1).repeat(1, num_envs).unsqueeze(-1) # (5, num_envs, 1)

    print("Input sequence: [1, 2, 3, 4, 5]")
    for i in range(5):
        state = inputs[i]
        out = model_no_delay.update(state)
        print(f"Step {i+1}: Input={state[0,0].item()}, Output={out[0,0].item()}")
        # Expect output == input immediately
        assert out[0,0].item() == state[0,0].item(), f"Step {i+1} failed: Expected {state[0,0].item()}, got {out[0,0].item()}"
    print("Test Case 1 Passed!")

    # --- Test Case 2: Fixed Delay (min=2, max=2, jitter=0) ---
    print("\n--- Test Case 2: Fixed Delay (2 steps) ---")
    cfg_fixed = DelayModelCfg()
    cfg_fixed.dim = dim
    cfg_fixed.default = 0.0
    cfg_fixed.min_delay_steps = 2
    cfg_fixed.max_delay_steps = 2
    cfg_fixed.jitter_delay_steps = 0

    model_fixed = DelayModel(cfg_fixed, device, num_envs)

    # Expected behavior:
    # Step 1: In=1, Out=0 (buffer init)
    # Step 2: In=2, Out=0 (buffer init)
    # Step 3: In=3, Out=1 (delayed 1 arrives)

    inputs = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected_outputs = [0.0, 0.0, 1.0, 2.0, 3.0]

    for i, val in enumerate(inputs):
        state = torch.full((num_envs, dim), val, device=device)
        out = model_fixed.update(state)
        print(f"Step {i+1}: Input={val}, Output={out[0,0].item()}")
        assert out[0,0].item() == expected_outputs[i], f"Step {i+1} failed: Expected {expected_outputs[i]}, got {out[0,0].item()}"
    print("Test Case 2 Passed!")

    # --- Test Case 3: Variable Bias & Reset (min=1, max=3, jitter=0) ---
    print("\n--- Test Case 3: Variable Bias & Reset ---")
    cfg_var = DelayModelCfg()
    cfg_var.dim = dim
    cfg_var.default = 0.0
    cfg_var.min_delay_steps = 1
    cfg_var.max_delay_steps = 3
    cfg_var.jitter_delay_steps = 0

    model_var = DelayModel(cfg_var, device, num_envs)

    print(f"Biases after init: {model_var.bias.cpu().tolist()}")

    # Check if biases are within range
    assert torch.all(model_var.bias >= 1) and torch.all(model_var.bias <= 3)

    # Reset specific envs
    reset_ids = [0, 1]
    old_biases = model_var.bias.clone()
    model_var.reset(reset_ids)
    new_biases = model_var.bias

    print(f"Biases after reset envs {reset_ids}: {new_biases.cpu().tolist()}")

    # Check if non-reset envs kept their bias (might be same by chance, but logic holds)
    # Check if buffer is cleared for reset envs
    assert torch.all(model_var.buffer[reset_ids] == 0), "Buffer not cleared after reset"
    print("Test Case 3 Passed!")

    # --- Test Case 4: Jitter (min=1, max=1, jitter=2) ---
    print("\n--- Test Case 4: Jitter Effect ---")
    # With base delay 1 and jitter 2, delay can be 1, 2, or 3.
    # This means output order might change or hold (Zero Order Hold).
    cfg_jitter = DelayModelCfg()
    cfg_jitter.dim = dim
    cfg_jitter.default = 0.0
    cfg_jitter.min_delay_steps = 0
    cfg_jitter.max_delay_steps = 2  # Large enough buffer
    cfg_jitter.jitter_delay_steps = 1

    model_jitter = DelayModel(cfg_jitter, device, num_envs)
    print(f"Biases for jitter test: {model_jitter.bias.cpu().tolist()}")
    print(f"Jitter caps for jitter test: {model_jitter.jitter_cap.cpu().tolist()}")
    inputs = torch.arange(1, 11, device=device, dtype=torch.float32).unsqueeze(1).repeat(1, num_envs).unsqueeze(-1) # (10, num_envs, 1)
    for i, val in enumerate(inputs):
        state = val.expand(-1, dim).clone()
        out = model_jitter.update(state)
        print(f"Step {i+1}: Input={val[0,0].item()}")
        print(f"         Output={out}")
        import ipdb; ipdb.set_trace()
    print("Test Case 4 Completed! (Manual verification of jitter behavior may be needed)")


if __name__ == "__main__":
    test_delay_model()
    simulation_app.close()