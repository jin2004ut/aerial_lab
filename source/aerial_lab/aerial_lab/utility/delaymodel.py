from collections.abc import Sequence
from functools import partial
from numpy import double
import torch


class DelayModelCfg:
    name: str = "none"
    dim: int = 0
    default: float = 0.0
    max_delay_steps: int = 0
    min_delay_steps: int = 0
    jitter_delay_steps: int = 0


class DelayModel:
    """
    A class to simulate signal delay and jitter for multiple environments using PyTorch tensors.
    It maintains a buffer of past states and applies a randomized delay logic.
    """

    def __init__(self, cfg: DelayModelCfg, device: torch.device | str, num_envs: int):
        """
        Initialize the DelayModel.

        Args:
            cfg (DelayModelCfg): Configuration for the delay model.
            device (torch.device | str): Device to store tensors on.
            num_envs (int): Number of environments.
        """
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        self.dim = cfg.dim

        # Buffer size is max_delay + 1 to accommodate the current step up to the max delay
        self.buffer_len = cfg.max_delay_steps + 1

        # StateBuffer: (num_envs, buffer_len, dim)
        # Stores the history/future queue of states
        self.buffer = torch.zeros((self.num_envs, self.buffer_len, self.dim), device=self.device)

        # Bias: Base delay for each environment
        self.bias = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Jitter: Max jitter range for each environment
        self.jitter_cap = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Initialize all environments
        self.reset()

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None):
        """
        Reset the delay model for specified environments.
        Resamples bias and jitter parameters and clears the buffer.

        Args:
            env_ids (torch.Tensor | Sequence[int] | None): Indices of environments to reset.
                                                           If None, resets all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids, device=self.device)

        num_reset = len(env_ids)

        # 1. Reset Buffer for these envs to zeros
        self.buffer[env_ids] = self.cfg.default

        # 2. Sample bias uniformly from [min_delay_steps, max_delay_steps]
        if self.cfg.max_delay_steps > self.cfg.min_delay_steps:
            self.bias[env_ids] = torch.randint(
                self.cfg.min_delay_steps,
                self.cfg.max_delay_steps + 1,
                (num_reset,),
                device=self.device,
            )
        else:
            self.bias[env_ids] = self.cfg.min_delay_steps

        # 3. Sample jitter cap uniformly from [0, jitter_delay_steps]
        # This represents the maximum jitter specific to this environment instance
        if self.cfg.jitter_delay_steps > 0:
            self.jitter_cap[env_ids] = torch.randint(
                0, self.cfg.jitter_delay_steps + 1, (num_reset,), device=self.device
            )
        else:
            self.jitter_cap[env_ids] = 0

    def update(self, state: torch.Tensor) -> torch.Tensor:
        """
        Update the delay buffer with the new state and retrieve the delayed state.

        Logic:
        1. Calculate current write delay = bias + random(0, jitter).
        2. Overwrite buffer from [delay:] with the new state (Zero-Order Hold effect).
        3. Retrieve the state at the front of the buffer (current output).
        4. Shift buffer (time progression) for the next step.

        Args:
            state (torch.Tensor): New state/signal of shape (num_envs, dim).

        Returns:
            torch.Tensor: Delayed state of shape (num_envs, dim).
        """
        # --- 1. Calculate current delay steps ---
        # Random jitter for this specific step: randint(0, jitter_cap + 1)
        r = torch.rand(self.num_envs, device=self.device)
        current_jitter = (r * (self.jitter_cap + 1)).long()

        # Total delay = bias + jitter
        write_indices = self.bias + current_jitter

        # Clamp to ensure we don't exceed buffer size
        write_indices = torch.clamp(write_indices, max=self.cfg.max_delay_steps)

        # --- 2. Update Buffer (Overwrite future) ---
        # We want: buffer[i, write_indices[i]:, :] = state[i]
        # This implies that the new state arrives at 'write_indices' and persists 
        # until a newer state overwrites it (Zero Order Hold).

        # Create a time index grid: (1, buffer_len)
        time_idx = torch.arange(self.buffer_len, device=self.device).unsqueeze(0)

        # Create a mask: (num_envs, buffer_len)
        # True where time_idx >= write_indices
        mask = time_idx >= write_indices.unsqueeze(1)

        # Expand mask to match dim: (num_envs, buffer_len, dim)
        mask = mask.unsqueeze(-1).expand(-1, -1, self.dim)

        # Apply update: Where mask is True, use new state; otherwise keep existing buffer
        self.buffer = torch.where(mask, state.unsqueeze(1), self.buffer)

        # --- 3. Retrieve the output (front of the buffer) ---
        # Now that the buffer is updated, index 0 contains the correct state for the current time step.
        # If delay was 0, we just wrote to index 0, so we get the current state back immediately.
        delayed_state = self.buffer[:, 0].clone()

        # --- 4. Shift the buffer (Time Step) ---
        # Move everything one step closer to the front (index 0) for the NEXT step.
        # buffer[t] becomes buffer[t-1]
        self.buffer[:, :-1] = self.buffer[:, 1:].clone()

        return delayed_state
