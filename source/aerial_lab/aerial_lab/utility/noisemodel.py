from functools import partial

import torch


class NoiseModel:
    def __init__(self, noise_cfg: dict, device: torch.device | str, num_envs: int):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.params = {}
        for name, cfg in noise_cfg.items():
            dim = cfg["dim"]
            mean = torch.as_tensor(cfg.get("mean", 0.0), device=self.device, dtype=torch.float32).expand(dim)
            std = torch.as_tensor(cfg.get("std", 0.0), device=self.device, dtype=torch.float32).expand(dim)
            clip = torch.as_tensor(cfg.get("clip", float("inf")), device=self.device, dtype=torch.float32).expand(dim)
            self.params[name] = {
                "type": cfg["type"],
                "mean": mean.unsqueeze(0).expand(self.num_envs, dim).clone(),
                "std": std.unsqueeze(0).expand(self.num_envs, dim).clone(),
                "clip": clip.unsqueeze(0).expand(self.num_envs, dim).clone(),
            }
            setattr(self, f"{name}_apply", partial(self._add_noise, name=name))

    def apply(self, values: torch.Tensor, name: str) -> torch.Tensor:
        return self._add_noise(values, name)

    def _add_noise(self, values: torch.Tensor, name: str) -> torch.Tensor:
        cfg = self.params[name]
        shape = (self.num_envs, cfg["mean"].size(-1))
        if cfg["type"] == "uniform":
            noise = cfg["mean"] + cfg["std"] * (torch.rand(shape, device=self.device) * 2.0 - 1.0)
        elif cfg["type"] == "gaussian":
            noise = torch.normal(mean=cfg["mean"], std=cfg["std"], size=shape, device=self.device)
        else:
            raise ValueError(f"Unsupported noise type {cfg['type']}")

        if cfg["clip"] is not None:
            noise = torch.clamp(noise, min=-cfg["clip"], max=cfg["clip"])

        return values + noise


# noisemodel.py
if __name__ == "__main__":
    """Main function."""
    # run the main function
    noise_cfg = {
        "pos": {"type": "gaussian", "dim": 3, "mean": 0.0, "std": 0.1, "clip": 0.2},
        "vel": {"type": "uniform", "dim": 3, "mean": 0.0, "std": 0.05, "clip": 0.1},
    }
    num_envs = 11
    device = "cpu"
    noise_model = NoiseModel(noise_cfg, device, num_envs)

    pos = torch.zeros((num_envs, 3), device=device)
    vel = torch.zeros((num_envs, 3), device=device)

    noisy_pos = noise_model.apply(pos, "pos")
    noisy_vel = noise_model.apply(vel, "vel")

    print("Original pos:", pos)
    print("Noisy pos:", noisy_pos)
    print("Original vel:", vel)
    print("Noisy vel:", noisy_vel)
