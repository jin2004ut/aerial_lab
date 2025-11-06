# rotor.py
from pyparsing import Enum
import torch
import torch.nn as nn


class SecondOrderSystem:
    """
    Second order system dynamics simulator.

    Args:
        device (torch.device): Torch device (e.g., torch.device("cuda")).
        wn (float): Natural frequency.
        zeta (float): Damping ratio.
        state (torch.Tensor): State tensor.
    """

    def __init__(self, device: torch.device, wn: float, zeta: float):
        self.device = device
        self.wn = wn
        self.zeta = zeta
        self.state = torch.zeros(2, device=self.device)  # [x, xdot]

    def step(self, input: torch.Tensor, dt: float):
        """
        Step the second order system.
        x¨+2ζωnx˙+ωn2x=ωn2u
        u <=> x, is the same physical variable
        """
        x, xdot = self.state
        xddot = (
            -2 * self.zeta * self.wn * xdot - self.wn**2 * x + self.wn**2 * input
        )
        xdot += xddot * dt
        x += xdot * dt
        self.state = torch.tensor([x, xdot], device=self.device)
        return x


class FirstOrderSystem:
    """
    First order system dynamics simulator.

    Args:
        device (torch.device): Torch device (e.g., torch.device("cuda")).
        tau (float): Time constant.
        state (torch.Tensor): State tensor.
    """

    def __init__(self, device: torch.device, tau: float):
        self.device = device
        self.tau = tau
        self.state = torch.zeros(1, device=self.device)  # [x]

    def step(self, input: torch.Tensor, dt: float):
        """
        Step the first order system.
        τx˙+x=u
        u <=> x, is the same physical variable
        """
        x = self.state
        xdot = (-x + input) / self.tau
        x += xdot * dt
        self.state = x
        return x


class Rotor:
    """
    Rotor dynamics simulator.

    Args:
        device (torch.device): Torch device (e.g., torch.device("cuda")).
        rotor_ids (torch.Tensor): optional unique IDs.
        rotor_directions (torch.Tensor): Rotation direction of each rotor (align with Z-axis).
        thrust_coeff (torch.Tensor): Coefficient mapping thrust command to force.
        torque_coeff (torch.Tensor): Coefficient mapping thrust command to torque.
        KF (torch.Tensor): Thrust coefficient for each rotor.
        KM (torch.Tensor): Torque coefficient for each rotor.
        vel (torch.Tensor): Rotor angular velocities.
        max_vel (torch.Tensor): Maximum rotor angular velocities.
        thrust (torch.Tensor, shape: (1, 3)): Thrust force vector.
        torque (torch.Tensor, shape: (1, 3)): Torque vector.
    """
    def __init__(
        self,
        device: torch.device,
        dt: float,
        id: torch.Tensor,
        direction: torch.Tensor,
        thrust_coeff: torch.Tensor,
        torque_coeff: torch.Tensor,
        max_vel: float,
        max_foc: float,
        vel_wn: float | None = 0.0,
        vel_zeta: float | None = 0.0,
        vel_tau: float | None = 0.0,
        foc_zeta: float | None = 0.0,
        foc_wn: float | None = 0.0,
        foc_tau: float | None = 0.0,
    ):
        self.device = device
        self.dt = dt
        self.id = id
        self.thrust_to_torque_ratio = torque_coeff / thrust_coeff
        self.rotor_direction = direction.to(self.device)
        self.thrust_coeff = thrust_coeff.to(self.device)
        self.torque_coeff = torque_coeff.to(self.device)

        self.KF = thrust_coeff.to(self.device)
        self.KM = torque_coeff.to(self.device)

        self.vel = torch.zeros(1, device=self.device)
        self.velsys = SecondOrderSystem(
            device,
            wn=0.0 if vel_wn is None else vel_wn,
            zeta=0.0 if vel_zeta is None else vel_zeta
        )
        self.max_vel = max_vel

        self.foc = torch.zeros(1, device=self.device)
        self.focsys = SecondOrderSystem(
            device,
            wn=0.0 if foc_wn is None else foc_wn,
            zeta=0.0 if foc_zeta is None else foc_zeta
        )
        self.max_foc = max_foc

        self.thrust = torch.zeros((1, 3), device=self.device)
        self.torque = torch.zeros((1, 3), device=self.device)

    def cmdvel(self, vel_cmd: torch.Tensor):
        self.vel = self.velsys.step(vel_cmd, dt=self.dt)
        self.vel = torch.clamp(self.vel, 0.0, self.max_vel)
        self.thrust[:, 2] = self.KF * self.vel**2
        self.torque[:, 2] = self.rotor_direction * self.KM * self.vel**2

    def cmdfoc(self, foc_cmd: torch.Tensor):
        self.foc = self.focsys.step(foc_cmd, dt=self.dt)
        self.foc = torch.clamp(self.foc, 0.0, self.max_foc)
        self.thrust[:, 2] = self.foc
        self.torque[:, 2] = self.rotor_direction * self.thrust_to_torque_ratio * self.foc


class RotorGroup(nn.Module):
    """
    Rotor dynamics simulator for multi-rotor systems.

    Args:
        rotor_directions (torch.Tensor): shape (num_envs, num_rotors)
            Rotation direction of each rotor (+1 for CCW, -1 for CW).
        thrust_coeff (float): Coefficient mapping thrust command to force.
        torque_coeff (float): Coefficient mapping thrust command to torque.
        rotor_ids (torch.Tensor): shape (num_rotors,) optional unique IDs.
        device (torch.device): Torch device (e.g., torch.device("cuda")).
    """
    dir: torch.Tensor
    kf: torch.Tensor
    km: torch.Tensor
    max_vel: torch.Tensor
    max_foc: torch.Tensor
    torque_thrust_ratio: torch.Tensor
    vel_wn: torch.Tensor
    vel_zeta: torch.Tensor
    vel_tau: torch.Tensor
    vel: torch.Tensor
    vel_dot: torch.Tensor
    foc_wn: torch.Tensor
    foc_zeta: torch.Tensor
    foc_tau: torch.Tensor
    foc: torch.Tensor
    foc_dot: torch.Tensor

    def __init__(
        self,
        cfg,
        devices: torch.device,
        num_envs: int,
        rotor_ids: slice | torch.Tensor,
        rotor_names: list[str],
        rotor_directions: slice | torch.Tensor,
    ):
        super().__init__()
        self.device = devices
        self.num_envs = num_envs
        self.cfg = cfg
        if cfg["rotor_num"] != len(rotor_names):
            raise ValueError(
                f"rotor_num {cfg['rotor_num']} does not match length of rotor_names {len(rotor_names)}"
            )
        self.num_rotors = cfg["rotor_num"]
        self.names = rotor_names
        self.ids = rotor_ids
        self.dt = cfg["dt"]
        self.mode = cfg["mode"]

        self.register_buffer("dir", self._expand_param(rotor_directions, "rotor_directions", dtype=torch.float32))
        self.register_buffer("kf", self._expand_param(cfg["thrust_coeff"], "thrust_coeff", dtype=torch.float32))
        self.register_buffer("km", self._expand_param(cfg["torque_coeff"], "torque_coeff", dtype=torch.float32))
        self.register_buffer("max_vel", self._expand_param(cfg["max_vel"], "max_vel", dtype=torch.float32))
        self.register_buffer("max_foc", self._expand_param(cfg["max_foc"], "max_foc", dtype=torch.float32))
        torque_thrust_ratio = torch.where(
            self.kf.abs() > 1e-6, self.km / torch.clamp(self.kf, min=1e-6), torch.zeros_like(self.km)
        )
        self.register_buffer("torque_thrust_ratio", torque_thrust_ratio)

        # # # #
        if self.mode not in ["vel", "foc"]:
            raise ValueError(f"Unknown rotor mode {cfg['mode']}, must be 'vel', 'foc'")

        if self.mode == "vel":
            self.register_buffer("vel_wn", self._expand_param(cfg["vel_wn"], "vel_wn", dtype=torch.float32))
            self.register_buffer("vel_zeta", self._expand_param(cfg["vel_zeta"], "vel_zeta", dtype=torch.float32))
            # self.register_buffer("vel_tau", self._expand_param(cfg["vel_tau"], "vel_tau", dtype=torch.float32))
        elif self.mode == "foc":
            self.register_buffer("foc_wn", self._expand_param(cfg["foc_wn"], "foc_wn", dtype=torch.float32))
            self.register_buffer("foc_zeta", self._expand_param(cfg["foc_zeta"], "foc_zeta", dtype=torch.float32))
            # self.register_buffer("foc_tau", self._expand_param(cfg["foc_tau"], "foc_tau", dtype=torch.float32))

        self.register_buffer("vel", torch.zeros((self.num_envs, self.num_rotors), device=self.device))
        self.register_buffer("vel_dot", torch.zeros((self.num_envs, self.num_rotors), device=self.device))
        self.register_buffer("foc", torch.zeros((self.num_envs, self.num_rotors), device=self.device))
        self.register_buffer("foc_dot", torch.zeros((self.num_envs, self.num_rotors), device=self.device))

        self.requires_grad_(False)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset the rotor group.

        This method resets the state of the rotors for the specified environments.

        Args:
            env_ids: The environment ids to reset the rotor group for. Defaults to None,
                in which case all environments are considered.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.vel[env_ids, :] = 0.0
        self.vel_dot[env_ids, :] = 0.0
        self.foc[env_ids, :] = 0.0
        self.foc_dot[env_ids, :] = 0.0

    def _expand_param(self, value, name: str, dtype=torch.float32) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self.device, dtype=dtype)
        if tensor.ndim == 0:
            tensor = tensor.expand(self.num_envs, self.num_rotors).clone()
        elif tensor.ndim == 1:
            if tensor.shape[0] != self.num_rotors:
                raise ValueError(f"{name} length {tensor.shape[0]} != num_rotors {self.num_rotors}")
            tensor = tensor.view(1, self.num_rotors).expand(self.num_envs, -1).clone()
        elif tensor.shape != (self.num_envs, self.num_rotors):
            raise ValueError(f"{name} must have shape ({self.num_envs}, {self.num_rotors})")
        else:
            tensor = tensor.clone()
        return tensor

    def stepvel(self, vel_cmd: torch.Tensor):
        """Step the rotor group velocity commands.

        Args:
            vel_cmd: The desired rotor velocities, shape (num_envs, num_rotors).
        """
        # # # # second order system
        vel_cmd = torch.clamp(vel_cmd, min=torch.zeros_like(self.max_vel), max=self.max_vel)
        wn = self.vel_wn
        zeta = self.vel_zeta
        x = self.vel
        xdot = self.vel_dot
        wn_sq = wn * wn
        xddot = -2.0 * zeta * wn * xdot - wn_sq * x + wn_sq * vel_cmd
        xdot = xdot + xddot * self.dt
        x = torch.clamp(x + xdot * self.dt, min=torch.zeros_like(self.max_vel), max=self.max_vel)
        self.vel.copy_(x)
        self.vel_dot.copy_(xdot)

        # # # # first order system
        # vel_cmd = torch.clamp(vel_cmd, min=torch.zeros_like(self.max_vel), max=self.max_vel)
        # tau = self.vel_tau
        # x = self.vel
        # xdot = self.vel_dot
        # xdot = (-x + vel_cmd) / tau
        # x = torch.clamp(x + xdot * self.dt, min=torch.zeros_like(self.max_vel), max=self.max_vel)
        # self.vel.copy_(x)
        # self.vel_dot.copy_(xdot)
        return x

    def stepfoc(self, foc_cmd: torch.Tensor):
        """Step the rotor group force commands.

        Args:
            foc_cmd: The desired rotor forces, shape (num_envs, num_rotors).
        """
        foc_cmd = torch.clamp(foc_cmd, min=torch.zeros_like(self.max_foc), max=self.max_foc)
        wn = self.foc_wn
        zeta = self.foc_zeta
        x = self.foc
        xdot = self.foc_dot
        wn_sq = wn * wn
        xddot = -2.0 * zeta * wn * xdot - wn_sq * x + wn_sq * foc_cmd
        xdot = xdot + xddot * self.dt
        x = torch.clamp(x + xdot * self.dt, min=torch.zeros_like(self.max_foc), max=self.max_foc)
        self.foc.copy_(x)
        self.foc_dot.copy_(xdot)

        # # # # first order system
        # foc_cmd = torch.clamp(foc_cmd, min=torch.zeros_like(self.max_foc), max=self.max_foc)
        # tau = self.foc_tau
        # x = self.foc
        # xdot = self.foc_dot
        # xdot = (-x + foc_cmd) / tau
        # x = torch.clamp(x + xdot * self.dt, min=torch.zeros_like(self.max_foc), max=self.max_foc)
        # self.foc.copy_(x)
        # self.foc_dot.copy_(xdot)
        return x

    def forward(self, cmd: torch.Tensor):
        if cmd.shape != (self.num_envs, self.num_rotors):
            raise ValueError(f"cmd shape {cmd.shape} != ({self.num_envs}, {self.num_rotors})")
        if self.mode == "vel":
            self.stepvel(cmd)
            thrusts = self.kf * self.vel * self.vel
        elif self.mode == "foc":
            self.stepfoc(cmd)
            thrusts = self.foc
        else:
            raise ValueError(f"Unknown rotor mode {self.mode}")

        forces = torch.zeros((self.num_envs, self.num_rotors, 3), device=self.device)
        forces[:, :, 2] = thrusts  # (N_env, N_rotor, 3)
        torques = torch.zeros((self.num_envs, self.num_rotors, 3), device=self.device)
        torques[:, :, 2] = self.dir * self.torque_thrust_ratio * thrusts  # (N_env, N_rotor, 3)
        return forces, torques


# debug.py
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import math
    # run the main function
    cfg = {
        "rotor_num": 4,
        "dt": 0.01,
        "mode": "vel",
        "thrust_coeff": 3.2e-3,
        "torque_coeff": 1.4e-3,
        "max_vel": 200.0,
        "max_foc": 6.0,
        "vel_wn": 1.0,
        "vel_zeta": 0.8,
        "foc_wn": 1.0,
        "foc_zeta": 0.8,
    }

    rotor_directions = torch.tensor([1.0, -1.0, -1.0, 1.0])
    group = RotorGroup(
        cfg=cfg,
        devices=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_envs=16,
        rotor_ids=torch.arange(4),
        rotor_names=["FL", "FR", "RL", "RR"],
        rotor_directions=rotor_directions,
    )
    group.reset()

    print(f"{'='*10}Parameter Reset Debugging{'='*10}")
    vel_cmd = 80.0 + 50.0 * torch.randn(16, 4, device=group.device)
    forces, torques = group(vel_cmd)
    print("forces shape:", forces.shape)
    print("torques shape:", torques.shape)
    print("mean thrust per env:", forces[..., 2].mean(dim=-1))
    group.kf[2, :] = 0.0
    vel_cmd = 80.0 + 50.0 * torch.randn(16, 4, device=group.device)
    forces, torques = group(vel_cmd)
    print("forces shape after setting kf[2,:]=0:", forces.shape)
    print("torques shape after setting kf[2,:]=0:", torques.shape)
    print("mean thrust per env after setting kf[2,:]=0:", forces[..., 2].mean(dim=-1))

    print(f"{'='*10}Direction Debugging{'='*10}")
    print("Rotor directions:", group.dir[3, :])
    print("Torque ", torques[3, :, :])
    print("Thrust", forces[3, :, :])

    print(f"{'='*10}Velocity Response Debugging{'='*10}")
    sim_duration = 15.0
    steps = int(sim_duration / cfg["dt"])
    target_env, target_rotor = 0, 0

    cmd_history = []
    foc_history = []
    vel_history = []
    time_axis = torch.arange(steps, device=group.device) * cfg["dt"]

    for step in range(steps):
        foc_cmd = torch.full((16, 4), 80.0, device=group.device) + 50.0 * math.sin(2 * step * cfg["dt"])
        forces, torques = group.forward(foc_cmd)
        cmd_history.append(foc_cmd[target_env, target_rotor].item())
        actual_vel = group.vel[target_env, target_rotor].item()
        vel_history.append(actual_vel)
        foc_history.append(forces[target_env, target_rotor, 2].item())
        print(
            f"Step {step}: Time {time_axis[step]:.2f}s foc_cmd={foc_cmd[target_env, target_rotor]:.2f}, "
            f"vel_val={actual_vel:.2f}, foc_val={forces[target_env, target_rotor, 2]:.2f}"
        )

    print("foc_history:", len(foc_history))
    print("cmd_history:", len(cmd_history))
    print("vel_history:", len(vel_history))

    plt.figure()
    plt.plot(time_axis.cpu().numpy(), cmd_history, label="foc_cmd [0,0]")
    plt.plot(time_axis.cpu().numpy(), vel_history, label="vel_val [0,0]")
    plt.plot(time_axis.cpu().numpy(), foc_history, label="foc_val [0,0]")
    plt.xlabel("Time [s]")
    plt.ylabel("Rotor speed")
    plt.title("Rotor (env 0, rotor 0) velocity response")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # close sim app
