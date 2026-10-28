"""Bench-identified model of the AK80-9 (custom KV50) prosthetic motor under MIT control.

Parameters come from the characterization in TMotorCANControl
(src/TMotorCANControl/test/characterization, model_params.json fitted 2026-10-28 on 122 runs).

Frames: "motor" is the AK80-9 output shaft (after its internal 9:1 gear), where the model was
identified and where the MIT protocol works (p_des, kp, kd, the reply's p, v and torque).
The simulated joint sits after the belt: theta_motor = transmission_ratio * theta_joint and
tau_joint = transmission_ratio * tau_motor.

Per physics step (dt = 1 / sim fps), with the command already on the wire, in the motor frame:

    tau_cmd = clip(kp * (p_des - theta) - kd * omega, +-tau_max)      firmware PD (v_des = t_ff = 0)
    tau_m  <- first-order lag of tau_cmd, time constant torque_lag    (= kt * I * n)
              then limited by the supply voltage minus back-EMF (torque-speed envelope)
    effort  = tau_m + cogging(theta) - B * omega - friction           times the ratio -> PhysX

The motor's own inertia is NOT applied here: it is the joint armature in the robot config,
J_motor * ratio^2 (the PhysX solver handles it implicitly). Friction is Karnopp stick-slip with
Stribeck and current-dependent terms; at 360 Hz it is applied as a bounded impulse: the friction
torque that would stop the motor's own inertia within one step, clamped to the friction level.
Using the motor inertia alone (a lower bound on what the joint really carries) keeps it from
ever reversing the motion. The torque lag is averaged over the step and the cogging amplitude
attenuated by its step average, because PhysX holds the effort constant for the whole step
(end-of-step values shorten the lag by ~1 ms, and raw cogging aliases at speed).

The reply's torque field is a sensor reading, not tau_m: the firmware subtracts a rotor-inertia
term, so reported = rep_gain * (tau_m - rep_J * acceleration), in the motor frame.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import torch

from protomotions.utils.config_builder import ConfigBuilder

# MIT protocol ranges of this motor (vendor-confirmed for the custom variant), motor frame
P_MAX = 12.5   # rad
V_MAX = 25.0   # rad/s
T_MAX = 28.0   # Nm, scales t_ff and the reply torque field only, it is not the current limit
KP_MAX = 500.0
KD_MAX = 5.0
INTERNAL_GEAR_RATIO = 9.0


def mit_quantize(x: torch.Tensor, lo: float, hi: float, bits: int) -> torch.Tensor:
    """Round-trip through the MIT protocol encoding (clamp, truncate to `bits`, decode)."""
    n = (1 << bits) - 1
    return torch.floor((x.clamp(lo, hi) - lo) * (n / (hi - lo))) * ((hi - lo) / n) + lo


@dataclass
class MotorModelConfig(ConfigBuilder):
    """AK80-9 actuator model. Defaults are the 2026-10-28 bench fit (model_params.json)."""

    enabled: bool = True

    # Belt between the AK80-9 output and the simulated joint (theta_motor = ratio * theta_joint)
    transmission_ratio: float = 4.61
    # The policy's angle action in [-1, 1] maps to a motor-frame p_des of +-desired_angle_range
    # (the MIT position field ends at +-12.5 rad, i.e. +-2.7 rad at the joint)
    desired_angle_range: float = P_MAX

    # Firmware PD and drive
    torque_lag: float = 0.00302  # s, first-order lag from command to motor torque
    tau_max: float = 52.6  # Nm, firmware current limit (the data only bounds it from below)
    quantize_commands: bool = True  # p_des, kp, kd go through the MIT encoding

    # Back-EMF: the torque available at speed w is limited to
    # stall * (1 - w / w0) when driving, stall * (1 + |w| / w0) when braking, with
    # w0 = V * kv / 9 (no-load speed) and stall = kt * 9 * V / R (voltage-limited stall torque).
    # NOT identified on the bench: kv is the rating, kt the value used by the characterization,
    # phase_resistance a guess (standard KV100 AK80-9 ~0.17 ohm, x4 for the KV50 winding).
    # Bench sanity check: 40 Nm at 12 rad/s was reached unsaturated (envelope gives 53 Nm).
    back_emf: bool = True
    supply_voltage: float = 48.0  # V
    kv: float = 50.0  # rpm/V, rotor
    kt: float = 0.147  # Nm/A, rotor
    phase_resistance: float = 0.68  # ohm, guess

    # Friction: break-away fc + fs, sliding fc + fs*exp(-(v/vs)^2) + fg*|tau_m|, plus B*v
    B: float = 0.00457  # Nm*s/rad
    fc: float = 0.195  # Nm
    fs: float = 0.0741  # Nm
    vs: float = 0.228  # rad/s
    fg: float = 0.0209  # Nm per Nm of motor torque

    # Cogging: cog_amp * sin(cog_cycles * theta + phase), 189 = 9 x 21 pole pairs per output rev.
    # The phase is tied to the encoder zero, which the sim does not share with the real motor,
    # so by default it is drawn uniformly per env at every reset.
    cog_amp: float = 0.375  # Nm
    cog_cycles: float = 189.0
    cog_phase: float = 2.13  # rad, used only when randomize_cog_phase is False
    randomize_cog_phase: bool = True

    # Reply torque field (sensor model, no effect on the motion)
    rep_gain: float = 0.968
    rep_J: float = 0.0291  # kg*m^2, rotor inertia the firmware subtracts


class MotorModel:
    """Batched AK80-9 model for `num_motors` motor joints in each of `num_envs` envs.

    Args:
        config: Model parameters.
        num_envs: Number of parallel environments.
        joint_armature: Armature of each motor joint (J_motor * ratio^2), used by the friction impulse.
        dt: Physics step in seconds.
        device: Torch device.
    """

    def __init__(
        self,
        config: MotorModelConfig,
        num_envs: int,
        joint_armature: List[float],
        dt: float,
        device: torch.device,
    ):
        self.config = config
        self.dt = dt
        self.device = device
        self.ratio = config.transmission_ratio
        num_motors = len(joint_armature)

        self.lag_alpha = 1.0 - math.exp(-dt / config.torque_lag) if config.torque_lag > 0 else 1.0
        # Weight of the previous torque in the step average of the lag response
        self.lag_mean = config.torque_lag / dt * self.lag_alpha if config.torque_lag > 0 else 0.0
        self.motor_inertia = torch.tensor(joint_armature, device=device, dtype=torch.float) / self.ratio**2

        self.noload_speed = config.supply_voltage * config.kv * 2 * math.pi / 60 / INTERNAL_GEAR_RATIO
        self.voltage_stall_torque = (
            config.kt * INTERNAL_GEAR_RATIO * config.supply_voltage / config.phase_resistance
        )

        self.motor_torque = torch.zeros(num_envs, num_motors, device=device)  # lag state, end of step
        self._applied_motor_torque = torch.zeros(num_envs, num_motors, device=device)  # step average
        # Reply torque field as of the last physics step (what a reply to the last command reads)
        self.reported_torque = torch.zeros(num_envs, num_motors, device=device)
        self.cog_phase = torch.full((num_envs, num_motors), config.cog_phase, device=device)
        self._prev_vel = torch.zeros(num_envs, num_motors, device=device)
        self._fresh = torch.ones(num_envs, 1, dtype=torch.bool, device=device)
        self.reset()

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Zero the motor torque (MIT entry is limp) and resample the cogging phase."""
        if env_ids is None:
            env_ids = slice(None)
        self.motor_torque[env_ids] = 0.0
        self._applied_motor_torque[env_ids] = 0.0
        self.reported_torque[env_ids] = 0.0
        self._fresh[env_ids] = True
        if self.config.randomize_cog_phase:
            self.cog_phase[env_ids] = (
                torch.rand_like(self.cog_phase[env_ids]) * (2 * math.pi)
            )

    def reported_torque_at(self, joint_vel: torch.Tensor, env_ids=slice(None)) -> torch.Tensor:
        """Reply torque field (motor frame) at the end of the last physics step, given the joint
        velocity now. Valid until the next step() call."""
        vel = joint_vel * self.ratio
        acc = torch.where(self._fresh[env_ids], torch.zeros_like(vel), (vel - self._prev_vel[env_ids]) / self.dt)
        return self.config.rep_gain * (self._applied_motor_torque[env_ids] - self.config.rep_J * acc)

    def _voltage_limit(self, tau: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        upper = (self.voltage_stall_torque * (1 - vel / self.noload_speed)).clamp(0.0, self.config.tau_max)
        lower = (self.voltage_stall_torque * (1 + vel / self.noload_speed)).clamp(0.0, self.config.tau_max)
        return torch.maximum(torch.minimum(tau, upper), -lower)

    def step(
        self,
        kp: torch.Tensor,
        kd: torch.Tensor,
        p_des: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one physics step and return the joint effort to apply.

        Args:
            kp, kd, p_des: MIT command per motor, motor frame [num_envs, num_motors] (v_des = t_ff = 0).
            joint_pos, joint_vel: Motor joint position and velocity at the start of the step.
        """
        cfg = self.config
        pos = joint_pos * self.ratio
        vel = joint_vel * self.ratio

        self.reported_torque = self.reported_torque_at(joint_vel)
        self._prev_vel = vel.clone()
        self._fresh[:] = False

        if cfg.quantize_commands:
            p_des = mit_quantize(p_des, -P_MAX, P_MAX, 16)
            kp = mit_quantize(kp, 0.0, KP_MAX, 12)
            kd = mit_quantize(kd, 0.0, KD_MAX, 12)

        tau_cmd = (kp * (p_des - pos) - kd * vel).clamp(-cfg.tau_max, cfg.tau_max)
        # Exact lag over the step: PhysX holds the effort for the whole step, so apply the
        # step average of tau_m (applying the end value would make the lag ~1 ms too short)
        tau_applied = tau_cmd + (self.motor_torque - tau_cmd) * self.lag_mean
        self.motor_torque = self.motor_torque + self.lag_alpha * (tau_cmd - self.motor_torque)
        if cfg.back_emf:
            tau_applied = self._voltage_limit(tau_applied, vel)
            self.motor_torque = self._voltage_limit(self.motor_torque, vel)
        self._applied_motor_torque = tau_applied

        # Cogging, attenuated by its average over the step's travel (sinc) so it does not alias at
        # speed. The phase is not advanced to mid-step: the start velocity mispredicts the travel
        # near stick-slip, which made low-gain runs 7x worse against the bench model
        half_travel = cfg.cog_cycles * vel * (self.dt / 2)
        cogging = (
            cfg.cog_amp
            * torch.sin(cfg.cog_cycles * pos + self.cog_phase)
            * torch.sinc(half_travel / math.pi)
        )
        drive = tau_applied + cogging - cfg.B * vel
        friction_level = (
            cfg.fc
            + cfg.fs * torch.exp(-((vel / cfg.vs) ** 2))
            + cfg.fg * tau_applied.abs()
        )
        stop_torque = drive + self.motor_inertia * vel / self.dt
        friction = torch.maximum(torch.minimum(stop_torque, friction_level), -friction_level)
        return (drive - friction) * self.ratio
