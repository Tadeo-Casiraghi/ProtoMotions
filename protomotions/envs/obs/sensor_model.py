"""Per-env sensor model for observation vectors made of named channels.

Each channel follows SensorChannelConfig: reading = saturate(quantize(true(t - latency) + bias + noise)).
Latency: push() stores the true readings at every physics step (newest first); apply() picks, per env
and channel, the sample `latency` steps old, where 0 is the reading passed to apply().
The bias, the per-env noise scale and the per-env latency (domain randomization) are redrawn by
resample(), once per episode; the white noise is drawn on every apply().
"""

import math
from typing import Dict, List, Optional, Tuple

import torch

from protomotions.envs.obs.config import ProstheticSensorConfig


class SensorModel:
    """
    Args:
        config: Model of every channel name used in `layout`.
        layout: (channel name, width) of each consecutive slice of the observation vector.
        num_envs: Number of parallel environments.
        physics_dt: Time between push() calls (s).
        device: Torch device.
        noise_scale_ranges: Optional {channel: (low, high)} per-env scale of std and bias_std.
        latency_ranges: Optional {channel: (low, high)} per-env latency in s, replaces the config's.
    """

    def __init__(
        self,
        config: ProstheticSensorConfig,
        layout: List[Tuple[str, int]],
        num_envs: int,
        physics_dt: float,
        device: torch.device,
        noise_scale_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        latency_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        channels = list(dict.fromkeys(name for name, _ in layout))
        noise_scale_ranges = dict(noise_scale_ranges or {})
        latency_ranges = dict(latency_ranges or {})
        for what, ranges in (("noise_scale_range", noise_scale_ranges), ("latency_range", latency_ranges)):
            unknown = set(ranges) - set(channels)
            if unknown:
                raise ValueError(f"sensors.{what} has unknown channels {sorted(unknown)}, valid: {channels}")

        def per_dim(get):
            return torch.tensor(
                [get(getattr(config, name)) for name, width in layout for _ in range(width)],
                dtype=torch.float,
                device=device,
            )

        self.std = per_dim(lambda c: c.std)
        self.bias_std = per_dim(lambda c: c.bias_std)
        self.resolution = per_dim(lambda c: c.resolution)
        self.limit = per_dim(lambda c: c.limit if c.limit is not None else math.inf)
        self.quantized = self.resolution > 0
        self.safe_resolution = torch.where(self.quantized, self.resolution, torch.ones_like(self.resolution))

        # Channel of each dim, so one scale / latency is drawn per channel (all axes of an IMU together)
        self.dim_channel = torch.tensor(
            [channels.index(name) for name, width in layout for _ in range(width)], device=device
        )

        def channel_range(ranges, nominal):
            lo = torch.tensor([ranges.get(c, (nominal(c), nominal(c)))[0] for c in channels], device=device)
            hi = torch.tensor([ranges.get(c, (nominal(c), nominal(c)))[1] for c in channels], device=device)
            return lo.float(), hi.float()

        self.scale_low, self.scale_high = channel_range(noise_scale_ranges, lambda c: 1.0)
        latency_low, latency_high = channel_range(latency_ranges, lambda c: getattr(config, c).latency)
        self.latency_low = latency_low / physics_dt
        self.latency_high = latency_high / physics_dt
        self.max_latency_steps = int(torch.round(self.latency_high).max().item())

        self.num_dims = len(self.std)
        self.scale = torch.ones(num_envs, self.num_dims, device=device)
        self.bias = torch.zeros(num_envs, self.num_dims, device=device)
        self.latency_steps = torch.zeros(num_envs, self.num_dims, dtype=torch.long, device=device)
        # history[k] = true reading k + 1 physics steps before the newest one
        self.history = (
            torch.zeros(self.max_latency_steps, num_envs, self.num_dims, device=device)
            if self.max_latency_steps > 0
            else None
        )
        # Envs whose history predates their reset: read with zero latency until refilled
        self._stale = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.resample()

    @property
    def needs_history(self) -> bool:
        """True if push() must be called every physics step."""
        return self.history is not None

    def resample(self, env_ids=slice(None)) -> None:
        """Draw a new noise scale, bias and latency for these envs (call on episode reset)."""
        n = self.scale[env_ids].shape[0]
        device = self.scale.device
        u = torch.rand(n, len(self.scale_low), device=device)
        self.scale[env_ids] = (self.scale_low + (self.scale_high - self.scale_low) * u)[:, self.dim_channel]
        self.bias[env_ids] = torch.randn(n, self.num_dims, device=device) * self.bias_std * self.scale[env_ids]
        u = torch.rand(n, len(self.latency_low), device=device)
        steps = torch.round(self.latency_low + (self.latency_high - self.latency_low) * u).long()
        self.latency_steps[env_ids] = steps[:, self.dim_channel]
        self._stale[env_ids] = True

    @torch.no_grad()
    def push(self, true_reading: torch.Tensor) -> None:
        """Store the true reading of all envs [num_envs, num_dims] at this physics step."""
        if self.history is None:
            return
        self.history = self.history.roll(1, dims=0)
        self.history[0] = true_reading
        stale = self._stale.nonzero(as_tuple=True)[0]
        if len(stale):
            self.history[:, stale] = true_reading[stale]
            self._stale[stale] = False

    def apply(self, true_reading: torch.Tensor, env_ids=slice(None)) -> torch.Tensor:
        """Return the sensor reading for these envs, given the true reading now [len(env_ids), num_dims]."""
        x = true_reading
        if self.history is not None:
            delays = self.latency_steps[env_ids]
            stacked = torch.cat([true_reading.unsqueeze(0), self.history[:, env_ids]], dim=0)
            delayed = stacked.gather(0, delays.unsqueeze(0)).squeeze(0)
            x = torch.where(self._stale[env_ids].unsqueeze(-1), true_reading, delayed)
        x = x + self.bias[env_ids] + torch.randn_like(x) * self.std * self.scale[env_ids]
        x = torch.where(self.quantized, torch.round(x / self.safe_resolution) * self.safe_resolution, x)
        return torch.maximum(torch.minimum(x, self.limit), -self.limit)
