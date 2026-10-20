# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Data utilities for experience management and batching.

This module provides utilities for managing experience buffers and creating
minibatch datasets for training. Handles efficient storage and retrieval of
rollout data collected during environment interaction.

Key Classes:
    - ExperienceBuffer: Buffer for storing rollout experience
    - DictDataset: Dataset for creating minibatches from experience

Key Functions:
    - swap_and_flatten01: Reshape tensors for batching
    - get_dict: Extract dictionary view of experience buffer
"""

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from typing import Dict
import numpy as np


def swap_and_flatten01(arr: Tensor):
    """Swap and flatten first two dimensions of a tensor.

    Converts (num_steps, num_envs, ...) to (num_steps * num_envs, ...).
    Commonly used to batch experience from parallel environments.

    Args:
        arr: Tensor with at least 2 dimensions.

    Returns:
        Tensor with first two dimensions flattened.
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


class ExperienceBuffer(nn.Module):
    """Buffer for storing rollout experience from parallel environments.

    Collects observations, actions, rewards, and other data during environment
    rollouts. Provides efficient storage and batching for on-policy algorithms.
    Uses PyTorch buffers for automatic device management.

    Args:
        num_envs: Number of parallel environments.
        num_steps: Number of steps per rollout.

    Attributes:
        store_dict: Dictionary tracking which keys have been populated.

    Example:
        >>> buffer = ExperienceBuffer(num_envs=1024, num_steps=16)
        >>> buffer.register_key("obs", shape=(128,))
        >>> buffer.update_data("obs", step=0, data=observations)
        >>> data_dict = buffer.get_dict()
    """

    def __init__(self, num_envs: int, num_steps: int, device: torch.device):
        super().__init__()
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.store_dict = {}
        self.virtual_keys = {}
        self._device = device

    def register_key(self, key: str, shape=(), dtype=torch.float):
        assert not hasattr(self, key), key
        buffer = torch.zeros(
            (self.num_steps, self.num_envs) + shape, dtype=dtype, device=self._device
        )
        self.register_buffer(key, buffer, persistent=False)
        self.store_dict[key] = 0

    def register_shared_key(self, key: str, tensor: Tensor):
        """Register a tensor owned by another buffer (same storage, no copy).

        Used to share observation storage between agents. The owning buffer
        writes the data and tracks its completeness, so it is not tracked here.
        """
        assert not hasattr(self, key), key
        assert tensor.shape[:2] == (self.num_steps, self.num_envs), (
            f"Shared key '{key}' has shape {tuple(tensor.shape)}, "
            f"expected leading dims {(self.num_steps, self.num_envs)}"
        )
        self.register_buffer(key, tensor, persistent=False)

    def register_virtual_key(self, key: str, store: "HistoryStore"):
        """Register a key whose data is reconstructed lazily from a HistoryStore."""
        assert not hasattr(self, key) and key not in self.virtual_keys, key
        assert (store.num_steps, store.num_envs) == (self.num_steps, self.num_envs), key
        self.virtual_keys[key] = store

    def update_data(self, key: str, index: int, data: Tensor):
        assert not data.requires_grad
        getattr(self, key)[index] = data
        self.store_dict[key] += index + 1

    def total_sum(self):
        return (self.num_steps + 1) * (self.num_steps / 2)

    def batch_update_data(self, key: str, data: Tensor):
        assert not data.requires_grad
        getattr(self, key)[:] = data
        self.store_dict[key] = self.total_sum()

    def make_dict(self):
        # Flatten (num_steps, num_envs) step-major. Unlike swap_and_flatten01 this is a view,
        # so no copy of the buffers is made; sample order is irrelevant since minibatches are shuffled.
        data = {
            k: v.reshape(self.num_steps * self.num_envs, *v.shape[2:])
            for k, v in self.named_buffers()
        }
        for k, store in self.virtual_keys.items():
            data[k] = store.flat_view()
        self.check_and_reset_store_counts()
        return data

    def check_and_reset_store_counts(self):
        for k, v in self.store_dict.items():
            assert v == self.total_sum(), f"Problem with '{k}', {v}, {self.total_sum()}"
            self.store_dict[k] = 0


class HistoryStore:
    """Deduplicated rollout storage for a history observation.

    A history obs at step t is [frame_t, frame_t-1, ..., frame_t-H+1] (flattened, newest first),
    so consecutive steps share H-1 frames. Instead of storing H frames per (step, env), frames are
    stored once in a pool and each (step, env) keeps H int32 pointers into it.

    Exactness: every step the new history is compared to the previous one. Only if it is exactly
    the previous history shifted by one frame is a single new frame stored; otherwise (e.g. after a
    reset refilled the history) the whole history is stored as a snapshot, with consecutive
    identical frames stored once. Reconstruction therefore always reproduces the observed history.
    """

    def __init__(self, num_steps: int, num_envs: int, hist_len: int, frame_dim: int, dtype, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.hist_len = hist_len
        self.frame_dim = frame_dim
        self.device = device
        self.ptr = torch.zeros(num_steps, num_envs, hist_len, dtype=torch.int32, device=device)
        # Initial snapshot (N*H) + one frame per (step, env); grows if resets need more
        capacity = num_envs * hist_len + 2 * num_steps * num_envs
        self.pool = torch.zeros(capacity, frame_dim, dtype=dtype, device=device)
        self.size = 0
        self.next_step = 0
        self._prev_hist = None
        self._prev_ptr = None

    def _append(self, rows: Tensor) -> int:
        start = self.size
        end = start + rows.shape[0]
        if end > self.pool.shape[0]:
            if end >= 2**31:
                raise RuntimeError("HistoryStore pool exceeds int32 pointer range")
            new_pool = torch.empty(
                max(end, int(self.pool.shape[0] * 1.5)), self.frame_dim,
                dtype=self.pool.dtype, device=self.device,
            )
            new_pool[:start] = self.pool[:start]
            self.pool = new_pool
        self.pool[start:end] = rows
        self.size = end
        return start

    def _store_snapshot(self, hist: Tensor) -> Tensor:
        """Store full histories [B, H, D] (consecutive duplicate frames once); returns pointers [B, H]."""
        unique = torch.ones(hist.shape[:2], dtype=torch.bool, device=self.device)
        unique[:, 1:] = (hist[:, 1:] != hist[:, :-1]).any(dim=-1)
        start = self._append(hist[unique])
        # Duplicates point at the most recent stored frame before them (slot 0 is always stored)
        row_ids = torch.cumsum(unique.flatten(), 0).view(unique.shape) - 1
        return start + row_ids

    @torch.no_grad()
    def update(self, step: int, flat_hist: Tensor):
        assert step == self.next_step, f"HistoryStore expected step {self.next_step}, got {step}"
        hist = flat_hist.reshape(self.num_envs, self.hist_len, self.frame_dim)

        if step == 0:
            # New rollout: earlier frames are not in this rollout's pool
            self.size = 0
            ptr = self._store_snapshot(hist)
        else:
            shifted = (hist[:, 1:] == self._prev_hist[:, :-1]).flatten(1).all(dim=1)
            ptr = torch.empty(self.num_envs, self.hist_len, dtype=torch.long, device=self.device)

            ok = shifted.nonzero(as_tuple=True)[0]
            start = self._append(hist[ok, 0])
            ptr[ok, 0] = torch.arange(start, start + ok.numel(), device=self.device)
            ptr[ok, 1:] = self._prev_ptr[ok, :-1]

            bad = (~shifted).nonzero(as_tuple=True)[0]
            if bad.numel() > 0:
                ptr[bad] = self._store_snapshot(hist[bad])

        self.ptr[step] = ptr.to(torch.int32)
        self._prev_ptr = ptr
        self._prev_hist = hist.clone()
        self.next_step = (step + 1) % self.num_steps

    def check_rollout_complete(self):
        assert self.next_step == 0, f"HistoryStore rollout incomplete (next step {self.next_step})"

    def compression_ratio(self) -> float:
        """Stored frames / frames a dense buffer would hold (pointers not included)."""
        return self.size / (self.num_steps * self.num_envs * self.hist_len)

    def flat_view(self) -> "HistoryView":
        return HistoryView(
            self.pool[: self.size],
            self.ptr.view(self.num_steps * self.num_envs, self.hist_len),
        )


class HistoryView:
    """Read-only (num_steps * num_envs, H * D) view that rebuilds histories on indexing."""

    def __init__(self, pool: Tensor, ptr: Tensor):
        self.pool = pool
        self.ptr = ptr

    def __len__(self):
        return self.ptr.shape[0]

    @property
    def shape(self):
        return (self.ptr.shape[0], self.ptr.shape[1] * self.pool.shape[1])

    def __getitem__(self, idx):
        if isinstance(idx, np.ndarray):
            idx = torch.as_tensor(idx, device=self.ptr.device)
        ptr = self.ptr[idx].long()
        return self.pool[ptr].reshape(*ptr.shape[:-1], -1)


class DictDataset(Dataset):
    """PyTorch Dataset for dictionary of tensors with minibatching.

    Creates minibatches from a dictionary of tensors. Supports shuffling and
    automatic batching for training. Used to create minibatch iterators from
    collected experience buffers.

    Args:
        batch_size: Size of each minibatch.
        tensor_dict: Dictionary of tensors to batch (all same length in dim 0).
        shuffle: Whether to shuffle indices before batching.

    Example:
        >>> data = {"obs": obs_tensor, "actions": action_tensor}
        >>> dataset = DictDataset(batch_size=256, tensor_dict=data, shuffle=True)
        >>> for batch in dataset:
        >>>     train_on_batch(batch)
    """

    def __init__(
        self,
        batch_size: int,
        tensor_dict: Dict[str, Tensor],
        shuffle=False,
    ):
        assert len(tensor_dict) > 0
        lengths_dict = {k: len(v) for k, v in tensor_dict.items()}
        assert (
            len(set(lengths_dict.values())) == 1
        ), f"All tensors must have the same length. Found: {lengths_dict}"

        self.num_tensors = next(iter(lengths_dict.values()))
        self.batch_size = batch_size
        assert (
            self.num_tensors % self.batch_size == 0
        ), f"{self.num_tensors} {self.batch_size}"
        self.tensor_dict = tensor_dict
        self.do_shuffle = shuffle
        self.shuffled_to_original = np.arange(self.num_tensors)

        if shuffle:
            self.shuffle()

    def shuffle(self):
        self.shuffled_to_original = np.random.permutation(self.num_tensors)

    def num_batches(self):
        return self.num_tensors // self.batch_size

    def __len__(self):
        return self.num_batches()

    def __getitem__(self, index):
        assert index < len(self), f"{index} {len(self)}"
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.num_tensors)
        return {
            k: v[self.shuffled_to_original[start_idx:end_idx]]
            for k, v in self.tensor_dict.items()
        }
