"""
Prioritized Experience Replay Buffer: https://arxiv.org/pdf/1511.05952
PER + N-Step Buffer for Nstep Learning: http://incompleteideas.net/papers/sutton-88-with-erratum.pdf
"""

import sys
import warnings
from typing import Callable

import numpy as np
import torch as th

sys.path.append("../../")

from eutils import DEVICE

from ._fsegtree import SegmentTreeSM

# from ._psegtree import SegmentTreeSM


class ObservationHolder:
    def __init__(self, cap: int, device: str = "cpu") -> None:
        self.cap = cap

        self._perceptions = np.full(cap, None, dtype=object)
        self._states = th.zeros((cap, 29), dtype=th.float32, device=device)
        self._dev = device

    def __setitem__(self, idx, obs):
        perception, state = obs
        if not isinstance(perception, th.Tensor):
            perception = th.from_numpy(perception)
            state = th.from_numpy(state)

        self._perceptions[idx] = perception.to(device=self._dev)
        self._states[idx] = state

    def __getitem__(self, idx):
        self._ridxs = idx
        return self

    def to(self, device: str):
        return (
            [prec.to(device=device) for prec in self._perceptions[self._ridxs]],
            self._states[self._ridxs].to(device=device),
        )

    def __del__(self):
        del self._perceptions
        del self._states


class ReplayBuffer:
    """Standard Replay Buffer"""

    def __init__(self, capacity: int, obsh_gen: Callable[[int], object]):
        """
        Args:
            capacity (int)
            obsh_gen (Callable[int])
                it should return either a th.Tensor or an holder class
                which implements getitem, setitem & to(device: str) methods.
        """

        self.capacity = capacity

        self.data = {
            "obs": obsh_gen(capacity),
            "action": th.zeros(capacity, dtype=th.long),
            "reward": th.zeros((capacity, 1), dtype=th.float32),
            "next_obs": obsh_gen(capacity),
            "done": th.zeros((capacity, 1), dtype=th.bool),
        }

        self.next_idx = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        idx = self.next_idx

        # store in the queue
        self.data["obs"][idx] = obs
        self.data["action"][idx] = action
        self.data["reward"][idx] = reward
        self.data["next_obs"][idx] = next_obs
        self.data["done"][idx] = done

        # Increment next available slot
        self.next_idx = (idx + 1) % self.capacity
        # Calculate the current size
        self.size = min(self.capacity, self.size + 1)

    def _random_sample(self, batch_size: int) -> dict:
        indexes = th.randint(0, self.size, size=(batch_size,))

        return dict(
            obs=self.data["obs"][indexes].to(device=DEVICE),
            next_obs=self.data["next_obs"][indexes].to(device=DEVICE),
            action=self.data["action"][indexes].to(device=DEVICE),
            reward=self.data["reward"][indexes].to(device=DEVICE),
            done=self.data["done"][indexes].to(device=DEVICE),
        )

    def sample(self, batch_size: int) -> dict:
        return self._random_sample(batch_size)


class PERBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer
    """

    def __init__(
        self,
        capacity: int,
        obsh_gen: Callable[[int], object],
        alpha: float,
        batch_size: int,
    ):
        if capacity & (capacity - 1) != 0:
            pow = int(np.round(np.log2(capacity)))

            new_capacity = 2**pow

            warnings.warn(
                f"The Given capacity {capacity} is not a power of 2, the new capacity will be {new_capacity} which is 2^{pow}"
            )
            capacity = new_capacity

        super().__init__(capacity, obsh_gen)
        self.alpha = alpha

        self.segment_tree = SegmentTreeSM(capacity, batch_size)
        self.max_priority = 1.0

    def add(self, obs, action, reward, next_obs, done):
        # update the segment trees
        self.segment_tree[self.next_idx] = self.max_priority**self.alpha
        super().add(obs, action, reward, next_obs, done)

    def sample(self, beta: float) -> dict:
        self.segment_tree.send_sample_request(self.size)
        indexes = self.segment_tree.get_indexes()

        weights = th.tensor(
            ((self.segment_tree[indexes] / self.segment_tree._sum) * self.size)
            ** (-beta),
            dtype=th.float32,
        )

        # Normalize the weights
        prob_min = self.segment_tree._min / self.segment_tree._sum
        max_weight = (prob_min * self.size) ** (-beta)
        weights /= max_weight

        return dict(
            indexes=indexes,
            weights=weights.to(device=DEVICE),
            obs=self.data["obs"][indexes].to(device=DEVICE),
            next_obs=self.data["next_obs"][indexes].to(device=DEVICE),
            action=self.data["action"][indexes].to(device=DEVICE),
            reward=self.data["reward"][indexes].to(device=DEVICE),
            done=self.data["done"][indexes].to(device=DEVICE),
        )

    def update_priorities(
        self, indexes: np.ndarray, priorities: np.ndarray, presample: bool
    ):
        """
        Update the priorities of the SARS pairs
        If presample=True, initiate index retreival for next batch
        """
        for idx, priority in zip(indexes, priorities):
            self.segment_tree[idx] = priority**self.alpha
            self.max_priority = max(self.max_priority, priority)

        if presample:
            self.segment_tree.send_presample_request(self.size)


class NStepPERBuffer(PERBuffer):
    """

    N-Step Priority Experience Replay Buffer
    .add() should follow the same fixed order 0...n_actors

    o_1_0, ... , o_1_A, .... ,o_T_0, ..., o_T_A
    A is [0 to n_actors]
    T is timestep, doesn't break for new episode/trajectory

    Instead of fixed steps, here we can set steps range from M to N.
    During sampling a random number between the range will be picked.
    """

    def __init__(
        self,
        capacity: int,
        obsh_gen: Callable[[int], object],
        alpha: float,
        batch_size: int,
        mn_step: tuple[int, int],
        n_actors: int,
        gamma: float,
    ):
        """
        mn_step [m, n]
        """
        assert mn_step[0] < mn_step[1], "Invalid range"
        assert (
            mn_step[0] > 1
        ), "Increase the low or Use Normal Replay Buffer or PREBuffer Instead"

        # Updating (using .add()) should wo
        super().__init__(capacity, obsh_gen, alpha, batch_size)
        self.mn_step = mn_step
        self.n_actors = n_actors
        self.gamma = gamma ** th.arange(self.mn_step[1])
        self.gamma_sca = gamma

    @staticmethod
    def _mask_dones(dones: th.Tensor):
        batch_size, step_size = dones.shape
        dones = dones.clone()

        # setting the last possible index as default
        first_done = th.full((batch_size,), step_size - 1, dtype=th.long)
        for i in range(batch_size):
            for j in range(step_size):
                if dones[i][j]:
                    first_done[i] = j
                    dones[i, j] = False  # flip it to include it in calculation
                    break

        # (1,step_size) > (batch_size, 1), here we creating a mask for
        # each row (batch) idx number comparing with first done idx
        # by broadcasting these two shapes we get (batch_size, step_size)
        entries_to_change = th.arange(step_size)[None, :] > first_done[:, None]
        dones[entries_to_change] = True

        return ~dones, first_done

    @staticmethod
    def _pad_tensors(x: th.Tensor, N: int):
        return th.cat((x, th.zeros(N - x.size(0), dtype=x.dtype)))

    def sample_n_steps(self, indices: np.ndarray):
        batch_size = len(indices)

        random_mn_steps = np.random.randint(
            self.mn_step[0], self.mn_step[1] + 1, batch_size
        )
        real_steps = th.zeros(batch_size, dtype=th.long)
        end = indices + (random_mn_steps * self.n_actors)
        # there will be no continuation after next_idx-1 if the start step
        # is less than write pointer, so setting the upper bound to nxt_idx (exclusive)
        maskl = indices < self.next_idx
        end[maskl] = np.clip(end[maskl], 0, self.next_idx)

        # Create Matrix of rewards & dones for this batch
        rewards_paded = [None] * batch_size
        # here we calculating real X_steps
        for i, (_s, _e) in enumerate(zip(indices, end)):
            eprews = self.data["reward"][_s : _e : self.n_actors].squeeze(1)
            real_steps[i] = len(eprews)
            rewards_paded[i] = self._pad_tensors(eprews, self.mn_step[1])

        rewards_paded = th.stack(rewards_paded)

        dones_paded = th.stack(
            [
                self._pad_tensors(
                    self.data["done"][_s : _e : self.n_actors].squeeze(1),
                    self.mn_step[1],
                )
                for _s, _e in zip(indices, end)
            ]
        )

        masked_dones, first_done_idx = self._mask_dones(dones_paded)

        # N step rewards
        R_t_tn = (rewards_paded * self.gamma * masked_dones).sum(dim=1)

        # Idx relative to the batch n_step matrix. here
        # the max possible index can be n_step-1 for each sample
        nxt_obs_idx = th.minimum(first_done_idx, real_steps - 1)
        # converting the relative idxs to real buffer indices
        # if n_step is 1, the idx will be 0 and final nxt_obs_idx will be ==
        # indices, if the n_step>1 the nxt_obs_idx>0 and the final
        # idx will point to last step in buffer
        nxt_obs_idx = th.from_numpy(indices) + nxt_obs_idx * self.n_actors
        # here we ignoring cycle through buffer to get rest of the steps in
        # that episode, it is harder to track if the step belongs to that or
        # next episode or not (need another array to keep track of it)
        # instead we only considering steps after start idx, because here it is
        # guaranteed (>nxt_index) to be the step of same episode or the next episode.
        # other wise it will be considered as min_step and it can be less than M.

        return dict(
            obs=self.data["obs"][indices].to(device=DEVICE),
            next_obs=self.data["next_obs"][nxt_obs_idx].to(device=DEVICE),
            action=self.data["action"][indices].to(device=DEVICE),
            reward=R_t_tn.reshape(batch_size, 1).to(device=DEVICE),
            done=self.data["done"][nxt_obs_idx].to(device=DEVICE),
            # discounted gamma after n_steps (gamma ^ n_step)
            gamma_k=self.gamma_sca
            ** real_steps.reshape(batch_size, 1).to(device=DEVICE),
        )
