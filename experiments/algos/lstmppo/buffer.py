from collections import Counter
from typing import Optional

import numpy as np
import torch as th
from torch.nn.utils.rnn import pad_sequence


class PPOBuffer:
    def __init__(
        self,
        actor_steps: int,
        n_actors: int,
        observation_dim: tuple,
        max_sequence_length: int,
        minibatch_size: Optional[int] = None,
        num_minibatches: Optional[int] = None,
        min_sequence_length: int = 2,
        device: str = "cpu",
    ):
        self.actor_steps = actor_steps
        self.n_actors = n_actors
        self.observation_dim = observation_dim

        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length

        assert minibatch_size is not None or num_minibatches is not None
        self.minibatch_size = minibatch_size
        self.n_minibatches = num_minibatches

        self.device = device

        self.minibatch_size = minibatch_size
        self.hidden_states = RecycList(n_actors)

        self.obs = np.zeros(
            (self.n_actors, self.actor_steps, self.observation_dim),
            dtype=np.float32,
        )

        self.rewards = np.zeros(
            (self.n_actors, self.actor_steps),
            dtype=np.float32,
        )

        self.actions = np.zeros(
            (self.n_actors, self.actor_steps),
            dtype=np.int16,
        )

        self.end_of_episode = np.zeros(
            (self.n_actors, self.actor_steps + 1),
            dtype=bool,
        )

        self.values = np.zeros(
            (self.n_actors, self.actor_steps),
            dtype=np.float32,
        )

        self.log_probs = np.zeros(
            (self.n_actors, self.actor_steps),
            dtype=np.float32,
        )

        self.advantages = np.zeros(
            (self.n_actors, self.actor_steps),
            dtype=np.float32,
        )

        # to mark end of the final sequence even if it is not an end
        self.end_of_episode[:, self.actor_steps] = True

    def reset(self):
        self.hidden_states.reset_order()

    def add(
        self,
        observations,
        actions,
        log_probs,
        values,
        rewards,
        done,
        sample_episode_step,
        hidden_states: tuple[th.Tensor, th.Tensor],
        # sample ts
        timestep: int,
    ):
        """add sampled data to the buffer"""
        self.obs[:, timestep] = observations
        self.actions[:, timestep] = actions
        self.log_probs[:, timestep] = log_probs
        self.values[:, timestep] = values
        self.rewards[:, timestep] = rewards
        self.end_of_episode[:, timestep] = done

        seq_start = (sample_episode_step % self.max_sequence_length) == 0
        for actor, is_ss in enumerate(seq_start):
            if is_ss:
                self.hidden_states.add_data(
                    actor,
                    (
                        hidden_states[0][:, actor].cpu().numpy().copy(),
                        hidden_states[1][:, actor].cpu().numpy().copy(),
                    ),
                    timestep,
                )

    def compute_gae(
        self, last_value, _gamma: float = 0.99, _lambda: float = 0.95
    ):
        """
        1) Compute Advantages using Generalized Advantages Estimation
        2) Clears previous sequence batch

        NOTE: Only Call at the end of the sampling process
        """
        self.advantages[:] = 0
        # V[s_{t+1}]
        last_gae = 0

        for t in reversed(range(self.actor_steps)):  # T to 0
            # to only consider values that are not from terminal states
            mask = 1 - self.end_of_episode[:, t]
            last_value = last_value * mask
            last_gae = last_gae * mask

            # \delta_t
            delta_t = (
                self.rewards[:, t] + _gamma * last_value - self.values[:, t]
            )

            # \hat{A_t} = \delta_t + (\gamma \lambda) * \hat{A_{t+1}}
            last_gae = delta_t + _gamma * _lambda * last_gae
            self.advantages[:, t] = last_gae
            last_value = self.values[:, t]

        # Prepare to make sequences for the current batch ---
        self.batch_sequences = None

    def _intialize_batch_sequences(
        self,
    ):
        """Initializes the sequence & its indexes"""
        if self.batch_sequences is not None:
            return

        # start and end position of each episodes
        row, end_positions = np.where(self.end_of_episode)
        # shifting the end position by one to get the
        # start positions of the next episodes.
        start_positions = end_positions.copy() + 1
        end_positions += 1  # slicing is exclusive at the end
        episode_count = Counter(row)
        start_pos = 0
        for actor in range(self.n_actors):
            end_pos = start_pos + episode_count[actor]

            start_positions[start_pos:end_pos] = np.roll(
                start_positions[start_pos:end_pos], 1
            )
            # start position of the first episode will be 0
            start_positions[start_pos] = 0
            start_pos = end_pos

        # get start and end positions of all the sequence
        batch_sequences = []
        for i in range(len(row)):  # iterate through episodes
            episode_start = start_positions[i]
            episode_end = end_positions[i]
            # final idx where the min_sequence_length requirement is valid
            # end + 1 to make it inclusive when min_sequence_length is 1
            final_valid_sequence_start = (
                episode_end + 1 - self.min_sequence_length
            )
            for seq_start in range(
                episode_start,
                final_valid_sequence_start,
                self.max_sequence_length,
            ):
                seq_stop = min(
                    seq_start + self.max_sequence_length, episode_end
                )
                batch_sequences.append((row[i], slice(seq_start, seq_stop)))

        self.batch_sequences = np.array(batch_sequences)

    def _create_mini_batch(self, mb_sequence: np.ndarray) -> dict:
        obs = [
            th.tensor(self.obs[row, section], device=self.device)
            for row, section in mb_sequence
        ]
        actions = [
            th.tensor(self.actions[row, section], device=self.device)
            for row, section in mb_sequence
        ]

        values = [
            th.tensor(self.values[row, section], device=self.device)
            for row, section in mb_sequence
        ]
        log_probs = [
            th.tensor(self.log_probs[row, section], device=self.device)
            for row, section in mb_sequence
        ]
        advantages = [
            th.tensor(self.advantages[row, section], device=self.device)
            for row, section in mb_sequence
        ]

        sequence_sizes = [len(x) for x in actions]

        hidden_states, cell_states = zip(
            *[
                self.hidden_states.get_data(row, section.start)
                for row, section in mb_sequence
            ]
        )

        hidden_states = th.tensor(
            np.array(hidden_states), device=self.device
        ).transpose(0, 1)
        cell_states = th.tensor(
            np.array(cell_states), device=self.device
        ).transpose(0, 1)

        samples = {
            "hidden_states": (hidden_states, cell_states),
            # all paded on the right side
            "obs": pad_sequence(obs, batch_first=True),
            "actions": pad_sequence(actions, batch_first=True).reshape(-1),
            # false for padded regions, True for original data, all flat
            "padding_mask": pad_sequence(
                [th.ones(size, device=self.device) for size in sequence_sizes],
                batch_first=True,
            )
            .to(dtype=th.bool)
            .reshape(-1),
            # No padding & all flat
            "values": th.cat(values, dim=0),
            "log_probs": th.cat(log_probs, dim=0),
            "advantages": th.cat(advantages, dim=0),
        }
        return samples

    def sample_mini_batches(
        self,
    ):
        """
        Generator to sample minibatches
        Output:
            {

                "key": (batch * seq_len no padding, *data dim)
                "hidden_states": (batch size x hidden size, ...)
            }

        """
        self._intialize_batch_sequences()

        batch_size = len(self.batch_sequences)
        if self.n_minibatches is not None:
            mini_batch_size = batch_size // self.n_minibatches
        else:
            mini_batch_size = self.minibatch_size

        indexes = th.randperm(batch_size)
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mini_batch_idxs = indexes[start:end]

            # batch size is too small, [idx] will return an element
            # instead of list of elements
            if len(mini_batch_idxs) <= 1:
                continue

            mini_batch = self._create_mini_batch(
                self.batch_sequences[mini_batch_idxs]
            )

            yield mini_batch


class RecycList:
    """
    Second-level structure for reusing existing nodes.
    here there is no separation of chains—new episodes do not create new chains.
    """

    def __init__(self, num_branches: int) -> None:
        self._branches = [Chain(toplevel=False) for _ in range(num_branches)]
        self._update_order = np.zeros(num_branches, dtype=int)

    def reset_order(self):
        self._update_order *= 0
        for branch in self._branches:
            branch.head.key = float("inf")

    def add_data(self, branch_idx: int, data: np.ndarray, timestep: int):
        rclist = self._branches[branch_idx]

        rclist.update_at(data, timestep, self._update_order[branch_idx])
        self._update_order[branch_idx] += 1

    def get_data(self, branch_idx: int, timestep: int):
        rclist = self._branches[branch_idx]

        node = rclist.get_substructure(timestep)
        assert (
            node.key == timestep
        ), f"saved timestamp {node.key} is not equal to requested timestamp {timestep}"

        return node.data


class TreeNode:
    def __init__(self, data, key, _next=None):
        self.data = data
        self.key = key
        self.next = _next


class Chain:
    def __init__(self, toplevel: bool):
        if toplevel:
            # create sub Chains
            self.head = TreeNode(Chain(toplevel=False), float("inf"))
        else:
            self.head = TreeNode(None, None)

    def _traverse_substructure(self, key: int):
        """
        Returns the previous and current position,
        where the key lies in between prev<=key<=curr
        """
        curr: TreeNode = self.head
        prev = None
        while curr.next:
            if curr.key < key:
                prev, curr = curr, curr.next
            else:
                break
        return prev, curr

    def get_substructure(self, key):
        """
        Returns the substructure that contains the given key
        """
        _, curr = self._traverse_substructure(key)
        return curr

    def update_at(self, data: np.ndarray | tuple, timestep: int, order: int):
        """
        Update the Treenode data at given index,
        it will create one if the index doesn't exist
        """
        current: TreeNode = self.head
        prev = None
        for _ in range(order):
            prev, current = current, current.next
            # will throw an error if we didn't handle the order properly

        if current is None:
            prev.next = TreeNode(data, timestep, None)
        else:
            current.data = data
            current.key = timestep

    def get_neighbours(self, timestep: int):
        """
        Get Adjacent memory matrix from both left and right side of the
        given time step
        """
        prev, curr = self._traverse_substructure(timestep)
        if prev is None:  # start of the list
            prev = curr

        return (prev.data, prev.key), (curr.data, curr.key)
