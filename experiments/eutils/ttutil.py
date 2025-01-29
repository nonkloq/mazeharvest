from typing import Tuple

import numpy as np
import torch

from .play_env import play
from .record import Record

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def obs_to_torch(perceptions, state):
    return [
        torch.tensor(perc, dtype=torch.float32, device=DEVICE) for perc in perceptions
    ], torch.tensor(state, dtype=torch.float32, device=DEVICE)


def obs_proc(obs):
    (
        perception,
        loot_heuristics,
        mole_heuristics,
        damage_directions,
        player_stats,
    ) = obs

    perception = [torch.tensor(perception, dtype=torch.float32, device=DEVICE)]

    state = np.concatenate(
        (loot_heuristics, mole_heuristics, damage_directions, player_stats)
    ).reshape(1, -1)

    state = torch.tensor(state, dtype=torch.float32, device=DEVICE)

    return (perception, state)


def get_baselines(test_env, N_test=20) -> Tuple[float, float]:
    random_moves = 0
    for _ in range(N_test):
        steps, _, _ = play(
            test_env,
            agent=lambda _: test_env.action_space.sample(),
            no_head=True,
        )
        random_moves += steps

    no_move = 0
    for _ in range(N_test):
        steps, _, _ = play(
            test_env,
            agent=lambda _: 0,
            no_head=True,
        )
        no_move += steps

    return random_moves / N_test, no_move / N_test


def log_policy_performance(
    agent,
    baseline_scores: Tuple[float, float],
    policy_records: Record,
    env,
    update_no,
    n_test=20,
):
    tot_counts = np.zeros(3)
    for _ in range(n_test):
        steps, kill_c, harvest_c = play(
            env, agent=agent, no_head=False, wait_for_quit=False
        )
        tot_counts[0] += steps
        tot_counts[1] += kill_c
        tot_counts[2] += harvest_c

    tot_counts = tot_counts / n_test

    compared_to_random = tot_counts[0] - baseline_scores[0]
    compared_to_nothing = tot_counts[0] - baseline_scores[1]

    step_improvement_from_baseline = (compared_to_random + compared_to_nothing) / 2

    print(
        f"Average Step Improvement: {step_improvement_from_baseline:.2f} | Compared to Random: {compared_to_random:.2f}, Compared to Nothing: {compared_to_nothing:.2f}"
    )
    print(
        f"Steps: {tot_counts[0]:.2f}, Kills: {tot_counts[1]:.2f}, Harvests: {tot_counts[2]:.2f} (averaged over {n_test} tests)"
    )
    policy_records.append(
        {
            "update_no": update_no,
            "avg_step_count": tot_counts[0],
            "avg_kill_count": tot_counts[1],
            "avg_harvest_count": tot_counts[2],
            "imp_from_baseline": step_improvement_from_baseline,
            "imp_from_random": compared_to_random,
            "imp_from_nothing": compared_to_nothing,
        }
    )

    return tot_counts[0]


# TODO: Change all to use np arrays instead of torch, only load it to device when training
class ObsHolder:
    """Wrapper to Hold Observation of single timestep"""

    def __init__(self, n_actors: int):
        self.n_actors = n_actors
        self.perception = np.full(self.n_actors, None, dtype=object)
        self.state = torch.zeros(
            (self.n_actors, 29), dtype=torch.float32, device=DEVICE
        )

    def __setitem__(self, actor: int, obs: tuple[np.ndarray]):
        perception, state = obs
        self.perception[actor] = torch.from_numpy(perception).to(device=DEVICE)
        self.state[actor] = torch.from_numpy(state)  # .to(device=DEVICE)

    def __getitem__(self, actor: int):
        return (self.perception[actor], self.state[actor])

    def __iter__(self):
        return iter((self.perception, self.state))


class ObsHolderMult:
    """Wrapper to Hold Observation of multiple timesteps timestep"""

    def __init__(self, n_actors: int, roll_out: int):
        self.n_actors = n_actors
        self.roll_out = roll_out

        self.perception = np.full((self.n_actors, self.roll_out), None, dtype=object)
        self.state = np.zeros((self.n_actors, self.roll_out, 29), dtype=np.float32)

        self.shape = (self.n_actors, self.roll_out, -1)

    def __setitem__(self, index: int, obs):
        perception, state = obs

        # if isinstance(index, tuple):
        row_index, col_index = index
        # copying the data from tensor
        perception = [perc.numpy().copy() for perc in perception]
        # assert row_index == slice(None)
        self.perception[row_index, col_index] = perception
        self.state[row_index, col_index] = state
        # return

        # self.perception[index] = np.array(perception)  # copying the data
        # self.state[index] = state

    # def __getitem__(self, index: int):
    #     if isinstance(index, tuple):
    #         row_index, col_index = index
    #         return (
    #             self.perception[row_index, col_index],
    #             self.state[row_index, col_index],
    #         )
    #
    #     return (self.perception[index], self.state[index])
    #
    # def __iter__(self):
    #     return iter((self.perception, self.state))

    def reshape(self, *shape):
        # print(perc)
        return FlatObs(
            np.array(
                [
                    torch.tensor(perc, device=DEVICE, dtype=torch.float32)
                    for perc in self.perception.reshape(shape[0])  # (N, )
                ],
                dtype=object,
            ),
            torch.tensor(
                self.state.reshape(*shape), device=DEVICE, dtype=torch.float32
            ),
        )


class FlatObs:
    def __init__(self, perc, state):
        self.perc = perc
        self.state = state

    def __getitem__(self, idx):
        return (self.perc[idx], self.state[idx])


class RewardScaler:
    def __call__(self, reward):
        return np.clip(reward / 100, -1.5, 1.5)
