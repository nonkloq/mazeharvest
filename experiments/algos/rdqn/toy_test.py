import sys
from typing import Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

sys.path.append("../../")
from eutils import DEVICE, Record
from eutils.toyt import TestTrainer, bootstrap

from .rdqn import RainbowDQN
from .replaybuffer import NStepPERBuffer


# --------------------------------Common Model----------------------------------
class Model(nn.Module):
    def __init__(
        self,
        observation_space: int,
        action_space: int,
        n_atoms: int = 51,
        v_min: float = -100,
        v_max: float = 100,
    ):
        super().__init__()
        self.n_atoms = n_atoms
        self.n = action_space

        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))

        self.extract_features = nn.Sequential(
            nn.Linear(observation_space, 128), nn.ReLU()
        )

        self.advantage_function = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, self.n * self.n_atoms)
        )

        self.value_function = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, self.n_atoms)
        )

        # self.network = nn.Sequential(
        #     nn.Linear(observation_space, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, self.n * n_atoms),
        # )

    def get_action(self, obs: torch.Tensor, actions=None):
        q_values, pmfs = self(obs)

        if actions is None:
            actions = torch.argmax(q_values, 1)

        batch_idx = torch.arange(obs.size(0))
        return q_values[batch_idx, actions], actions, pmfs[batch_idx, actions]

    def forward(self, obs: torch.Tensor):
        # logits = self.network(obs)
        # pmfs = torch.softmax(logits.view(obs.size(0), self.n, self.n_atoms), dim=2)
        # q_values = (pmfs * self.atoms).sum(dim=2)

        features = self.extract_features(obs)
        advantages = self.advantage_function(features).view(-1, self.n, self.n_atoms)
        values = self.value_function(features).view(-1, 1, self.n_atoms)

        logits = values + advantages - advantages.mean(dim=1, keepdim=True)

        pmfs = torch.softmax(logits, dim=-1)

        q_values = (pmfs * self.atoms).sum(dim=2)

        return q_values, pmfs


# -------------------------------Algorithm Code---------------------------------
class _ReplayBuffer(NStepPERBuffer):
    """Modified For Test"""

    def __init__(
        self,
        capacity: int,
        alpha: float,
        obs_size: int,
        batch_size: int,
        mn_step: tuple[int, int],
        n_actors: int,
        gamma: float,
    ):
        super().__init__(
            capacity,
            lambda x: torch.zeros((x, obs_size), dtype=torch.float32),
            alpha,
            batch_size,
            mn_step,
            n_actors,
            gamma,
        )

    def extend(self, obss, actions, rewards, next_obss, dones):
        for x in range(len(obss)):
            self.add(
                obss[x],
                actions[x],
                rewards[x],
                torch.from_numpy(next_obss[x]),
                bool(dones[x]),
            )


class __Trainer(TestTrainer, RainbowDQN):
    def __init__(self, envs, convergence_f, **kwargs):
        TestTrainer.__init__(self, envs, convergence_f)
        RainbowDQN.__init__(
            self,
            modelC=Model,
            modelargs=dict(
                observation_space=self.observation_space, action_space=self.action_space
            ),
            **kwargs,
        )

        self.replay_buffer = _ReplayBuffer(
            self.buffer_capacity,
            self.buffer_alpha,
            self.observation_space,
            self.minibatch_size,
            self.mn_step,
            self.n_actors,
            self.gamma,
        )

        self.obs = torch.zeros(
            (self.n_actors, self.observation_space), dtype=torch.float32, device=DEVICE
        )

        self._reset_environments()

        self.tr_records = Record("train_stats", writer=False, save_log=False)

    def _reset_environments(self):
        obs, _ = self.envs.reset()
        self.obs.copy_(torch.from_numpy(obs))

    def _act_on_environment(self, actions, t):
        mask = np.random.random(self.n_actors) < self.epsilon(t)
        actions[mask] = self.envs.action_space.sample()[mask]

        nxt_obs, rewards, dones, truncs, _ = self.envs.step(actions)

        # For episode stats counts
        self._convf(self, d=dones, t=truncs)

        # dones = dones | truncs
        self.replay_buffer.extend(self.obs, actions, rewards, nxt_obs, dones)
        self.obs.copy_(torch.from_numpy(nxt_obs))

    @staticmethod
    def _obs_to_torch(obs):
        return torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=DEVICE)

    def calc_num_steps(self, update_t: int) -> int:
        return self.n_actors * self.actor_steps * update_t

    def run_training_loop(self, log_t: int) -> Tuple[int, float]:
        update = 0

        print("Filling samples...")
        while self.replay_buffer.size < self.minimum_steps:
            self.sample(update)

        try:
            for update in tqdm(
                range(1, self.updates + 1),
                desc="Training",
                unit="Update",
                unit_scale=True,
            ):
                # early termintaion
                if self._convf(self, check=True):
                    return update - 1, self.tr_records.get_latest_val("loss")

                # Sampling Phase
                for _ in range(self.actor_steps):
                    self.sample(update)

                # training phase
                for x in range(1, self.epochs + 1):
                    data = self.replay_buffer.sample(self.buffer_beta(update))
                    stats = self.train(data, precomp=x < self.epochs)
                    if x == self.epochs:
                        self.tr_records.append(stats)

                # logging & Eval
                if update % log_t == 0:
                    self._convf(self, eval=True)
                    print(
                        f"Epsilon={self.epsilon(update):.5f} | Total Steps={self.calc_num_steps(update)}"
                    )
                    self.tr_records.log()
                    self._convf(self, log=True)
                    print()

        except KeyboardInterrupt:
            print("\nTraining loop interrupted")
        except Exception as e:
            raise e
        finally:
            del self.replay_buffer
            self.tr_records.close()
            self.envs.close()

        return update, self.tr_records.get_latest_val("loss")


# ------------------------------------------------------------------------------

# Test Code
if __name__ == "__main__":
    bootstrap(
        __Trainer,
        __file__,
        updates=10_000,
        minibatch_size=64,
        actor_steps=1,  # env roll out
        eta=5e-4,
        gamma=0.99,
        v_min=-100,
        v_max=100,
        # buffer_capacity-1 to wait for it to become full
        minimum_steps=8000,  # Minimum Steps in replay buffer to start the training
        n_atoms=51,
        tau=0.005,
        epochs=8,
        buffer_capacity=2**13,
        buffer_alpha=0.6,
        buffer_beta_start=0.4,
        buffer_beta_end=1.0,
        # minimum_steps // n_actors * actor_steps
        rand_explore_till=200,  # high amount of random exploration for first N Steps
        max_epsilon=0.8,
        min_epsilon=0.01,
        mn_step=(6, 9),
    )
