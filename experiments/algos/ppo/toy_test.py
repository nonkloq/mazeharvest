import sys
from torch import nn
from torch.distributions import Categorical
import numpy as np
from .ppo import PPO

sys.path.append("../../")
from eutils.toyt import TestTrainer, bootstrap
import torch

from eutils import DEVICE, Record
from tqdm import tqdm


class Model(nn.Module):
    def __init__(
        self,
        observation_space: int,
        action_space: int,
    ):
        super().__init__()
        self.n = action_space

        self.extract_features = nn.Sequential(
            nn.Linear(observation_space, 128), nn.ReLU()
        )

        self.policy_logits = nn.Linear(128, action_space)

        self.value_function = nn.Linear(128, 1)

    def forward(self, obs):
        feats = self.extract_features(obs)
        logits = self.policy_logits(feats)
        value = self.value_function(feats).reshape(-1)

        return Categorical(logits=logits), value


class __Trainer(TestTrainer, PPO):
    def __init__(self, envs, convergence_f, **kwargs):
        TestTrainer.__init__(self, envs, convergence_f)
        self.model = Model(self.observation_space, self.action_space)
        PPO.__init__(self, **kwargs)

        self.batch_size = self.actor_steps * self.n_actors

        self.minibatch_size = self.batch_size // self.batches
        assert self.minibatch_size > 32

        self.obs = np.zeros(
            (self.n_actors, self.actor_steps, self.observation_space),
            dtype=np.float32,
        )
        self.prev_obs = np.zeros(
            (self.n_actors, self.observation_space), dtype=np.float32
        )

        self._reset_environments()

        self.tr_records = Record("train_stats", writer=False, save_log=False)

    def _reset_environments(self):
        obs, _ = self.envs.reset()
        self.prev_obs[:] = obs

    def _act_on_environment(self, actions, t):
        nxt_obs, rewards, dones, truncs, _ = self.envs.step(actions.tolist())

        # For episode stats counts
        self._convf(self, d=dones, t=truncs)
        self.prev_obs[:] = nxt_obs
        self.rewards[:, t] = rewards
        self.done[:, t] = dones

    @staticmethod
    def _obs_to_torch(obs):
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)

        return torch.tensor(obs, dtype=torch.float32, device=DEVICE)

    def calc_num_steps(self, update_t: int) -> int:
        return self.n_actors * self.actor_steps * update_t

    def run_training_loop(self, log_t: int) -> tuple[int, float]:
        update = 0
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

                samples = self.sample()

                avg_loss = self.train(samples)

                if update % log_t == 0:
                    self._convf(self, eval=True)
                    print(
                        f"Total Steps={self.calc_num_steps(update)} | Avg. Loss={avg_loss}"
                    )
                    self.tr_records.log()
                    self._convf(self, log=True)
                    print()

        except KeyboardInterrupt:
            print("\nTraining loop interrupted")
        except Exception as e:
            raise e
        finally:
            self.tr_records.close()
            self.envs.close()

        return update, self.tr_records.get_latest_val("loss")


if __name__ == "__main__":
    bootstrap(
        __Trainer,
        __file__,
        updates=10_000,
        epochs=8,
        actor_steps=216,
        batches=16,
        c1=1.0,  # value loss coef
        c2=0.001,  # entropy bonus coef
        ratio_clip_range=0.2,  # epsilon
        valf_clip_range=None,
        eta=1e-3,
        gae_gamma=0.99,
        gae_lambda=0.95,
    )
