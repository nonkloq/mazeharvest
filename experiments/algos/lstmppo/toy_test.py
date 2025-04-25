import sys

import torch
from tqdm import tqdm

sys.path.append("../../")
from eutils import DEVICE, Record
from eutils.toyt import TestTrainer, bootstrap

from .lstmppo import LSTMPPO


class __Trainer(TestTrainer, LSTMPPO):
    def __init__(self, envs, convergence_f, **kwargs):
        TestTrainer.__init__(self, envs, convergence_f)
        LSTMPPO.__init__(self, **kwargs)

        self._reset_environments()

        self.tr_records = Record("train_stats", writer=False, save_log=False)

    def _reset_environments(self):
        obs, _ = self.envs.reset()
        self.obs[:] = torch.as_tensor(obs, device=DEVICE)

    def _act_on_environment(self, actions, t):
        nxt_obs, rewards, dones, truncs, _ = self.envs.step(actions.tolist())

        # For episode stats counts
        self._convf(self, d=dones, t=truncs)
        return nxt_obs, rewards, dones | truncs

    @staticmethod
    def _obs_to_torch(obs):
        if len(obs.shape) == 1:
            obs = obs.reshape(1, 1, -1)

        return torch.tensor(obs, dtype=torch.float32, device=DEVICE)

    def prep_for_eval(self):
        self._reset_state()

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

                self.sample()

                avg_loss = self.train()

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
        epochs=4,
        actor_steps=256,
        c1=0.2,  # value loss coef
        c2=0.001,  # entropy bonus coef
        ratio_clip_range=0.2,  # epsilon
        valf_clip_range=0.5,
        eta=1e-3,
        gae_gamma=0.99,
        gae_lambda=0.95,
        # model params
        state_size=64,
        hidden_size=128,
        hfeature_size=32,
        sequence_length=8,
        minibatch_size=None,
        n_batches=2,
    )
