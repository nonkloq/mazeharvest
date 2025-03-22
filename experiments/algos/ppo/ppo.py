"""
Proixmal Policy Optimzation: https://arxiv.org/pdf/1707.06347
"""

import sys
from typing import Dict, Tuple

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

sys.path.append("../../")


from eutils import DEVICE, Actor, Record
from eutils.mutil import save_checkpoint
from eutils.ttutil import ObsHolderMult, ObsHolder, obs_proc


def compute_GAE(
    rewards: np.ndarray,
    values: np.ndarray,
    done: np.ndarray,
    _gamma: float,
    _lambda: float,
    _table_shape: Tuple[int, int],
):
    """Return Generalized Advantage Estimation"""

    advantages = np.zeros(_table_shape, dtype=np.float32)

    # V[s_{t+1}]
    nxt_value = values[:, -1]
    nxt_gae = 0

    gamma_lambda = _gamma * _lambda

    for t in reversed(range(_table_shape[1])):  # T to 0
        # to only consider values that are not from terminal states
        mask = 1 - done[:, t]
        nxt_value = nxt_value * mask
        nxt_gae = nxt_gae * mask

        # \delta_t
        delta_t = rewards[:, t] + _gamma * nxt_value - values[:, t]

        # \hat{A_t} = \delta_t + (\gamma \lambda) * \hat{A_{t+1}}
        nxt_gae = delta_t + gamma_lambda * nxt_gae

        advantages[:, t] = nxt_gae
        nxt_value = values[:, t]

    return advantages


class PPO:
    """
    PPO Base Class, To build a specific env trainer inherit this.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        # c_1: value loss coef
        # c_2: entropy bonus coef
        # epsilon: ratio clip range
        # eps2: valf clip range

        # gae_gamma \gamma
        # gae_lambda \lambda

        self.model = self.model.to(DEVICE)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.eta, eps=1e-5
        )

        ## Buffers to hold sampled values
        # Trainer should implement this
        self.prev_obs = None
        self.obs = None

        self.rewards = np.zeros(
            (self.n_actors, self.actor_steps),
            dtype=np.float32,
        )
        self.actions = np.zeros(
            (self.n_actors, self.actor_steps),
            dtype=np.int16,
        )
        self.done = np.zeros(
            (self.n_actors, self.actor_steps),
            dtype=bool,
        )
        self.values = np.zeros(
            (self.n_actors, self.actor_steps + 1),
            dtype=np.float32,
        )
        self.log_probs = np.zeros(
            (self.n_actors, self.actor_steps),
            dtype=np.float32,
        )

    def _act_on_environment(self, actions, t):
        raise NotImplementedError("Actor is missing")

    def _reset_environments(self):
        raise NotImplementedError("Reset method is missing")

    def sample(self) -> dict:
        with torch.no_grad():
            for t in range(self.actor_steps):
                self.obs[:, t] = self.prev_obs

                policy, vals = self.model(self._obs_to_torch(self.prev_obs))
                self.values[:, t] = vals.cpu()
                action = policy.sample()
                self.actions[:, t] = action.cpu()

                self.log_probs[:, t] = policy.log_prob(action).cpu()

                self._act_on_environment(actions=action, t=t)

            # get V[s_{t+1}] of the final state
            # in advantage estimation, the delta term requires it
            # \delta_t = r_t + \gamma V[s_{t+1}] - V[s_{t}]
            _, v = self.model(self._obs_to_torch(self.prev_obs))
            self.values[:, self.actor_steps] = v.cpu()

        advantages = compute_GAE(
            self.rewards,
            self.values,
            self.done,
            self.gae_gamma,
            self.gae_lambda,
            (self.n_actors, self.actor_steps),
        )

        samples = {
            "obs": self.obs,
            "actions": self.actions,
            "values": self.values[:, :-1],
            "log_probs": self.log_probs,
            "advantages": advantages,
        }

        # samples are currently in `[actors, time_step]` table,
        # we should flatten it for training (a * t, whatever_the_rest_is)
        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            if "obs" == k:
                samples_flat[k] = self._obs_to_torch(v)
            else:
                samples_flat[k] = torch.tensor(v, device=DEVICE)
        return samples_flat

    def train(self, samples: dict):
        # K epochs
        avg_loss = None
        for _ in range(self.epochs):
            # shuffle for each epoch
            indexes = torch.randperm(self.batch_size)

            # minibatch size M < NT (actors * timesteps)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mini_batch_indexes = indexes[start:end]

                mini_batch = {
                    k: v[mini_batch_indexes] for k, v in samples.items()
                }

                # optimize surrogate loss L wrt \theta
                loss = self._calc_loss(mini_batch)

                # Zero out the previously calculated gradients
                self.optimizer.zero_grad()
                # Calculate gradients
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=0.5
                )
                # Update parameters based on gradients
                self.optimizer.step()

                if avg_loss is None:
                    avg_loss = loss.item()
                else:
                    avg_loss = avg_loss * 0.8 + 0.2 * loss.item()

        return avg_loss

    @staticmethod
    def _normalize(adv: torch.Tensor):
        """Normalize advantage function"""
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    def _calc_loss(self, samples: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate total loss"""

        # $\bar{A_t} = \frac{\hat{A_t} - \mu(\hat{A_t})}{\sigma(\hat{A_t})}$,
        sampled_advantage = samples["advantages"]  # self._normalize()

        # \pi_\theta(a_t|s_t) and V^{\pi_\theta}(s_t) (current policy)
        policy, values = self.model(samples["obs"])

        # -\ln \pi_\theta (a_t|s_t), a_t are actions sampled from \pi_{\theta_{OLD}}
        new_log_probs = policy.log_prob(samples["actions"])

        # r_t(\theta) = \frac{\pi(a_t|s_t)}{\pi_{OLD}(a_t|s_t)}
        ratio = torch.exp(new_log_probs - samples["log_probs"])

        # Calculate PPO Loss: L^{CLIP} = min(surr, cliped_surr)
        surrogate_loss = ratio * sampled_advantage  # samples['advantages']
        clipped_surrogate_loss = (
            torch.clamp(
                ratio, 1.0 - self.ratio_clip_range, 1.0 + self.ratio_clip_range
            )
            * sampled_advantage
        )  # samples['advantages']

        L_clip = torch.min(surrogate_loss, clipped_surrogate_loss).mean()

        # Calculate Value Function Loss: L^{VF}(\theta)
        # Squared-error loss (V_{\theta}(s_t) - V_t^{targ})^2

        # R_t returns sampled from \pi_{\theta_{OLD}}
        # A = R - V -> R = V + A
        sampled_return = samples["values"] + samples["advantages"]
        if self.valf_clip_range is None:
            values_pred = values
        else:
            values_pred = samples["values"] + (
                values - samples["values"]
            ).clamp(min=-self.valf_clip_range, max=self.valf_clip_range)
        L_vf = torch.nn.functional.mse_loss(sampled_return, values_pred)

        # Entropy bonus: S[\pi_{\theta}](s_t)
        S_pi = policy.entropy().mean()

        # flipping the signs to gradient descent instead
        # of doing ascent to maximize objective function
        L_clip_vf_s = -L_clip + self.c1 * L_vf - self.c2 * S_pi

        self.tr_records.append(
            {
                "policy_reward": L_clip.item(),
                "value_loss": L_vf.item(),
                "entropy_bonus": S_pi.item(),
                "loss": L_clip_vf_s.item(),
            }
        )

        return L_clip_vf_s

    @staticmethod
    def _obs_to_torch(obs):
        raise NotImplementedError("Missing Obs to torch method.")

    def act(self, obs, deterministic: bool = True):
        """Takes a single observation and return action"""
        with torch.no_grad():
            policy, _ = self.model(self._obs_to_torch(obs))
            if deterministic:
                action = torch.argmax(policy.probs)
            else:
                action = policy.sample()

        return action.item()


class PPOTrainer(PPO):
    def __init__(
        self,
        env_confs: dict,
        model,
        tb_writer: bool = False,
        log_writer: bool = False,
        **kwargs,
    ):
        self.model = model
        self.n_actors = len(env_confs)

        print("Training Parameters:")
        print(kwargs)
        super().__init__(**kwargs)

        self.batch_size = self.actor_steps * self.n_actors
        self.minibatch_size = self.batch_size // self.batches

        # assert self.minibatch_size > 32

        self.actors = [Actor(env_confs[i]) for i in range(self.n_actors)]

        self.obs = ObsHolderMult(self.n_actors, self.actor_steps)
        self.prev_obs = ObsHolder(self.n_actors)

        self._reset_environments()

        self.ep_records = Record(
            "episode_stats", writer=tb_writer, save_log=log_writer
        )
        self.tr_records = Record(
            "train_stats", writer=tb_writer, save_log=log_writer
        )

    def _reset_environments(self):
        for actor in self.actors:
            actor.coms.send(("reset", None))

        for i, actor in enumerate(self.actors):
            self.prev_obs[i] = actor.coms.recv()

    @staticmethod
    def _obs_to_torch(obs):
        if isinstance(obs, tuple):
            return obs_proc(obs)
        return obs

    @property
    def model_(self):
        """For saving checkpoint"""
        return self.model

    @model_.setter
    def model_(self, statedict):
        """For Loading checkpoint"""
        self.model.load_state_dict(statedict)

    def _act_on_environment(self, actions, t):
        for _a, actor in enumerate(self.actors):
            actor.coms.send(("step", actions[_a].item()))

        # get results from the above step
        for _a, actor in enumerate(self.actors):
            self.prev_obs[_a], self.rewards[_a, t], self.done[_a, t], info = (
                actor.coms.recv()
            )

            # if episode done / truncated,
            # it will return the episode info
            if info:
                self.prev_obs[_a] = info.pop("next_ep_obs")
                self.ep_records.append(info)

    def run_training_loop(
        self,
        log_t: int = 100,
        checkpoint_t: int = 1000,
        cp_name: str = "cdqn_trainer.pth",
        sep_checks: float = -1.0,
    ):
        for update in tqdm(
            range(1, self.updates + 1),
            desc="Training",
            unit="Update",
            unit_scale=True,
        ):
            # sample with current policy
            samples = self.sample()
            # train the model for some epochs
            avg_loss = self.train(samples)

            # Add a new line to the screen periodically
            if log_t > 0 and update % log_t == 0:
                self.tr_records.log()
                self.ep_records.log()

            if checkpoint_t > 0 and update % checkpoint_t == 0:
                print(f"Saving Checkpoint for update #{update}...")
                # create new cps after a part of the update
                _cp_name = (
                    cp_name
                    if update < (self.updates * sep_checks)
                    else f"{update}-{cp_name}"
                )

                save_checkpoint(self, update, avg_loss, _cp_name)

        self.tr_records.close()
        self.ep_records.close()

    def destroy(self):
        """Stop the child processes"""
        for actor in self.actors:
            actor.kill_your_self()
