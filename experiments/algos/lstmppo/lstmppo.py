"""
LSTM Proixmal Policy Optimzation
"""

import sys
from typing import Dict

import torch
from torch import optim
from tqdm import tqdm

import numpy as np

sys.path.append("../../")


from eutils import DEVICE, Actor, Record
from eutils.mutil import save_checkpoint
from .buffer import PPOBuffer
from .brain import Brain


class LSTMPPO:
    """
    LSTM - PPO Base Class, To build a specific env trainer inherit this.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        # c_1: value loss coef
        # c_2: entropy bonus coef
        # epsilon: ratio clip range
        # eps2: valf clip range

        # gae_gamma \gamma
        # gae_lambda \lambda
        self.model = Brain(
            self.state_size,
            self.hidden_size,
            self.observation_space,
            self.action_space,
            self.hfeature_size,
        ).to(DEVICE)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.eta, eps=1e-5
        )

        ## Buffers to hold sampled values
        # Trainer should implement this
        self.obs = torch.zeros(
            (self.n_actors, self.observation_space),
            dtype=torch.float32,
        )
        self.hidden_states = self.model.init_states(self.n_actors, DEVICE)

        self.sample_episode_step = np.zeros(self.n_actors)

        self.buffer = PPOBuffer(
            self.actor_steps,
            self.n_actors,
            self.observation_space,
            self.sequence_length,
            minibatch_size=self.minibatch_size,
            num_minibatches=self.n_batches,
            device=DEVICE,
        )

    def _act_on_environment(self, actions, t) -> tuple:
        raise NotImplementedError("Actor is missing")

    def _reset_environments(self):
        raise NotImplementedError("Reset method is missing")

    def sample(self):
        # reset the memory update order & episode counts
        self.buffer.reset()
        # current sample timesteps
        self.sample_episode_step *= 0

        with torch.no_grad():
            for t in range(self.actor_steps):
                policy, values, hidden_state = self.model(
                    self.obs.reshape(self.n_actors, 1, -1),
                    self.hidden_states,
                )

                actions = policy.sample().cpu()
                values = values.cpu()
                log_probs = policy.log_prob(actions).cpu()

                next_observation, reward, done = self._act_on_environment(
                    actions=actions, t=t
                )

                self.buffer.add(
                    observations=self.obs.cpu(),
                    actions=actions,
                    log_probs=log_probs,
                    values=values,
                    rewards=reward,
                    done=done,
                    sample_episode_step=self.sample_episode_step,
                    hidden_states=self.hidden_states,
                    timestep=t,
                )

                # Update current buffers
                self.obs.copy_(
                    torch.as_tensor(next_observation, device=DEVICE)
                )

                self.hidden_states = hidden_state
                eoe = self.buffer.end_of_episode[:, t]
                # reset for ended episodes
                self.hidden_states[0][:, eoe] *= 0
                self.hidden_states[1][:, eoe] *= 0

                # update & reset the counts
                self.sample_episode_step += 1
                self.sample_episode_step[eoe] = 0
            # final state values for advantage calculation
            _, last_values, _ = self.model(
                self.obs.reshape(self.n_actors, 1, -1), self.hidden_states
            )
            self.buffer.compute_gae(
                last_values.cpu().numpy(), self.gae_gamma, self.gae_lambda
            )

    def train(self):
        # K epochs
        avg_loss = None
        for _ in range(self.epochs):
            # minibatch size M < NT (actors * timesteps)
            for mini_batch in self.buffer.sample_mini_batches():
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
        policy, values, _ = self.model(
            samples["obs"],
            samples["hidden_states"],
        )

        # -\ln \pi_\theta (a_t|s_t), a_t are actions sampled from \pi_{\theta_{OLD}}
        log_probs = policy.log_prob(samples["actions"])

        # Entropy bonus: S[\pi_{\theta}](s_t)
        S_pi = policy.entropy()

        # flattening the sequence dimension & removing padded regions
        padding_mask = samples["padding_mask"]
        values = values[padding_mask]
        S_pi = S_pi[padding_mask]
        log_probs = log_probs[padding_mask]

        sampled_values = samples["values"]
        sampled_advantages = samples["advantages"]
        sampled_log_probs = samples["log_probs"]

        # $\bar{A_t} = \frac{\hat{A_t} - \mu(\hat{A_t})}{\sigma(\hat{A_t})}$,
        normalized_sampled_advantages = self._normalize(sampled_advantages)

        # r_t(\theta) = \frac{\pi(a_t|s_t)}{\pi_{OLD}(a_t|s_t)}
        policy_diff = log_probs - sampled_log_probs
        ratio = torch.exp(policy_diff)

        # Calculate PPO Loss: L^{CLIP} = min(surr, cliped_surr)
        surrogate_loss = ratio * normalized_sampled_advantages
        clipped_surrogate_loss = (
            torch.clamp(
                ratio, 1.0 - self.ratio_clip_range, 1.0 + self.ratio_clip_range
            )
            * normalized_sampled_advantages
        )

        # policy loss
        L_clip = torch.min(surrogate_loss, clipped_surrogate_loss).mean()

        # Calculate Value Function Loss: L^{VF}(\theta)
        # Squared-error loss (V_{\theta}(s_t) - V_t^{targ})^2

        # R_t returns sampled from \pi_{\theta_{OLD}}
        # A = R - V -> R = V + A
        sampled_return = sampled_values + sampled_advantages
        clipped_values = sampled_values + (values - sampled_values).clamp(
            min=-self.valf_clip_range, max=self.valf_clip_range
        )

        # value function loss
        L_vf = torch.max(
            (values - sampled_return) ** 2,
            (clipped_values - sampled_return) ** 2,
        ).mean()

        # average of entropy bonus
        S_pi = S_pi.mean()

        # flipping the signs to gradient descent instead
        # of doing ascent to maximize objective function
        L_clip_vf_s = -(L_clip - self.c1 * L_vf + self.c2 * S_pi)

        # http://joschu.net/blog/kl-approx.html
        approx_kl = (ratio.detach().cpu() - 1.0) - policy_diff.detach().cpu()
        clip_fraction = (
            (abs((ratio.detach().cpu() - 1.0)) > self.ratio_clip_range)
            .float()
            .mean()
        )
        self.tr_records.append(
            {
                "policy_reward": L_clip.item(),
                "value_loss": L_vf.item(),
                "entropy_bonus": S_pi.item(),
                "loss": L_clip_vf_s.item(),
                "kl_divergence": approx_kl.mean().cpu().data.numpy(),
                "clip_fraction": clip_fraction.cpu().data.numpy(),
            }
        )

        return L_clip_vf_s

    @staticmethod
    def _obs_to_torch(obs):
        raise NotImplementedError("Missing Obs to torch method.")

    def _reset_state(self):
        if not hasattr(self, "act_state"):
            self.act_state = self.model.init_states(1, DEVICE)
        else:
            self.act_state[0][:] *= 0
            self.act_state[1][:] *= 0
        self.prev_action = -1

    def act(self, obs, deterministic: bool = True):
        """Takes a single observation and return action"""
        with torch.no_grad():
            policy, _, self.act_state = self.model(
                self._obs_to_torch(obs, action=self.prev_action),
                self.act_state,
            )
            if deterministic:
                action = torch.argmax(policy.probs).item()
            else:
                action = policy.sample().item()
            self.prev_action = action

        return action


class PPOTrainer(LSTMPPO):
    def __init__(
        self,
        env_confs: dict,
        tb_writer: bool = False,
        log_writer: bool = False,
        **kwargs,
    ):
        self.n_actors = len(env_confs)
        # max rays size + other obs size + prev action encoding
        self.observation_space = 6 * 30 + 30 + 10
        self.action_space = 10

        print("Training Parameters:")
        print(kwargs)
        super().__init__(**kwargs)

        # with previous timestep action in observation
        self.actors = [
            Actor(
                env_confs[i],
                flat_obs=True,
                max_rays=30,
                include_prev_action=True,
            )
            for i in range(self.n_actors)
        ]

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
            self.obs[i] = torch.as_tensor(actor.coms.recv(), device=DEVICE)

    @staticmethod
    def _obs_to_torch(
        obs,
        action,
        allowed_rays=30,
    ):
        (
            perception,
            loot_heuristics,
            mole_heuristics,
            damage_directions,
            player_stats,
        ) = obs

        if perception.shape[0] > allowed_rays:
            print(
                f"Cutdown perceptions of shape {perception.shape} to {allowed_rays}"
            )
            perception = perception[:allowed_rays]

        elif perception.shape[0] < allowed_rays:
            diff = allowed_rays - perception.shape[0]
            padding = np.zeros((diff, perception.shape[1]))
            perception = np.concatenate((perception, padding))

        ace = np.zeros(10)
        if action >= 0:
            ace[action] = 1

        obs = np.concatenate(
            (
                player_stats,
                loot_heuristics,
                mole_heuristics,
                damage_directions,
                perception.reshape(-1),
                ace,
            )
        ).reshape(1, 1, -1)

        return torch.tensor(obs, dtype=torch.float32)

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
        nxt_obs = np.zeros(
            (self.n_actors, self.observation_space), dtype=np.float32
        )
        rewards = np.zeros((self.n_actors,), dtype=np.float32)
        dones = np.zeros((self.n_actors,), dtype=np.float32)
        for _a, actor in enumerate(self.actors):
            nxt_obs[_a], rewards[_a], dones[_a], info = actor.coms.recv()

            # if episode done / truncated,
            # it will return the episode info
            if info:
                nxt_obs[_a] = info.pop("next_ep_obs")
                self.ep_records.append(info)

        return nxt_obs, rewards, dones

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
            self.sample()
            # train the model for some epochs
            avg_loss = self.train()

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
