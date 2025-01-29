"""
RainbowDQN

- C51 Model & Training [https://arxiv.org/pdf/1707.06887] (qnet.py)
- Double Q Learning [https://arxiv.org/pdf/1509.06461] in _calc_cdqn_loss method
- Prioritized Experience Replay Buffer (replay_buffer.py)
- N-Step Learning NStepPERBuffer (replay_buffer.py)
- Noisy Layers & Dueling Net (qnet.py), while using noisy layers in modle uncomment
  reset_noise function in model and uncomment reset_noise method call, before return in base class train method
"""

import sys

import numpy as np
import torch
from torch import nn

sys.path.append("../../")

from eutils import ACTION_SPACE, DEVICE, Actor, Record
from eutils.mutil import save_checkpoint
from eutils.schedulers import Piecewise
from eutils.ttutil import ObsHolder, obs_proc
from tqdm import tqdm

from .replaybuffer import ObservationHolder, NStepPERBuffer


class RainbowDQN:
    """
    RainbowDQN Base Class, To build a specific env trainer inherit this.

    Trainer Should Implement its own replay_buffer and obs holder.
    """

    def __init__(self, modelC, modelargs: dict, **kwargs):
        self.__dict__.update(kwargs)

        if "use_nstep" not in kwargs:
            self.use_nstep = True

        self.q_network: nn.Module = modelC(
            n_atoms=self.n_atoms, v_min=self.v_min, v_max=self.v_max, **modelargs
        ).to(device=DEVICE)

        self.target_network: nn.Module = modelC(
            n_atoms=self.n_atoms, v_min=self.v_min, v_max=self.v_max, **modelargs
        ).to(device=DEVICE)

        # Hard Copying q-net params and disabling gradient comp
        self._hard_update()
        self.target_network.load_state_dict(self.q_network.state_dict())
        for p in self.target_network.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.eta)

        # Trainer should initialize its buffer
        self.replay_buffer = None

        self.buffer_beta = Piecewise(
            [(0, self.buffer_beta_start), (self.updates, self.buffer_beta_end)],
            outside_value=1,
        )
        # exploration during training
        # very random exploration for first N steps
        self.epsilon = Piecewise(
            [
                (0, 1.0),
                (self.rand_explore_till, self.max_epsilon),
                (self.updates // 2, self.min_epsilon),
            ],
            outside_value=self.min_epsilon,
        )

        # CDQN consts
        self.atoms = torch.linspace(
            self.v_min, self.v_max, steps=self.n_atoms, device=DEVICE
        )
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        # Holder for m_i
        # will cause inplace change issue during backprob because of the
        # two step loss calc in train step
        # self.target_pmfs = torch.zeros((self.minibatch_size, self.n_atoms))
        self.offset = (
            torch.linspace(
                0,
                (self.minibatch_size - 1) * self.n_atoms,
                self.minibatch_size,
                device=DEVICE,
            )
            .unsqueeze(-1)
            .long()
        )

        # Fixed Observation Holder
        # Trainer should initialize this, it is either an torch.Tensor or a wrapper class
        self.obs = None

    def _reset_environments(self):
        raise NotImplementedError("Reset Method is missing")

    def _act_on_environment(self, actions, t):
        """It should add data to replay buffer and modify the fixed self.obs"""
        raise NotImplementedError("Missing Sample Actor")

    def sample(self, t: int):
        """Actor Rollout"""  # using e-greedy strategy"""

        with torch.no_grad():
            _, actions, _ = self.q_network.get_action(self.obs)
            actions = actions.detach().cpu().numpy()

        self._act_on_environment(actions, t)

    @torch.no_grad()
    def _soft_update(self):
        # https://pytorch.org/rl/stable/_modules/torchrl/objectives/utils.html#SoftUpdate
        for target_param, q_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.lerp_(q_param.data, self.tau)

    def _hard_update(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self, data, precomp: bool = False) -> dict:
        # Get the Q values & pmfs for the sampled actions using q_net
        # (Using same pass for both loss calculation)
        q_val_sampled_action, _, pmfs_sampled = self.q_network.get_action(
            data["obs"], data["action"]
        )

        losses, td_error, q_val_sampled_action = self._calc_cdqn_loss(
            data, self.gamma, pmfs_sampled, q_val_sampled_action
        )

        if self.use_nstep:
            n_step_samples = self.replay_buffer.sample_n_steps(data["indexes"])
            n_step_losses, td_err_n_step, _ = self._calc_cdqn_loss(
                n_step_samples,
                n_step_samples["gamma_k"],
                pmfs_sampled,
                q_val_sampled_action,
            )

            losses += n_step_losses

        # weighting the loss with current sample weights
        if self.replay_buffer.isPER:
            loss = (losses * data["weights"]).mean()
        else:
            loss = losses.mean()

        # ---- Out of comp graph
        # td_error -> difference
        if self.use_nstep:
            sample_td = (np.abs(td_error) + np.abs(td_err_n_step)) / 2
        else:
            sample_td = np.abs(td_error)

        if self.replay_buffer.isPER:
            loss_for_prior = losses.detach().cpu().numpy()
            new_priorities = loss_for_prior + 1e-6  # sample_td + 1e-6
            self.replay_buffer.update_priorities(
                data["indexes"], new_priorities, precomp
            )
        # ------------------------

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update Target network parameters
        self._soft_update()

        # reset noisy layers
        # self.target_network.reset_noise()
        # self.q_network.reset_noise()

        return {
            "loss": loss.item(),
            "q_values": q_val_sampled_action.detach().cpu().mean().item(),
            "sample_td": sample_td.mean(),
        }

    def _calc_cdqn_loss(
        self,
        data: dict,
        gamma: float | torch.Tensor,
        pmfs_sampled: torch.Tensor,
        q_val_sampled_action: torch.Tensor,
    ):
        # Get the Q values & pmfs for the sampled actions using q_net
        # q_val_sampled_action, _, pmfs_sampled = self.q_network.get_action(
        #     data["obs"], data["action"]
        # )

        with torch.no_grad():
            # DQL, decoupling action selection and value function.
            # best action for next state using q-net
            _, best_nxt_action, _ = self.q_network.get_action(data["next_obs"])
            # get pmfs for the best actions
            _, _, nxt_pmfs = self.target_network.get_action(
                data["next_obs"], best_nxt_action
            )

            # Projection
            nxt_atoms = data["reward"] + gamma * self.atoms * (~data["done"])

            T_zj = nxt_atoms.clamp(min=self.v_min, max=self.v_max)

            bj = (T_zj - self.v_min) / self.delta_z
            lo = bj.floor().long()
            up = bj.ceil().long()

            # Distribute probabilities
            delta_ml = (up + (lo == up) - bj) * nxt_pmfs
            delta_mu = (bj - lo) * nxt_pmfs

            # self.target_pmfs.zero_()  # self.target_pmfs *= 0

            target_pmfs = torch.zeros(
                (self.minibatch_size, self.n_atoms), device=DEVICE
            )
            target_pmfs.view(-1).index_add_(
                0, (lo + self.offset).view(-1), delta_ml.view(-1)
            )
            target_pmfs.view(-1).index_add_(
                0, (up + self.offset).view(-1), delta_mu.view(-1)
            )

            # Target Q values
            q_val_update = (target_pmfs * self.atoms).sum(dim=1)

            # temporal difference error
            td_error = q_val_sampled_action - q_val_update
            td_error = td_error.cpu().numpy()

        # Calculating Cross entropy loss
        losses = -(target_pmfs * pmfs_sampled.clamp(min=1e-5, max=1 - 1e-5).log()).sum(
            dim=-1
        )

        return losses, td_error, q_val_sampled_action

    @staticmethod
    def _obs_to_torch(obs):
        raise NotImplementedError("Missing Obs to torch method.")

    def act(self, obs, deterministic: bool = True, temperature: float = 0.8) -> int:
        """Takes a single observation and return action"""
        with torch.no_grad():
            if deterministic:
                _, action, _ = self.q_network.get_action(self._obs_to_torch(obs))
            else:
                q_vals, _ = self.q_network(self._obs_to_torch(obs))
                q_vals = q_vals / temperature
                probs = torch.softmax(q_vals, dim=1)

                action = torch.multinomial(probs, num_samples=1).squeeze()

        return action.item()


class RDQNTrainer(RainbowDQN):
    def __init__(
        self,
        env_confs: dict,
        modelC,
        tb_writer: bool = False,
        log_writer: bool = False,
        **kwargs,
    ) -> None:
        print("Training Parameters:")
        print(kwargs)
        super().__init__(modelC, {}, **kwargs)

        self.n_actors = len(env_confs)

        self.actors = [Actor(env_confs[i]) for i in range(self.n_actors)]

        # self.replay_buffer = NStepBuffer(
        #     self.buffer_capacity,
        #     lambda x: ObservationHolder(x),
        #     self.mn_step,
        #     self.n_actors,
        #     self.gamma,
        # )
        self.replay_buffer = NStepPERBuffer(
            self.buffer_capacity,
            lambda x: ObservationHolder(x),
            self.buffer_alpha,
            self.minibatch_size,
            self.mn_step,
            self.n_actors,
            self.gamma,
        )
        # Tensors to store observations
        self.obs = ObsHolder(self.n_actors)

        # Reset all environments and store the observations
        self._reset_environments()

        # Loggers
        self.ep_records = Record("episode_stats", writer=tb_writer, save_log=log_writer)
        self.tr_records = Record("train_stats", writer=tb_writer, save_log=log_writer)

        self.episode_counts = 0
        self.total_steps = 0

    def _reset_environments(self):
        for actor in self.actors:
            actor.coms.send(("reset", None))

        for i, actor in enumerate(self.actors):
            obs = actor.coms.recv()
            self.obs[i] = obs

    def _act_on_environment(self, actions, t):
        mask = np.random.random(self.n_actors) < self.epsilon(t)
        actions[mask] = np.random.randint(0, ACTION_SPACE, mask.sum())

        for _a, actor in enumerate(self.actors):
            actor.coms.send(("step", actions[_a]))
            self.total_steps += 1

        for _a, actor in enumerate(self.actors):
            nxt_obs, reward, done, info = actor.coms.recv()

            self.replay_buffer.add(self.obs[_a], actions[_a], reward, nxt_obs, done)

            self.obs[_a] = nxt_obs
            # end of an episode
            if info:
                self.obs[_a] = info.pop("next_ep_obs")
                self.episode_counts += 1
                self.ep_records.append(info)

    @staticmethod
    def _obs_to_torch(obs):
        """Single Observation Preprocessor"""
        return obs_proc(obs)

    @property
    def model_(self):
        """For saving checkpoint"""
        return self.q_network

    @model_.setter
    def model_(self, statedict):
        """For Loading checkpoint"""
        self.q_network.load_state_dict(statedict)
        self._hard_update()

    def run_training_loop(
        self,
        log_t: int = 100,
        checkpoint_t: int = 1000,
        cp_name: str = "rdqn_trainer.pth",
        sep_checks: float = -1.0,
        agent_performance_logger=None,
    ):
        print(f"Filling samples till {self.minimum_steps} samples in buffer...")
        for _ in tqdm(
            range(self.minimum_steps // self.n_actors),
            desc="Sampling Steps",
            unit="A*Sample",
            leave=False,
        ):
            self.sample(0)

        for update in tqdm(
            range(1, self.updates + 1),
            desc="Training",
            unit="Update",
            unit_scale=True,
        ):
            # Sampling Phase
            for _ in range(self.actor_steps):
                self.sample(update)

            # training phase
            for x in range(1, self.epochs + 1):
                data = self.replay_buffer.sample(
                    # self.minibatch_size
                    self.buffer_beta(update)
                )

                # presample indices for every epoch except the last
                # so at new update step the sample will include new samples
                # precomp = x < self.epochs
                # we can also set this to true as default
                stats = self.train(data, precomp=x < self.epochs)
                if x % 4:
                    self.tr_records.append(stats)

            # logging latest results
            if log_t > 0 and update % log_t == 0:
                print(
                    f"Epsilon={self.epsilon(update):.3f} | Total Steps={self.total_steps} | Ended Episodes={self.episode_counts}"
                )

                self.tr_records.log()
                self.ep_records.log()
                if agent_performance_logger is not None:
                    agent_performance_logger(self, update)

            # saving model Checkpoint
            if checkpoint_t > 0 and update % checkpoint_t == 0:
                print(f"Saving Checkpoint for update #{update}...")
                # create new cps after a part of the update
                _cp_name = (
                    cp_name
                    if update < (self.updates * sep_checks)
                    else f"{update}-{cp_name}"
                )

                save_checkpoint(
                    self, update, self.tr_records.get_latest_val("loss"), _cp_name
                )

        self.tr_records.close()
        self.ep_records.close()

        return True

    def destroy(self):
        """Stop the child processes"""
        del self.replay_buffer
        for actor in self.actors:
            actor.kill_your_self()
