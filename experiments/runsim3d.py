import torch

from algos.lstmppo.brain import Brain
from eutils.actor import HomeGymWrapper
from eutils.mutil import load_params

from homegym import EnvParams
from homegym.render3d import RealLifeSim

MODEL_PATH: str = "./agents/lstmppo_v2_dense_large.pth.ign"
HEIGHT: int = 20
WIDTH: int = 30
SEED: int = 69420
MAX_STEPS: int = 2000
ENV_MODE: str | EnvParams = "hard"


class Player:
    def __init__(self, deterministic=True):
        self.env = HomeGymWrapper(
            dict(
                height=HEIGHT,
                width=WIDTH,
                seed=SEED,
                env_mode=ENV_MODE,
                max_steps=MAX_STEPS,
            ),
            flat_obs=True,
            max_rays=30,
            include_prev_action=True,
        )
        self.deterministic = deterministic
        self.agent = Brain(
            state_size=1024,
            hidden_size=768,
            inp_parser_features=1024,
            observation_dim=220,
            action_dim=10,
        )

        load_params(self.agent, MODEL_PATH)

        self.hidden_state = list(self.agent.init_states(1, "cpu"))
        self.obs = self.env.reset()

    def act_policy(self, action):
        try:
            (
                self.obs,
                _,
                done,
                ep_i,
            ) = self.env.step(action)
        except AssertionError:
            done = True

        return done

    def policy(
        self,
    ):
        with torch.no_grad():
            pi, _, rnn_state = self.agent(
                torch.tensor(
                    self.obs.reshape(1, 1, -1),
                    dtype=torch.float32,
                ),
                self.hidden_state,
            )

        if self.deterministic:
            action = pi.probs.argmax().item()
        else:
            action = pi.sample.sample().item()

        self.hidden_state = rnn_state
        return self.act_policy(action)


player = Player(deterministic=True)
simulator = RealLifeSim(
    env=player.env.env, action_interval=0.2, policy_function=player.policy
)

try:
    simulator.run()
finally:
    simulator.destroy()
print("Simulation Ended")
