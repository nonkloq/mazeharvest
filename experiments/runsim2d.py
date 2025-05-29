from eutils.mutil import load_params
from eutils.play_env import play
from eutils.actor import HomeGymWrapper
import torch
from algos.lstmppo.brain import Brain
from homegym import EnvParams

MODEL_PATH: str = "./agents/lstmppo_v2_dense_large.pth.ign"
HEIGHT: int = 20
WIDTH: int = 30
SEED: int = 69420
MAX_STEPS: int = 2000
ENV_MODE: str | EnvParams = "hard"

env = HomeGymWrapper(
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

agent = Brain(
    state_size=1024,
    hidden_size=768,
    inp_parser_features=1024,
    observation_dim=220,
    action_dim=10,
)

load_params(agent, MODEL_PATH)

hidden_state = list(agent.init_states(1, "cpu"))


def policy(obs, deterministic=False):
    with torch.no_grad():
        pi, _, rnn_state = agent(
            torch.tensor(obs.reshape(1, 1, -1), dtype=torch.float32),
            hidden_state,
        )
        hidden_state[0] = rnn_state[0]
        hidden_state[1] = rnn_state[1]

        if deterministic:
            action = pi.probs.argmax().item()
        else:
            action = pi.sample().item()

    return action


play(
    env,
    policy,
    fps=5,
    title="MazeHarvest",
    wait_for_quit=True,
    record_vid=False,
    video_name="test_run_lstmppo",
)
print("Simulation Ended")
