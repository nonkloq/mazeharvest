import multiprocessing
import multiprocessing.connection

import numpy as np

from homegym import MazeHarvest


class HomeGymWrapper:
    def __init__(
        self,
        env_config,
        flat_obs=False,
        max_rays: int = 30,
        include_prev_action=False,
    ):
        self.env = MazeHarvest(**env_config)
        self.flat_obs = flat_obs
        self.max_rays = max_rays
        self.include_prev_action = include_prev_action
        self._action_dim = self.env.action_space.n

    def render(self):
        return self.env.render(agent_center=True)

    def reset(self, seed=None):
        obs = self.env.reset(seed)

        if self.flat_obs:
            pobs = self._process_observation_flat(
                obs, self.max_rays, self._get_action_encoding(-1)
            )
        else:
            pobs = self._process_observation(obs)

        return pobs

    def episode_info(self):
        return self.env.episode_info()

    def step(self, action):
        obs, reward, done, trunc = self.env.step(action)
        episode_info = None
        if done or trunc:
            steps, kill_count, harvest_count = self.env.episode_info()
            episode_info = {
                "steps": steps,
                "kills": kill_count,
                "harvests": harvest_count,
                "next_ep_obs": self.reset(None),
            }

        if self.flat_obs:
            pobs = self._process_observation_flat(
                obs, self.max_rays, self._get_action_encoding(action)
            )
        else:
            pobs = self._process_observation(obs)
        return pobs, reward, done, episode_info

    def _get_action_encoding(self, action) -> np.ndarray:
        if not self.include_prev_action:
            return []

        ace = np.zeros(self._action_dim)
        if action >= 0:
            ace[action] = 1

        return ace

    @staticmethod
    def _process_observation(obs):
        (
            perception,
            loot_heuristics,
            mole_heuristics,
            damage_directions,
            player_stats,
        ) = obs
        return perception, np.concatenate(
            (loot_heuristics, mole_heuristics, damage_directions, player_stats)
        )

    @staticmethod
    def _process_observation_flat(obs, allowed_rays: int, ace):
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
        return np.concatenate(
            (
                player_stats,
                loot_heuristics,
                mole_heuristics,
                damage_directions,
                perception.reshape(-1),
                ace,
            )
        )


def actor_process(remote: multiprocessing.connection.Connection, envargs):
    env = HomeGymWrapper(envargs[0], **envargs[1])

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(env.step(data))
            elif cmd == "reset":
                remote.send(env.reset(data))
            elif cmd == "close":
                break
            else:
                raise Exception("Uknown Command")
    except Exception as e:
        raise e
    finally:
        remote.close()


class Actor:
    def __init__(self, envargs, **kwargs):
        self.coms, child_conn = multiprocessing.Pipe()
        # parent connection

        self.process = multiprocessing.Process(
            target=actor_process,
            args=(child_conn, (envargs, kwargs)),
            daemon=True,
        )
        self.process.start()

    def kill_your_self(self):
        if not self.process.is_alive():
            return

        if self.coms:
            self.coms.send(("close", None))

        self.process.join(timeout=5)

        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
