import multiprocessing
import multiprocessing.connection

import numpy as np

from homegym import MazeHarvest


class HomeGymWrapper:
    def __init__(self, env_config):
        self.env = MazeHarvest(**env_config)

    def reset(self, seed):
        obs = self.env.reset(seed)

        return self._process_observation(obs)

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

        return self._process_observation(obs), reward, done, episode_info

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


def actor_process(remote: multiprocessing.connection.Connection, envargs):
    env = HomeGymWrapper(envargs)

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
    def __init__(self, envargs):
        self.coms, child_conn = multiprocessing.Pipe()
        # parent connection

        self.process = multiprocessing.Process(
            target=actor_process, args=(child_conn, envargs), daemon=True
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
