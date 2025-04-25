import argparse
from datetime import datetime
import os
import time
from typing import Dict, Tuple, Union

import gymnasium
import numpy as np
import pytz


class TestTrainer:
    def __init__(
        self, envs: gymnasium.vector.VectorEnv, convergence_f
    ) -> None:
        self._converge_buffer: Dict[str, Union[int, float]] = {
            "dones": 0,
            "truncs": 0,
            "reward": float("-inf"),
        }
        self._convf = convergence_f
        self.envs = envs

        self.n_actors = self.envs.num_envs

        self.observation_space = self.envs.single_observation_space.shape[0]
        self.action_space = self.envs.single_action_space.n

    def get_train_stats(self, update_t: int):
        eps = self._converge_buffer["dones"] + self._converge_buffer["truncs"]
        rew = self._converge_buffer["reward"]
        steps = self.calc_num_steps(update_t)
        return eps, steps, rew

    def calc_num_steps(self, update_t: int) -> int:
        raise NotImplementedError("Steps Count Calculator Missing")

    def run_training_loop(self, log_t: int) -> Tuple[int, float]:
        raise NotImplementedError("Trainer Missing")

    def prep_for_eval(self):
        pass


def eval_func(env, trainer: TestTrainer, n_test: int = 3):
    rewards = np.zeros(n_test, dtype=np.float32)
    for x in range(n_test):
        state, _ = env.reset()
        done = False
        total_reward = 0
        trainer.prep_for_eval()
        while not done:
            ac = trainer.act(state, deterministic=True)
            state, reward, done, trunc, _ = env.step(ac)
            total_reward += reward
            done = done or trunc

        rewards[x] = total_reward

    return np.mean(rewards), np.std(rewards)


def convergence_checker(env_id: str, threshold: int):
    """
    Returns a function to check if the agent is converged or not by eval

    Right Now it only considere the scores from the latest eval & it does not
    mean it is actually converged.
    """
    eval_env = gymnasium.make(env_id)

    def convergence_checker(
        self: TestTrainer, d=None, t=None, check=False, log=False, eval=False
    ):
        if log:
            trunc = self._converge_buffer.get("truncs")
            done = self._converge_buffer.get("dones")
            reward = self._converge_buffer.get("reward")

            print(
                f"Total Episodes={trunc + done} | Truncated={trunc} | Dones={done}"
            )
            print(f"Latest Eval Reward: {reward:.2f}")
            return

        if check:
            return self._converge_buffer.get("reward") >= threshold

        if eval:
            mu, _ = eval_func(eval_env, self, n_test=3)
            self._converge_buffer["reward"] = int(mu)
            return

        self._converge_buffer["truncs"] += np.sum(t)
        self._converge_buffer["dones"] += np.sum(d)

    return convergence_checker, eval_env


TOY_ENVS = [
    ("CartPole-v1", 500),
    ("MountainCar-v0", -110),
    ("LunarLander-v3", 200),
    ("Acrobot-v1", -100),
]


class ToyTester:
    def __init__(
        self,
        trainer_cls,
        parent_folder: str,
        n_envs: int = 1,
        eval_runs: int = 100,
        deterministic: bool = True,
        log_t: int = 100,
        max_steps: int = -1,
        **kwargs,
    ):
        self.trainer_cls = trainer_cls
        self.targs = kwargs

        self.eval_runs = eval_runs
        self.n_envs = n_envs

        self._action_type = deterministic

        self.__parent_folder = parent_folder
        self.log_t = log_t

        self.maxsteps = max_steps

    def _train(self, env_id: str, convf):
        envs = gymnasium.make_vec(
            env_id,
            num_envs=self.n_envs,
            vectorization_mode="sync",
            max_episode_steps=self.maxsteps,
        )

        trainer: TestTrainer = self.trainer_cls(
            envs=envs, convergence_f=convf, **self.targs
        )

        start = time.time()
        # env will be closed by the trainer
        update_steps, loss = trainer.run_training_loop(log_t=self.log_t)
        if not envs.closed:
            print("Trainer Failed to Close Environment.")
            envs.close()

        return trainer, time.time() - start, update_steps, loss

    def _results_to_md(
        self, results: dict, ts, test_name: str, test_desc: str
    ):
        _time = datetime.fromtimestamp(
            ts, tz=pytz.timezone("Asia/Kolkata")
        ).strftime("%Y-%m-%d %H:%M:%S IST")

        rmsg = (
            ""
            if self.maxsteps is None
            else f"> Env Steps Truncated after {self.maxsteps} steps."
        )
        args_table = "| Key | Value |\n| --- | --- |\n"
        for key, value in self.targs.items():
            args_table += f"| {key} | {value} |\n"

        result_table = """
| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
"""

        for key, res in results.items():
            result_table += f"| {key} | {res['mu']:.2f} | {res['sigma']:.2f} | {res['tt']:.2f} | {res['updates']} | {res['episodes']} | {res['steps']} | {res['loss']:.4f} | {res['final_reward']} | {res['solvet']} |\n"

        md_section = f"""
# {test_name.capitalize()} [{_time}]

{test_desc}

### Trainer Args 

{args_table}

### Results 

One Common Model & HyperParameters used for all {len(results)} snvironments, trained model evaluated for {self.eval_runs} runs.

> Actor(s) Count=`{self.n_envs}`. Action Mode Deterministic=`{self._action_type}`.
> Early Termination Enabled: Training will stop once the agent reaches the required amount of scores to be considered solved (By Eval at every {self.log_t} update).
{rmsg}

{result_table}

"""

        md_path = f"{self.__parent_folder}/ttresults.md"

        with open(md_path, "a") as md_file:
            md_file.write(md_section)

        print(f'Test Results saved at "{md_path}"')

    def run_test(self, test_name: str, test_desc: str):
        ts = time.time()
        results = {}
        for env_id, solvest in TOY_ENVS:
            print(
                f"\nStarting Training on {env_id}, Actors #[{self.n_envs}]..."
            )
            convergence_function, test_env = convergence_checker(
                env_id, solvest
            )
            trainer, timetook, update_steps, loss = self._train(
                env_id, convergence_function
            )
            timetook /= 60
            print(f"Training Finished in {timetook:.2f} Minutes.")

            print("Evaluating Model...")
            mean, std = eval_func(test_env, trainer, self.eval_runs)
            test_env.close()
            print(f"Evaluation Results: mu={mean:.2f} sigma={std:.2f}")
            episodes, steps, frew = trainer.get_train_stats(update_steps)
            results[env_id] = {
                "mu": mean,
                "sigma": std,
                "tt": timetook,
                "updates": update_steps,
                "episodes": episodes,
                "steps": steps,
                "loss": loss,
                "final_reward": frew,
                "solvet": solvest,
            }

        print("Test Ended !!")
        self._results_to_md(results, ts, test_name, test_desc)


def bootstrap(cls, fil, **kwargs):
    parser = argparse.ArgumentParser(description="Toy Tests for RL algorithm")

    parser.add_argument(
        "-t", "--title", type=str, required=True, help="Title of the test"
    )
    parser.add_argument(
        "-d",
        "--description",
        type=str,
        required=True,
        help="Description / Key Features of the algorithm.",
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Stochastic Evaluation Policy",
    )

    parser.add_argument(
        "-l", "--log", type=int, help="Log Time step.", default=100
    )

    parser.add_argument(
        "-e", "--eruns", type=int, help="Evaluation runs.", default=100
    )

    parser.add_argument(
        "-a", "--actors", type=int, help="Parallel Actor Count", default=8
    )

    parser.add_argument(
        "-m",
        "--maxenvsteps",
        type=int,
        help="Parallel Actor Count",
        default=None,
    )

    args = parser.parse_args()

    tester = ToyTester(
        cls,
        parent_folder=os.path.dirname(os.path.abspath(fil)),
        n_envs=args.actors,
        eval_runs=args.eruns,
        deterministic=not args.stochastic,
        log_t=args.log,
        max_steps=args.maxenvsteps,
        **kwargs,
    )

    try:
        print("Test starting For parameters:")
        print(kwargs)
        print()
        tester.run_test(args.title, args.description)
    except KeyboardInterrupt:
        print("Test Terminated Before Completion")
