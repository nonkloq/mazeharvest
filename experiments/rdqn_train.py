import numpy as np

from algos.rdqn.rdqn import RDQNTrainer
from algos.rdqn.qnet import Brain
from eutils import DEVICE, Record
from eutils.mutil import get_param_count, save_checkpoint, load_checkpoint
from eutils.ttutil import log_policy_performance, get_baselines

from homegym import MazeHarvest
from homegym._envtypings import EnvParams
from homegym.constants import high_risk_dist


densenv = EnvParams(0.15, 0.2, 0.04, -0.8, 0.01, -1.7, high_risk_dist)

envch = ["easy", "medium", densenv]
actor_confs = [
    *[
        dict(
            height=20,
            width=20,
            env_mode="easy",
            seed=1000 + x,
            num_rays=np.random.randint(15, 25),
        )
        for x in range(4)
    ],
    *[
        dict(
            height=10,
            width=10,
            env_mode="easy",
            seed=2000 + x,
            num_rays=np.random.randint(15, 25),
        )
        for x in range(2)
    ],
    *[
        dict(
            height=10,
            width=10,
            env_mode=densenv,
            seed=3000 + x,
            num_rays=np.random.randint(15, 25),
        )
        for x in range(2)
    ],
    *[
        dict(
            height=20,
            width=20,
            env_mode=envch[np.random.randint(3)],
            seed=4000 + x,
            num_rays=np.random.randint(15, 25),
        )
        for x in range(4)
    ],
]


test_env = MazeHarvest(height=10, width=10, env_mode="easy", num_rays=21, seed=69420)


BASELINE_SCORES = get_baselines(test_env, N_test=10)

policy_records = Record("policy_performance", writer=False, save_log=True)


def agent_performance_logger(trainer: RDQNTrainer, t: int):
    log_policy_performance(
        lambda x: trainer.act(x, deterministic=True),
        baseline_scores=BASELINE_SCORES,
        policy_records=policy_records,
        env=test_env,
        update_no=t,
        n_test=7,
    )
    print("\n")


def main():
    trainer = RDQNTrainer(
        env_confs=actor_confs,
        modelC=Brain,
        updates=10_000,
        minibatch_size=256,
        actor_steps=2,  # env roll out
        eta=1e-4,
        gamma=0.99,
        v_min=-100,
        v_max=100,
        # buffer_capacity-1 to wait for it to become full
        minimum_steps=1000,  # Minimum Steps in replay buffer to start the training
        n_atoms=51,
        tau=0.005,
        epochs=48,
        buffer_capacity=2**18,
        buffer_alpha=0.8,
        buffer_beta_start=0.4,
        buffer_beta_end=1.0,
        # minimum_steps // n_actors * actor_steps
        rand_explore_till=300,  # high amount of random exploration for first N Steps
        max_epsilon=0.3,
        min_epsilon=0.01,
        mn_step=(4, 20),
        tb_writer=False,
        log_writer=True,
    )

    try:
        load_checkpoint(trainer, "/tmp/checkpointdir/rdqn_trainer-interrupted.pth")
    except FileNotFoundError:
        print("Checkpoint is not found.")
    except RuntimeError:
        print("Unable to load checkpoint cause of model architecture mismatch!!")
        # print(e)

    print("Model Param Count:", get_param_count(trainer.q_network))

    print("Actors Count:", len(actor_confs))
    print("baseline Scores (random, nomoves):", BASELINE_SCORES)

    print(f"Starting Training on Dev {DEVICE}\n")
    train_complete = False
    try:
        train_complete = trainer.run_training_loop(
            log_t=100,
            checkpoint_t=1000,
            cp_name="rdqn_trainer-v1.pth",
            sep_checks=0.6,
            agent_performance_logger=agent_performance_logger,
        )
    except KeyboardInterrupt:
        print("\nTraining loop interrupted")
    except Exception as e:
        raise e
    finally:
        if not train_complete:
            save_checkpoint(trainer, -1, None, "rdqn_trainer-interrupted.pth")
            print("Latest Checkpoint saved.")

        trainer.destroy()
        trainer.tr_records.close()
        trainer.ep_records.close()
        policy_records.close()

        print(
            "\nActor Process Killed [x] | Checkpoint Saved [x] | Closed All Records [x]"
        )


if __name__ == "__main__":
    main()
