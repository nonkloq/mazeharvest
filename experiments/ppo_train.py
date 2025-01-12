import numpy as np
from tqdm import tqdm

from algos.ppo.brain import Brain
from algos.ppo.ppo import PPOTrainer
from eutils import DEVICE, Record
from eutils.mutil import print_param_counts, save_checkpoint
from eutils.ttutil import get_baselines, log_policy_performance

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


def main():
    model = Brain()

    print("Model Info:")
    print_param_counts(model)

    print("Actors Count:", len(actor_confs))
    print("baseline Scores (random, nomoves):", BASELINE_SCORES)

    trainer = PPOTrainer(
        env_confs=actor_confs,
        model=model,
        updates=0,
        epochs=10,
        actor_steps=512,
        batches=16,
        c1=0.5,  # value loss coef
        c2=0.001,  # entropy bonus coef        ,
        ratio_clip_range=0.2,
        valf_clip_range=None,
        eta=2.5e-4,
        gae_gamma=0.99,
        gae_lambda=0.95,
        tb_writer=False,
        log_writer=True,
    )

    updates = 2_000
    log_t = 50
    checkpoint_t = 100
    cp_name = "ppo-v1.pth"
    alpha = 1
    epsilon = 0.2
    trainer.ratio_clip_range = epsilon
    eta = 3e-4
    trainer.c1 = 0.5
    eta_min = 1e-5
    eps_min = 0.03

    print(f"Starting Training on Dev {DEVICE}")
    update = 0
    try:
        for update in tqdm(
            range(1, updates + 1), desc="Training", unit="Update", unit_scale=True
        ):
            # sample with current policy
            trainer.ratio_clip_range = max(alpha * epsilon, eps_min)
            for pg in trainer.optimizer.param_groups:
                pg["lr"] = max(alpha * eta, eta_min)

            samples = trainer.sample()

            # train the model
            avg_loss = trainer.train(samples)

            # Log training status
            if log_t > 0 and update % log_t == 0:
                print(f"\nUpdate #{update}: Avg. Loss {avg_loss:.5f}")
                _ = log_policy_performance(
                    trainer.act,
                    BASELINE_SCORES,
                    policy_records,
                    test_env,
                    update,
                    n_test=10,
                )
                trainer.tr_records.log()
                trainer.ep_records.log()
                print("\n")

            # save the trainer checkpoint
            if checkpoint_t > 0 and update % checkpoint_t == 0:
                print(f"Saving Checkpoint for update #{update}...")
                # create new cps after 30% of the update
                _cp_name = (
                    cp_name if update < (updates * 0.4) else f"{update}-{cp_name}"
                )
                save_checkpoint(trainer, update, avg_loss, _cp_name)

            alpha = 1 - update / updates  # linear annealing

    except KeyboardInterrupt:
        print("\nTraining loop interrupted")

    except Exception as e:
        # print(f"\nError:\n{e}\n")
        raise e
    finally:
        if (update < updates) and (update % checkpoint_t) != 0:
            save_checkpoint(trainer, update, avg_loss, "interrupted-" + cp_name)
            print(f"Latest Checkpoint saved for update {update}")

        trainer.destroy()
        trainer.tr_records.close()
        trainer.ep_records.close()
        policy_records.close()

        print(
            "\nActor Process Killed [x] | Checkpoint Saved [x] | Closed All Records [x]"
        )


if __name__ == "__main__":
    main()
