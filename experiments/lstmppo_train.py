import numpy as np
from tqdm import tqdm

from algos.lstmppo.lstmppo import PPOTrainer
from eutils import DEVICE, Record
from eutils.mutil import print_param_counts, save_checkpoint, load_checkpoint
from eutils.ttutil import get_baselines, log_policy_performance

from homegym import MazeHarvest

actor_confs = [
    *[
        dict(
            height=np.random.randint(25, 35),
            width=np.random.randint(25, 35),
            env_mode="hard" if np.random.random() < 0.5 else "extreme",
            seed=1000 + x,
            max_steps=4000,
        )
        for x in range(16)
    ],
]


test_env = MazeHarvest(height=20, width=20, env_mode="medium", seed=69420)


BASELINE_SCORES = get_baselines(test_env, N_test=10)

policy_records = Record("policy_performance", writer=False, save_log=True)


def main():
    trainer = PPOTrainer(
        env_confs=actor_confs,
        updates=0,
        epochs=4,
        actor_steps=512,
        c1=0.25,  # value loss coef
        c2=0.001,  # entropy bonus coef        ,
        ratio_clip_range=0.1,
        valf_clip_range=0.1,
        eta=2.5e-4,
        gae_gamma=0.99,
        gae_lambda=0.95,
        tb_writer=False,
        log_writer=True,
        # model params
        state_size=1024,
        hidden_size=768,
        hfeature_size=1024,
        sequence_length=8,
        minibatch_size=None,
        n_batches=8,
    )
    cp_name = "lstmppo_v2_dense_large.pth"
    print("Model Info:")
    print_param_counts(trainer.model)

    print("Actors Count:", len(actor_confs))
    print("baseline Scores (random, nomoves):", BASELINE_SCORES)

    try:
        load_checkpoint(
            trainer,
            "/tmp/checkpointdir/interrupted-lstmppo_v2_dense_large.pth",
        )
    except FileNotFoundError:
        print("Checkpoint was not found.")
    except RuntimeError:
        print(
            "Unable to load checkpoint cause of model architecture mismatch!!"
        )

    updates = 1_000
    log_t = 1
    checkpoint_t = 100
    alpha = 1
    epsilon = 0.1
    trainer.ratio_clip_range = epsilon
    eta = 1e-4
    trainer.c1 = 0.5
    eta_min = 1e-5
    eps_min = 0.03

    print(f"Starting Training on Dev {DEVICE}")
    update = 0
    try:
        for update in tqdm(
            range(1, updates + 1),
            desc="Training",
            unit="Update",
            unit_scale=True,
        ):
            # sample with current policy
            trainer.ratio_clip_range = max(alpha * epsilon, eps_min)
            for pg in trainer.optimizer.param_groups:
                pg["lr"] = max(alpha * eta, eta_min)

            trainer.sample()
            avg_loss = trainer.train()

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
                    pre_func=lambda: trainer._reset_state(),
                )
                trainer.tr_records.log()
                trainer.ep_records.log()
                print("\n")

            # save the trainer checkpoint
            if checkpoint_t > 0 and update % checkpoint_t == 0:
                print(f"Saving Checkpoint for update #{update}...")
                # create new cps after 30% of the update
                _cp_name = (
                    cp_name
                    if update < (updates * 0.4)
                    else f"{update}-{cp_name}"
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
            save_checkpoint(
                trainer, update, avg_loss, "interrupted-" + cp_name
            )
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
