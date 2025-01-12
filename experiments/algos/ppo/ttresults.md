
# Initial test [2025-01-09 17:40:32 IST]

Implemented PPO

### Trainer Args 

| Key | Value |
| --- | --- |
| updates | 10000 |
| epochs | 15 |
| actor_steps | 1024 |
| batches | 32 |
| c1 | 0.5 |
| c2 | 0.001 |
| ratio_clip_range | 0.2 |
| valf_clip_range | None |
| eta | 0.0003 |
| gae_gamma | 0.99 |
| gae_lambda | 0.95 |


### Results 

One Common Model & HyperParameters used for all 4 snvironments, trained model evaluated for 100 runs.

> Actor(s) Count=`8`. Action Mode Deterministic=`True`.
> Early Termination Enabled: Training will stop once the agent reaches the required amount of scores to be considered solved (By Eval at every 10 update).
> Env Steps Truncated after -1 steps.


| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole-v1 | 500.00 | 0.00 | 0.33 | 20 | 3403 | 163840 | 63.8947 | 500 | 500 |
| MountainCar-v0 | -111.04 | 24.35 | 9.82 | 570 | 5201 | 4669440 | 3.6446 | -97 | -110 |
| Acrobot-v1 | -87.93 | 16.31 | 1.00 | 40 | 765 | 327680 | 12.3106 | -83 | -100 |

