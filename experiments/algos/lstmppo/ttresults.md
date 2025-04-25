
# Init test [2025-04-25 15:18:20 IST]

LSTM with PPO, only lunar lander

### Trainer Args 

| Key | Value |
| --- | --- |
| updates | 10000 |
| epochs | 4 |
| actor_steps | 256 |
| c1 | 0.2 |
| c2 | 0.001 |
| ratio_clip_range | 0.2 |
| valf_clip_range | 0.5 |
| eta | 0.001 |
| gae_gamma | 0.99 |
| gae_lambda | 0.95 |
| state_size | 64 |
| hidden_size | 128 |
| hfeature_size | 32 |
| sequence_length | 8 |
| minibatch_size | None |
| n_batches | 2 |


### Results 

One Common Model & HyperParameters used for all 1 snvironments, trained model evaluated for 100 runs.

> Actor(s) Count=`8`. Action Mode Deterministic=`True`.
> Early Termination Enabled: Training will stop once the agent reaches the required amount of scores to be considered solved (By Eval at every 20 update).



| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LunarLander-v3 | 198.47 | 92.01 | 3.92 | 640 | 5008 | 1310720 | 19.2725 | 247 | 200 |


