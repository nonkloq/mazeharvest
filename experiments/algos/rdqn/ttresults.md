
# C51v [2024-12-31 22:11:11 UTC]

Categorical DQN with Vectorized environments

### Trainer Args 

| Key | Value |
| --- | --- |
| updates | 10000 |
| minibatch_size | 64 |
| actor_steps | 5 |
| eta | 0.0001 |
| gamma | 0.99 |
| v_min | -100 |
| minimum_steps | 3000 |
| v_max | 100 |
| n_atoms | 51 |
| target_update_interval | 12 |
| epochs | 32 |
| buffer_capacity | 1048576 |
| max_epsilon | 0.2 |
| min_epsilon | 0.01 |


### Results 

One Common Model & HyperParameters used for all 4 snvironments, trained model evaluated for 100 runs.

> Actor(s) Count=`8`. Action Mode Deterministic=`True`.
> Early Termination Enabled: Training will stop once the agent reaches the required amount of scores to be considered solved (By Eval at every 100 update).


| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole-v1 | 498.29 | 17.01 | 4.01 | 4200 | 882 | 168000 | 1.3538 | 500 | 500 |
| MountainCar-v0 | -110.47 | 10.90 | 8.97 | 6400 | 1602 | 256000 | 2.6316 | -94 | -110 |
| LunarLander-v3 | 57.23 | 167.70 | 2.39 | 2500 | 228 | 100000 | 3.0777 | 205 | 200 |
| Acrobot-v1 | -91.08 | 27.11 | 0.84 | 1100 | 229 | 44000 | 3.0726 | -77 | -100 |



# C51-r [2025-01-01 10:06:42 UTC]

Categorical DQN with Vectorized environments, Soft Update at every train step & Decoupling next_obs action selection from target net. (Double QL)

### Trainer Args 

| Key | Value |
| --- | --- |
| updates | 10000 |
| minibatch_size | 1024 |
| actor_steps | 5 |
| eta | 0.00025 |
| gamma | 0.99 |
| v_min | -100 |
| minimum_steps | 3000 |
| v_max | 100 |
| n_atoms | 51 |
| tau | 0.005 |
| epochs | 8 |
| buffer_capacity | 1048576 |
| max_epsilon | 0.2 |
| min_epsilon | 0.01 |


### Results 

One Common Model & HyperParameters used for all 4 snvironments, trained model evaluated for 100 runs.

> Actor(s) Count=`8`. Action Mode Deterministic=`True`.
> Early Termination Enabled: Training will stop once the agent reaches the required amount of scores to be considered solved (By Eval at every 100 update).


| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole-v1 | 498.40 | 10.22 | 3.01 | 3000 | 851 | 120000 | 2.8820 | 500 | 500 |
| MountainCar-v0 | -139.57 | 28.61 | 4.51 | 4100 | 885 | 164000 | 2.6695 | -107 | -110 |
| LunarLander-v3 | 149.58 | 128.42 | 3.07 | 2600 | 254 | 104000 | 3.2776 | 248 | 200 |
| Acrobot-v1 | -86.29 | 21.58 | 2.04 | 1800 | 346 | 72000 | 3.1657 | -90 | -100 |



# C51-r Test2 [2025-01-01 10:21:25 UTC]

Categorical DQN with Vectorized environments, Soft Update at every train step & Decoupling next_obs action and value selection from target net. (Double QL). New HyperParams.

### Trainer Args 

| Key | Value |
| --- | --- |
| updates | 10000 |
| minibatch_size | 42 |
| actor_steps | 5 |
| eta | 0.0003 |
| gamma | 0.99 |
| v_min | -100 |
| minimum_steps | 3000 |
| v_max | 100 |
| n_atoms | 51 |
| tau | 0.005 |
| epochs | 128 |
| buffer_capacity | 1048576 |
| max_epsilon | 0.2 |
| min_epsilon | 0.01 |


### Results 

One Common Model & HyperParameters used for all 4 snvironments, trained model evaluated for 100 runs.

> Actor(s) Count=`8`. Action Mode Deterministic=`True`.
> Early Termination Enabled: Training will stop once the agent reaches the required amount of scores to be considered solved (By Eval at every 100 update).


| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole-v1 | 500.00 | 0.00 | 3.65 | 800 | 269 | 32000 | 0.4376 | 500 | 500 |
| MountainCar-v0 | -172.54 | 17.96 | 9.77 | 1015 | 209 | 40600 | 0.6072 | -158 | -110 |
| LunarLander-v3 | 207.73 | 89.97 | 2.43 | 400 | 89 | 16000 | 2.0948 | 212 | 200 |
| Acrobot-v1 | -92.78 | 36.09 | 0.73 | 200 | 15 | 8000 | 3.2250 | -87 | -100 |



# C51-r Test3 [2025-01-01 11:32:52 UTC]

Categorical DQN with Vectorized environments, Soft Update at every train step & Decoupling next_obs action and value selection from target net. (Double QL). New HyperParams. Faster Soft Update.

### Trainer Args 

| Key | Value |
| --- | --- |
| updates | 10000 |
| minibatch_size | 42 |
| actor_steps | 12 |
| eta | 0.00025 |
| gamma | 0.99 |
| v_min | -100 |
| minimum_steps | 3000 |
| v_max | 100 |
| n_atoms | 51 |
| tau | 0.003 |
| epochs | 128 |
| buffer_capacity | 1048576 |
| max_epsilon | 0.2 |
| min_epsilon | 0.01 |


### Results 

One Common Model & HyperParameters used for all 4 snvironments, trained model evaluated for 100 runs.

> Actor(s) Count=`8`. Action Mode Deterministic=`True`.
> Early Termination Enabled: Training will stop once the agent reaches the required amount of scores to be considered solved (By Eval at every 100 update).


| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole-v1 | 500.00 | 0.00 | 3.12 | 700 | 731 | 67200 | 1.7447 | 500 | 500 |
| MountainCar-v0 | -133.79 | 29.37 | 8.90 | 1400 | 703 | 134400 | 0.6691 | -107 | -110 |
| LunarLander-v3 | 72.25 | 152.99 | 2.77 | 600 | 257 | 57600 | 2.9835 | 249 | 200 |
| Acrobot-v1 | -89.71 | 17.14 | 1.80 | 400 | 193 | 38400 | 2.8941 | -93 | -100 |



# C51-r pbuf [2025-01-01 18:52:24 UTC]

All the above changes + Prioritized replay buffer.

### Trainer Args 

| Key | Value |
| --- | --- |
| updates | 10000 |
| minibatch_size | 42 |
| actor_steps | 12 |
| eta | 0.00025 |
| gamma | 0.99 |
| v_min | -100 |
| v_max | 100 |
| minimum_steps | 3000 |
| n_atoms | 51 |
| tau | 0.003 |
| epochs | 128 |
| buffer_capacity | 65536 |
| buffer_alpha | 0.8 |
| buffer_beta_start | 0.4 |
| max_epsilon | 0.2 |
| min_epsilon | 0.01 |


### Results 

One Common Model & HyperParameters used for all 4 snvironments, trained model evaluated for 100 runs.

> Actor(s) Count=`8`. Action Mode Deterministic=`True`.
> Early Termination Enabled: Training will stop once the agent reaches the required amount of scores to be considered solved (By Eval at every 100 update).


| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole-v1 | 499.62 | 3.78 | 7.06 | 500 | 597 | 48000 | 0.0612 | 500 | 500 |
| MountainCar-v0 | -107.83 | 11.73 | 35.09 | 2000 | 1005 | 192000 | 0.0066 | -104 | -110 |
| LunarLander-v3 | 109.39 | 201.26 | 4.94 | 700 | 172 | 67200 | 0.0176 | 244 | 200 |
| Acrobot-v1 | -83.78 | 15.65 | 1.96 | 300 | 149 | 28800 | 0.1335 | -79 | -100 |



# C51-r pbuff a1 [2025-01-02 20:16:30 UTC]

All the above changes + Prioritized replay buffer. First attempt to make the buffer fast. 

### Trainer Args 

| Key | Value |
| --- | --- |
| updates | 10000 |
| minibatch_size | 42 |
| actor_steps | 12 |
| eta | 0.00025 |
| gamma | 0.99 |
| v_min | -100 |
| v_max | 100 |
| minimum_steps | 3000 |
| n_atoms | 51 |
| tau | 0.003 |
| epochs | 128 |
| buffer_capacity | 65536 |
| buffer_alpha | 0.8 |
| buffer_beta_start | 0.4 |
| max_epsilon | 0.2 |
| min_epsilon | 0.01 |


### Results 

One Common Model & HyperParameters used for all 4 snvironments, trained model evaluated for 100 runs.

> Actor(s) Count=`8`. Action Mode Deterministic=`True`.
> Early Termination Enabled: Training will stop once the agent reaches the required amount of scores to be considered solved (By Eval at every 100 update).


| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole-v1 | 500.00 | 0.00 | 11.03 | 1100 | 669 | 105600 | 0.0068 | 500 | 500 |
| MountainCar-v0 | -200.00 | 0.00 | 28.30 | 2304 | 1045 | 221184 | 0.0052 | -200 | -110 |
| LunarLander-v3 | 165.05 | 97.59 | 5.03 | 600 | 155 | 57600 | 0.0630 | 210 | 200 |
| Acrobot-v1 | -99.58 | 20.45 | 1.77 | 200 | 76 | 19200 | 0.1972 | -97 | -100 |



# C51-r pbuff a2 [2025-01-02 22:31:20 UTC]

All the above changes + Fast Prioritized replay buffer. smoll buffer size.

### Trainer Args 

| Key | Value |
| --- | --- |
| updates | 10000 |
| minibatch_size | 42 |
| actor_steps | 12 |
| eta | 0.0003 |
| gamma | 0.99 |
| v_min | -100 |
| v_max | 100 |
| minimum_steps | 3000 |
| n_atoms | 51 |
| tau | 0.003 |
| epochs | 128 |
| buffer_capacity | 16384 |
| buffer_alpha | 0.8 |
| buffer_beta_start | 0.4 |
| max_epsilon | 0.2 |
| min_epsilon | 0.01 |


### Results 

One Common Model & HyperParameters used for all 4 snvironments, trained model evaluated for 100 runs.

> Actor(s) Count=`8`. Action Mode Deterministic=`True`.
> Early Termination Enabled: Training will stop once the agent reaches the required amount of scores to be considered solved (By Eval at every 100 update).


| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole-v1 | 500.00 | 0.00 | 1.20 | 200 | 204 | 19200 | 0.0297 | 500 | 500 |
| MountainCar-v0 | -149.50 | 29.06 | 15.45 | 1314 | 637 | 126144 | 0.0175 | -139 | -110 |
| LunarLander-v3 | 175.07 | 119.21 | 5.58 | 700 | 198 | 67200 | 0.0467 | 225 | 200 |
| Acrobot-v1 | -98.72 | 15.63 | 1.45 | 200 | 62 | 19200 | 0.0359 | -90 | -100 |



# C51-r pbuff a3 [2025-01-03 19:51:49 UTC]

New Hyper params - Pbuf with multiprocess is slower than thread

### Trainer Args 

| Key | Value |
| --- | --- |
| updates | 10000 |
| minibatch_size | 64 |
| actor_steps | 3 |
| eta | 0.0003 |
| gamma | 0.99 |
| v_min | -100 |
| v_max | 100 |
| minimum_steps | 3000 |
| n_atoms | 51 |
| tau | 0.005 |
| epochs | 64 |
| buffer_capacity | 10000 |
| buffer_alpha | 0.8 |
| buffer_beta_start | 0.4 |
| buffer_beta_end | 1.0 |
| rand_explore_till | 50 |
| max_epsilon | 0.2 |
| min_epsilon | 0.01 |


### Results 

One Common Model & HyperParameters used for all 4 snvironments, trained model evaluated for 100 runs.

> Actor(s) Count=`8`. Action Mode Deterministic=`True`.
> Early Termination Enabled: Training will stop once the agent reaches the required amount of scores to be considered solved (By Eval at every 100 update).


| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole-v1 | 500.00 | 0.00 | 6.35 | 1400 | 451 | 33600 | 0.0039 | 500 | 500 |
| MountainCar-v0 | -124.23 | 31.61 | 15.49 | 2800 | 359 | 67200 | 0.0030 | -110 | -110 |
| LunarLander-v3 | 97.50 | 114.62 | 7.47 | 1600 | 138 | 38400 | 0.0106 | 268 | 200 |
| Acrobot-v1 | -98.34 | 33.38 | 1.79 | 500 | 53 | 12000 | 0.0451 | -80 | -100 |



# Rainbow dqn nstep t1 [2025-01-04 19:10:29 UTC]

Added NStep PER Buffer, using loss for priority updates in PER buffer.

### Trainer Args 

| Key | Value |
| --- | --- |
| updates | 10000 |
| minibatch_size | 64 |
| actor_steps | 3 |
| eta | 0.0003 |
| gamma | 0.99 |
| v_min | -100 |
| v_max | 100 |
| minimum_steps | 3000 |
| n_atoms | 51 |
| tau | 0.005 |
| epochs | 64 |
| buffer_capacity | 10000 |
| buffer_alpha | 0.8 |
| buffer_beta_start | 0.4 |
| buffer_beta_end | 1.0 |
| rand_explore_till | 50 |
| max_epsilon | 0.2 |
| min_epsilon | 0.01 |
| mn_step | (4, 7) |


### Results 

One Common Model & HyperParameters used for all 4 snvironments, trained model evaluated for 100 runs.

> Actor(s) Count=`8`. Action Mode Deterministic=`True`.
> Early Termination Enabled: Training will stop once the agent reaches the required amount of scores to be considered solved (By Eval at every 100 update).


| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole-v1 | 500.00 | 0.00 | 3.96 | 600 | 335 | 14400 | 1.6585 | 500 | 500 |
| MountainCar-v0 | -200.00 | 0.00 | 11.51 | 1418 | 168 | 34032 | 0.0162 | -200 | -110 |
| LunarLander-v3 | 93.65 | 151.30 | 13.14 | 1600 | 103 | 38400 | 0.9351 | 205 | 200 |
| Acrobot-v1 | -95.34 | 32.24 | 2.21 | 400 | 46 | 9600 | 1.1441 | -96 | -100 |



# Rainbow dqn nstep t2 [2025-01-05 01:20:22 IST]

Added NStep PER Buffer, PER Prior update using temporal difference from q vals

### Trainer Args 

| Key | Value |
| --- | --- |
| updates | 10000 |
| minibatch_size | 64 |
| actor_steps | 3 |
| eta | 0.0003 |
| gamma | 0.99 |
| v_min | -100 |
| v_max | 100 |
| minimum_steps | 3000 |
| n_atoms | 51 |
| tau | 0.005 |
| epochs | 64 |
| buffer_capacity | 10000 |
| buffer_alpha | 0.8 |
| buffer_beta_start | 0.4 |
| buffer_beta_end | 1.0 |
| rand_explore_till | 50 |
| max_epsilon | 0.2 |
| min_epsilon | 0.01 |
| mn_step | (4, 7) |


### Results 

One Common Model & HyperParameters used for all 3 snvironments, trained model evaluated for 100 runs.

> Actor(s) Count=`8`. Action Mode Deterministic=`True`.
> Early Termination Enabled: Training will stop once the agent reaches the required amount of scores to be considered solved (By Eval at every 100 update).


| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole-v1 | 500.00 | 0.00 | 5.54 | 800 | 406 | 19200 | 0.2753 | 500 | 500 |
| LunarLander-v3 | 62.29 | 135.65 | 8.62 | 1100 | 109 | 26400 | 0.1482 | 209 | 200 |
| Acrobot-v1 | -91.73 | 47.93 | 3.13 | 500 | 64 | 12000 | 0.4669 | -89 | -100 |



# Rainbow dqn nstep t3 [2025-01-05 02:53:21 IST]

loss for prior updates & single qnet action for both train steps and bug fix Nstep buffer.

### Trainer Args 

| Key | Value |
| --- | --- |
| updates | 10000 |
| minibatch_size | 64 |
| actor_steps | 3 |
| eta | 0.0003 |
| gamma | 0.99 |
| v_min | -100 |
| v_max | 100 |
| minimum_steps | 3000 |
| n_atoms | 51 |
| tau | 0.005 |
| epochs | 64 |
| buffer_capacity | 10000 |
| buffer_alpha | 0.8 |
| buffer_beta_start | 0.4 |
| buffer_beta_end | 1.0 |
| rand_explore_till | 50 |
| max_epsilon | 0.2 |
| min_epsilon | 0.01 |
| mn_step | (4, 7) |


### Results 

One Common Model & HyperParameters used for all 4 snvironments, trained model evaluated for 100 runs.

> Actor(s) Count=`8`. Action Mode Deterministic=`True`.
> Early Termination Enabled: Training will stop once the agent reaches the required amount of scores to be considered solved (By Eval at every 100 update).


| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole-v1 | 500.00 | 0.00 | 5.27 | 800 | 399 | 19200 | 1.0413 | 500 | 500 |
| MountainCar-v0 | -200.00 | 0.00 | 5.71 | 816 | 96 | 19584 | 0.0960 | -200 | -110 |
| LunarLander-v3 | -103.30 | 441.89 | 4.18 | 700 | 68 | 16800 | 0.3374 | 202 | 200 |
| Acrobot-v1 | -105.87 | 63.11 | 2.15 | 400 | 36 | 9600 | 1.1821 | -92 | -100 |



# Rainbow dqn nstep t4 [2025-01-05 03:30:13 IST]

everything is same as above but only using M-N samples

### Trainer Args 

| Key | Value |
| --- | --- |
| updates | 10000 |
| minibatch_size | 64 |
| actor_steps | 3 |
| eta | 0.0003 |
| gamma | 0.99 |
| v_min | -100 |
| v_max | 100 |
| minimum_steps | 3000 |
| n_atoms | 51 |
| tau | 0.005 |
| epochs | 64 |
| buffer_capacity | 10000 |
| buffer_alpha | 0.8 |
| buffer_beta_start | 0.4 |
| buffer_beta_end | 1.0 |
| rand_explore_till | 50 |
| max_epsilon | 0.2 |
| min_epsilon | 0.01 |
| mn_step | (4, 7) |


### Results 

One Common Model & HyperParameters used for all 4 snvironments, trained model evaluated for 100 runs.

> Actor(s) Count=`8`. Action Mode Deterministic=`True`.
> Early Termination Enabled: Training will stop once the agent reaches the required amount of scores to be considered solved (By Eval at every 100 update).


| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole-v1 | 484.05 | 27.99 | 4.17 | 700 | 148 | 16800 | 0.0526 | 500 | 500 |
| MountainCar-v0 | -200.00 | 0.00 | 0.56 | 216 | 24 | 5184 | 0.4515 | -200 | -110 |
| LunarLander-v3 | 94.88 | 138.62 | 21.20 | 3000 | 217 | 72000 | 0.0087 | 236 | 200 |
| Acrobot-v1 | -100.60 | 20.73 | 2.12 | 400 | 39 | 9600 | 0.5368 | -84 | -100 |



# Rainbow dqn nstep t5 [2025-01-05 13:29:32 IST]

more bug fixes and seperate qnet passes at train. 

### Trainer Args 

| Key | Value |
| --- | --- |
| updates | 10000 |
| minibatch_size | 64 |
| actor_steps | 3 |
| eta | 0.0003 |
| gamma | 0.99 |
| v_min | -100 |
| v_max | 100 |
| minimum_steps | 3000 |
| n_atoms | 51 |
| tau | 0.005 |
| epochs | 64 |
| buffer_capacity | 10000 |
| buffer_alpha | 0.8 |
| buffer_beta_start | 0.4 |
| buffer_beta_end | 1.0 |
| rand_explore_till | 50 |
| max_epsilon | 0.2 |
| min_epsilon | 0.01 |
| mn_step | (4, 7) |


### Results 

One Common Model & HyperParameters used for all 4 snvironments, trained model evaluated for 100 runs.

> Actor(s) Count=`8`. Action Mode Deterministic=`True`.
> Early Termination Enabled: Training will stop once the agent reaches the required amount of scores to be considered solved (By Eval at every 100 update).


| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole-v1 | 500.00 | 0.00 | 3.97 | 600 | 350 | 14400 | 3.2284 | 500 | 500 |
| MountainCar-v0 | -104.65 | 8.82 | 13.16 | 1400 | 183 | 33600 | 1.4150 | -108 | -110 |
| LunarLander-v3 | 88.21 | 156.13 | 10.03 | 1200 | 115 | 28800 | 1.0379 | 207 | 200 |
| Acrobot-v1 | -83.28 | 14.04 | 2.27 | 400 | 52 | 9600 | 0.5597 | -97 | -100 |



# Rainbow dqn nstep t6 [2025-01-05 14:02:56 IST]

same as above but single qnet pass for both updates

### Trainer Args 

| Key | Value |
| --- | --- |
| updates | 10000 |
| minibatch_size | 64 |
| actor_steps | 3 |
| eta | 0.0003 |
| gamma | 0.99 |
| v_min | -100 |
| v_max | 100 |
| minimum_steps | 3000 |
| n_atoms | 51 |
| tau | 0.005 |
| epochs | 64 |
| buffer_capacity | 10000 |
| buffer_alpha | 0.8 |
| buffer_beta_start | 0.4 |
| buffer_beta_end | 1.0 |
| rand_explore_till | 50 |
| max_epsilon | 0.2 |
| min_epsilon | 0.01 |
| mn_step | (4, 7) |


### Results 

One Common Model & HyperParameters used for all 4 snvironments, trained model evaluated for 100 runs.

> Actor(s) Count=`8`. Action Mode Deterministic=`True`.
> Early Termination Enabled: Training will stop once the agent reaches the required amount of scores to be considered solved (By Eval at every 100 update).


| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole-v1 | 495.41 | 16.96 | 4.72 | 700 | 337 | 16800 | 2.7001 | 500 | 500 |
| MountainCar-v0 | -107.22 | 10.71 | 12.77 | 1500 | 196 | 36000 | 1.5611 | -110 | -110 |
| LunarLander-v3 | 131.53 | 122.32 | 8.29 | 1100 | 76 | 26400 | 0.5689 | 232 | 200 |
| Acrobot-v1 | -190.82 | 165.30 | 0.60 | 200 | 8 | 4800 | 3.5072 | -93 | -100 |



# Rainbow dqn duelingnet [2025-01-05 17:45:30 IST]

same as all the above (not single qnet pass) + modified CDQN architecture to dueling net.

### Trainer Args 

| Key | Value |
| --- | --- |
| updates | 10000 |
| minibatch_size | 42 |
| actor_steps | 3 |
| eta | 0.0003 |
| gamma | 0.99 |
| v_min | -100 |
| v_max | 100 |
| minimum_steps | 3000 |
| n_atoms | 51 |
| tau | 0.005 |
| epochs | 64 |
| buffer_capacity | 10000 |
| buffer_alpha | 0.8 |
| buffer_beta_start | 0.4 |
| buffer_beta_end | 1.0 |
| rand_explore_till | 50 |
| max_epsilon | 0.2 |
| min_epsilon | 0.01 |
| mn_step | (8, 17) |


### Results 

One Common Model & HyperParameters used for all 4 snvironments, trained model evaluated for 100 runs.

> Actor(s) Count=`8`. Action Mode Deterministic=`True`.
> Early Termination Enabled: Training will stop once the agent reaches the required amount of scores to be considered solved (By Eval at every 100 update).


| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole-v1 | 500.00 | 0.00 | 2.59 | 400 | 73 | 9600 | 3.4280 | 500 | 500 |
| MountainCar-v0 | -110.26 | 13.66 | 6.17 | 700 | 82 | 16800 | 1.9038 | -108 | -110 |
| Acrobot-v1 | -105.74 | 25.44 | 0.86 | 200 | 24 | 4800 | 1.2701 | -94 | -100 |



# Rainbow dqn duelingnet & noisy layer [2025-01-05 18:52:16 IST]

added noisy layer in the model

### Trainer Args 

| Key | Value |
| --- | --- |
| updates | 10000 |
| minibatch_size | 42 |
| actor_steps | 3 |
| eta | 0.0003 |
| gamma | 0.99 |
| v_min | -100 |
| v_max | 100 |
| minimum_steps | 3000 |
| n_atoms | 51 |
| tau | 0.005 |
| epochs | 64 |
| buffer_capacity | 10000 |
| buffer_alpha | 0.8 |
| buffer_beta_start | 0.4 |
| buffer_beta_end | 1.0 |
| rand_explore_till | 50 |
| max_epsilon | 0.2 |
| min_epsilon | 0.01 |
| mn_step | (3, 5) |


### Results 

One Common Model & HyperParameters used for all 4 snvironments, trained model evaluated for 100 runs.

> Actor(s) Count=`8`. Action Mode Deterministic=`True`.
> Early Termination Enabled: Training will stop once the agent reaches the required amount of scores to be considered solved (By Eval at every 100 update).


| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole-v1 | 496.52 | 14.22 | 7.25 | 900 | 385 | 21600 | 1.0005 | 500 | 500 |
| MountainCar-v0 | -200.00 | 0.00 | 0.99 | 219 | 24 | 5256 | 3.4833 | -200 | -110 |
| LunarLander-v3 | 96.99 | 219.01 | 16.32 | 1600 | 139 | 38400 | 0.1630 | 218 | 200 |
| Acrobot-v1 | -113.95 | 60.44 | 8.70 | 900 | 132 | 21600 | 1.0482 | -88 | -100 |



# Final Rainbow dqn (no noisylayers) [2025-01-10 01:38:55 IST]

testing final rdqn in LunarLander-v3 env with no early Termination.

### Trainer Args 

| Key | Value |
| --- | --- |
| updates | 10000 |
| minibatch_size | 42 |
| actor_steps | 3 |
| eta | 0.0003 |
| gamma | 0.99 |
| v_min | -100 |
| v_max | 100 |
| minimum_steps | 3000 |
| n_atoms | 51 |
| tau | 0.005 |
| epochs | 64 |
| buffer_capacity | 10000 |
| buffer_alpha | 0.8 |
| buffer_beta_start | 0.4 |
| buffer_beta_end | 1.0 |
| rand_explore_till | 50 |
| max_epsilon | 0.2 |
| min_epsilon | 0.01 |
| mn_step | (8, 17) |


### Results 

One Common Model & HyperParameters used for all 1 snvironments, trained model evaluated for 100 runs.

> Actor(s) Count=`8`. Action Mode Deterministic=`True`.
> Early Termination Disabled: Training Stopped manually.
> Env Steps Truncated after None steps.


| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LunarLander-v3 | 157.12 | 164.09 | 25.16 | 2514 | 177 | 60336 | 0.2280 | 262 | 200 |

> Stopped training after noticing 260+ avg reward at past 3 log step evals.

# Rdqn test x [2025-01-23 22:51:45 IST]

Final Test X, some code refactor...

### Trainer Args 

| Key | Value |
| --- | --- |
| updates | 10000 |
| minibatch_size | 1024 |
| actor_steps | 50 |
| eta | 0.001 |
| gamma | 0.99 |
| v_min | -100 |
| v_max | 100 |
| minimum_steps | 3000 |
| n_atoms | 51 |
| tau | 0.005 |
| epochs | 16 |
| buffer_capacity | 10000 |
| buffer_alpha | 0.8 |
| buffer_beta_start | 0.4 |
| buffer_beta_end | 1.0 |
| rand_explore_till | 200 |
| max_epsilon | 0.2 |
| min_epsilon | 0.01 |
| mn_step | (6, 9) |


### Results 

One Common Model & HyperParameters used for all 4 snvironments, trained model evaluated for 100 runs.

> Actor(s) Count=`8`. Action Mode Deterministic=`True`.
> Early Termination Enabled: Training will stop once the agent reaches the required amount of scores to be considered solved (By Eval at every 50 update).



| Environment | Mean (rewards) | Std (rewards) | TrainTime (min) | Updates | No. Episodes | Total Steps | Final Loss | Final Eval Reward | Solved? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CartPole-v1 | 500.00 | 0.00 | 2.40 | 150 | 1184 | 60000 | 4.4727 | 500 | 500 |
| MountainCar-v0 | -104.86 | 8.36 | 12.88 | 700 | 1478 | 280000 | 3.0948 | -105 | -110 |
| LunarLander-v3 | 154.51 | 95.64 | 7.47 | 400 | 559 | 160000 | 3.0562 | 225 | 200 |
| Acrobot-v1 | -84.02 | 36.01 | 9.45 | 500 | 630 | 200000 | 1.1048 | -74 | -100 |


