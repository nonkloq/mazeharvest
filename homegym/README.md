# HomeGym - MazeHarvest 

MazeHarvest is a grid-based survival reinforcement learning environment where an agent must protect the environment from toxic plants while defending itself against hostile moles. The world is fully connected (toroidal space), meaning all edges wrap around to their opposite sides.

![Overview](./assets/overview.svg)

## Overview
In MazeHarvest, the agent's primary tasks are:

- **Harvesting plants** to reduce environmental toxicity.
- **Defending against moles** that actively hunt the agent.

The environment is randomly generated with the following guarantees and features:

- **Path connectivity**: There is always at least one path between any two free cells.
- **World wrapping**: The grid is fully connected (toroidal).
- **Limited visibility**: The agent cannot see the entire grid but perceives its surroundings through ray perception.
- **Dynamic threats**: Plants & Moles are spawned randomly and the difficulty will progressively increase.

The agent also receives heuristic information about threats, plants and environmental conditions.

---

## Environment Features

### Random Generation
- Each grid is randomly generated and blockers are broken to guaranty the connectivity between free cells. (check [MazeGenerator](./homegym/envlib.py))
- Randomized placement of plants, moles, and walls, and the type distribution can shift during an episode after each spawn.
- Configurable difficulty and object types (check [constants.py](./homegym/constants.py)), can control object proportions and the spawn rates.

### World Dynamics
- **World wrapping**: The grid wraps around like a globe.
- **Moles**: 
  - Use a depth-restricted A* algorithm with probabilistic logic to actively hunt the agent.
  - Drop ammo and heal the agent when killed.
- **Plants**:
  - Increase environmental toxicity at each time step.
  - Provide healing and reduce toxicity when harvested.
- **Walls**:
  - Breakable walls require up to 1 to 2 hits by fist or 1 shot to destroy.
  - Unbreakable walls can not be destroyed and will not deal damage.
  - Electric walls (red colored one), when agent hits an electric wall it will lose some health (-5hp).

---

## Agent Overview

### Capabilities
- **Current Cell Auto Actions**: Pick up ammo (will not pick if inventory is full) and harvest plants from the current cell.
- **Combat**: Attack with fists or shoot with ammo (fists can be used even with/without ammo).
- **Navigation**: Move in 4 directions and change facing direction (left or right).
- **Vision modes**: Switch between normal and hunter vision modes.

### Action Space 

**Total Actions:** 10. (4 Movements,2 Face Turns, 2 Attack, 1 Vision Flip & 1 for no-action).

| Key | Name | Action | 
| --- | ---- | ------ |
| 0 | None | No Action |
| 1 | Left | Turn Left | 
| 2 | Right | Turn Right | 
| 3 | Switch Vision | Switch vision to hunter if cool down is over, when vision is hunter switch back to normal | 
| 4 | Shoot | If has ammo shoot, other wise attack | 
| 5 | Attack | Attack objects in opposite tile |
| 6 | Front | Move Front | 
| 7 | Right | Move Right | 
| 8 | Back | Move Back | 
| 9 | Left | Move Left | 


### Observation Space
The agent's observations consist of:

1. **Ray Perception**:
   - Sends `num_rays` in the facing direction.
   - Returns a array of perceived objects:
     - `(normalized angle, normalized distance, object ID, object type, object health, relative facing direction)`.
   - The object type will lie in range between 0 to 1, 0 represent the base type and 1 represent the extreme type

2. **View Modes**:
   - **Normal Vision**:
     - Wide field of view (180 degrees).
     - Limited range.
   - **Hunter Vision**:
     - Narrow field of view.
     - Longer range.

3. **Heuristics**:
   - Global plant heuristics.
   - Local mole heuristics (1.5x hunter vision range).

4. **Damage Directions**:
   -  Sum of damages from all 8 directions relative to the agent.

5. **Agent State**:
   - Health, ammo, air toxicity level, vision mode, vision switch cooldown, and facing direction.

**Total Observation:** 30 fixed-length values + arbitrary-length ray perception (`N x 6`).


| Observation           | Shape | Description |
|--------------------------|-----------|-----------------|
| **Ray Perception**       | (N, 6)    | A list of object representations for each unique object that the emitted rays have passed through or hit. Rays can pass through non-vision-blocking objects. Each entry contains 6 values representing the perceived object attributes. |
| **Loot Heuristics**      | (8,)      | A directional heuristic (in 8 directions relative to the agent) representing the presence and quality of nearby plants. Computed as a weighted sum: <br>  **Loot Heuristic** = ∑ (10 × *plant_type* × (1 − *distance* / *max_dist*))<br> The higher the plant's toxicity and the closer it is, the greater its contribution to the sum. |
| **Mole Heuristics**      | (8,)      | Similar to Loot Heuristics but for nearby moles. Only moles within a radius of **1.5 × hunter vision radius** are considered. The same weighted formula is applied: <br> **Mole Heuristic** = ∑ (10 × *mole_type* × (1 − *distance* / *max_dist*)) |
| **Damage Directions**    | (8,)      | Total accumulated damage from each of the 8 directions surrounding the agent, relative to the agent's current facing direction. |
| **Health**               | (1,)      | The agent’s current health normalized by maximum health:<br> **Health** = agent_health / max_health |
| **Air Poison Level**     | (1,)      | The true environmental poison concentration, not the "pl" (poison level) display which represents the **damage per step**. The damage may decrease in large environments or those with more walls to maintain balance. |
| **Ammo Inventory**       | (1,)      | The ratio of current ammo to the maximum ammo capacity:<br> **Ammo Ratio** = ammos / MAX_AMMOS |
| **Vision Mode** | (1,) | 1 if it is in hunter vision else 0 |
| **Vision Switch Cooldown** | (1,)    | Indicates whether the agent can switch vision modes:<br> 1 = available to switch<br> 0 ≤ value < 1 = cooldown in progress |
| **Facing Direction**     | (1,)      | The agent’s current facing direction, encoded as an integer or direction-specific label (e.g., 0 = North, 1 = North-East, etc.). |

**Note:**
The observation with shape one will be combined to single observation as **player_state** with shape (6, ).

---
## Rewards

- **Alive Bonus**:  
  +0.05 for each step the agent stays alive.

- **Combat**:  
  - -0.05 for shooting.  
  - +0.2 for a shot hitting a mole.  
  - +(0.3 × mole_type) for killing a mole.

- **Resource Collection**:  
  - +0.01 for picking up ammo.  
  - +1.0 for any successful harvest.  
  - +(1.0 × plant_type) as a bonus based on the plant type.  
  - +(0.8 × (1 − time_taken / max_time)) as a time-based bonus for faster harvests.

- **Penalties**:  
  - −(0.6 × (1 − health / max_health)) as a health-based penalty.  
  - −(0.8 × (min(dps, 2) / 2)) poison level penalty, based on damage per step (capped at 2).  
  - −10 for reaching a terminal state (agent death).

- **Exploration Bonus**:  
  +0.1 for visiting a new or rarely visited tile.  
  At each step, `visited_cells[current_cell]` is incremented.  
  Visit counts decay throughout the episode by a factor of 0.99.  
  The bonus is only given if `visited_cells[current_cell] ≤ 1.3`.

---

## Mission: Survive.

Navigate, harvest and combat moles to maintain the air poison level. The primary objective is to not die.

---

## How to Use?

The interface is similar to [Gymnasium](https://gymnasium.farama.org/) environments but doesn't have all of its features.

### Setup
Using a venv is recommended.
```sh
cd homegym # Not homegym/homegym/ src directory
pip install -e .
``` 

### Quick Start
```python
from homegym import MazeHarvest
import pygame
import time

env = MazeHarvest(width=20, height=30, env_mode="hard", seed=420, num_rays=21, max_steps=1000)

obs = env.reset(seed=1)
pygame.init()

d_height, d_width, _ = env.render().shape
screen = pygame.display.set_mode((d_height, d_width))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = env.action_space.sample()

    observation, reward, done, trunc = env.step(action)

    rgb_array = env.render()
    surf = pygame.surfarray.make_surface(rgb_array)
    screen.blit(surf, (0, 0))
    pygame.display.flip()
    time.sleep(0.01)

    if done:
        break

pygame.quit()
```

<img src="./assets/sample.png" alt="Sample" width="500">

**There are two render models available:** when `agent_center=True`, the agent remains in a bounding box in center of the grid, and moving outside the box shifts the environment while the agent stays on the same position (similar to MOBA-style games); when set to False, the grid remains fixed, and the agent moves across the grid. The default is True.

### Observation Unwrapping
```python
perception, loot_heuristics, mole_heuristics, damage_directions, agent_state = observation
```