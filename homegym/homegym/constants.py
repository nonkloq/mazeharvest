from typing import Dict, Tuple

import numpy as np

from homegym._envtypings import (
    BaseVecList,
    ColorRGB,
    EnvParams,
    ObjectRepr,
    ObjectTypes,
    PairInt,
    TupleF4,
    ColorRGBA,
)

CELL_CAPACITY: int = 100
BASE_DIRECTIONS: Tuple[BaseVecList, BaseVecList] = (
    # FRONT  RIGHT   BACK    LEFT
    [(-1, 0), (0, 1), (1, 0), (0, -1)],  # N
    [(-1, 1), (1, 1), (1, -1), (-1, -1)],  # NE
)
POS_INF = float("inf")


# Clockwise direction relative to North Face
REL_DIRECTION: Dict[PairInt, int] = {
    (1, 0): 0,  # Front
    (1, 1): 1,
    (0, 1): 2,
    (-1, 1): 3,
    (-1, 0): 4,
    (-1, -1): 5,
    (0, -1): 6,
    (1, -1): 7,  # FrontLeft
}

# Agent Parameters
MIN_VIEW_WIDTH: int = 3
MAX_VIEW_WIDTH: int = 11
MIN_VIEW_LENGTH: int = 3
MAX_VIEW_LENGTH: int = 20

MAX_AGENT_HEALTH: float = 100.0
NORMAL_VISION_ANGLE: float = 180.0
HUNTER_VISION_ANGLE: float = 45.0
HUNTER_VISION_DURATION: int = 15
VISION_SWITCH_COOLDOWN: int = 60

AGENT_MAX_AMMO_COUNT: int = 3

SHOT_STEP_SIZE: int = 3
MAX_FIST_RANGE: int = 1
FIST_POWER: float = 50.0
SHOT_POWER: float = 100.0


HUNTER_VIEW_LENGTH_INCREASE_FACTOR: float = 0.4

# Environment Parameters
MOLE_SPAWN_PROBABILITY: float = 0.01589
PLANT_SPAWN_PROBABILITY: float = 0.03589

# --- DO NOT MODIFY IT ---

# fron facing angle (NORTH), agent base orientation
BASE_ORIENTATION: float = 90.0

# +- this from the main orientation will change the facing direction
ROTATION_STEP_ANGLE: float = 45.0  # in degrees
ROTATION_STEP_ANGLE_RAD: float = np.pi / 4  # in radians

# --------------------------------------------

DEFAULT_NUM_RAYS: int = 21


MOLE_MAX_SEARCH_DEPTH: int = 7


# Object IDS
WALL_UID: int = 1
MOLE_UID: int = 2
AMMO_UID: int = 3
PLANT_UID: int = 4
SHOT_UID: int = 5

TRAIL_UID: int = -1
MOLE_ATTACK_TRAIL_UID: int = -2

NO_DIRECTION: int = -1
NO_HEALTH: float = -1.0

# object id, type, health, direction
EMPTY_CELL_REPR: ObjectRepr = (0, 0.0, -1.0, -1)

# Object Types
MOLE_TYPES: ObjectTypes = [
    {
        "step_size": 1,
        "damage": 0.5,
        "health": 70,
        "bullet_drop_prob": 0.3,
        "heal": 12,
        "dodge_bullets": False,
        "weight": 30,
    },
    {
        "step_size": 1,
        "damage": 1,
        "health": 100,
        "bullet_drop_prob": 0.6,
        "heal": 15,
        "dodge_bullets": True,
        "weight": 40,
    },
    {
        "step_size": 2,
        "damage": 1,
        "health": 120,
        "bullet_drop_prob": 0.75,
        "heal": 20,
        "dodge_bullets": False,
        "weight": 40,
    },
    {
        "step_size": 1,
        "damage": 1.5,
        "health": 150,
        "bullet_drop_prob": 1.0,
        "heal": 25,
        "dodge_bullets": True,
        "weight": 50,
    },
    {
        "step_size": 1,
        "damage": 2,
        "health": 200,
        "bullet_drop_prob": 1.5,
        "heal": 40,
        "dodge_bullets": True,
        "weight": 70,
    },
    {
        "step_size": 2,
        "damage": 3,
        "health": 300,
        "bullet_drop_prob": 2.0,
        "heal": 50,
        "dodge_bullets": True,
        "weight": 100,
    },
]

MAX_PLANT_ADDON: float = 5.0
MAX_WALL_DAMAGE: float = 5
# the env poison level stays the same for all size of environment,
# plant propotion config
PLANT_TYPES: ObjectTypes = [
    {"poison_add_on": 1, "poison_reduction": 2, "heal": 10.0},
    {"poison_add_on": 2, "poison_reduction": 7, "heal": 20.0},
    {"poison_add_on": 3, "poison_reduction": 8, "heal": 30.0},
    {"poison_add_on": 4, "poison_reduction": 10, "heal": 40.0},
    {"poison_add_on": 5, "poison_reduction": 15, "heal": 50.0},
]


WALL_TYPES: ObjectTypes = [
    {"health": FIST_POWER, "damage": 0},
    {"health": SHOT_POWER, "damage": 0},
    {"health": POS_INF, "damage": 0},
    {"health": POS_INF, "damage": MAX_WALL_DAMAGE},
]

# Game Modes & Distributions
# The Individual object type Distributions are determined by power law
# Good range for alpha for this environment is from 2 to -1.5
# 2: top few gets more probability
# -1.5: the last few get resonable probability
# 0: equal probability

# the xalpha is the extreme type distribution for the current level
# the step is step size (x << 1 and > 0), it gradually moves the initial
# distribution to the extreme distribution


# For a custom Mode make sure the wall+plan+mole <= .8
# during the maze generation process the some walls will be destroyed
# so the agent will have more than 20 tiles to move freely at worst case

# Weak Wall1, weak wall2, unbreakable wall, hot wall
# sum should be equal to one
standard_wall_dist: TupleF4 = (0.1, 0.2, 0.5, 0.2)
risky_wall_dist: TupleF4 = (0.1, 0.1, 0.55, 0.25)
# fortnite_ahh_dist: TupleF4 = (0.4, 0.4, 0.1, 0.1)
high_risk_dist: TupleF4 = (0.05, 0.15, 0.5, 0.3)
# dangerous_dist: TupleF4 = (0.05, 0.15, 0.3, 0.5)

ENV_MODES: Dict[str, EnvParams] = dict(
    easy=EnvParams(0.3, 0.1, 0.03, 1.7, 0.01, 0, standard_wall_dist),
    medium=EnvParams(0.4, 0.12, 0.04, 1.2, 0.01, 0, standard_wall_dist),
    hard=EnvParams(0.5, 0.14, 0.06, 0.7, 0.01, -0.2, risky_wall_dist),
    extreme=EnvParams(0.55, 0.15, 0.08, 0, 0.01, -0.7, risky_wall_dist),
    insane=EnvParams(0.55, 0.15, 0.1, -0.1, 0.01, -1.0, high_risk_dist),
)

# render parameters
CELL_SIZE: int = 30
STAT_BAR_HEIGHT: int = 60
CELL_CENTER: int = CELL_SIZE // 2
AGENT_SIZE: int = int(CELL_CENTER * 0.9)
AGENT_CENTER_BOUND_DENOM: int = 4


CANVAS_FILL_COLOR: ColorRGBA = (229, 233, 240, 155)
VISIBLE_CELL_COLOR: ColorRGBA = (76, 86, 106, 60)
EDGE_CELL_POINT_COLOR: ColorRGBA = (236, 239, 244, 10)

WALL_COLOR: ColorRGB = (59, 66, 82)

BULLET_COLOR: ColorRGB = (208, 135, 112)
SHOT_COLOR: ColorRGBA = (235, 203, 139, 255)
SHOT_TRAIL: ColorRGB = (206, 173, 107)
MOLE_COLOR: ColorRGB = (153, 106, 73)
MOLE_TRAIL: ColorRGB = (150, 122, 102)
MOLE_ATTACK_TRAIL: ColorRGB = (191, 97, 106)

AGENT_COLOR: ColorRGBA = (136, 192, 208, 220)
TEXT_COLOR: ColorRGB = (236, 239, 244)
STAT_BAR_BG: ColorRGB = (67, 76, 94)
LOADING_BAR_BG: ColorRGB = (76, 86, 106)

HEALTH_COLOR: ColorRGB = (163, 190, 140)
POISON_COLOR: ColorRGB = (191, 97, 106)
VISION_MODE_COLOR: ColorRGB = (136, 192, 208)
