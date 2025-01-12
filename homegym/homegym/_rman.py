import random
from typing import Optional

import numpy as np


class RNDManager:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.random = random.Random(seed)

    def set_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.random.seed(seed)


global_rand_manager = RNDManager()
