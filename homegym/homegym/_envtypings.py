from typing import Dict, NamedTuple, Tuple, List

from numpy import float32
from numpy.typing import NDArray


ObjectRepr = Tuple[int, float, float, int]
# id, type, health, face dir

PairInt = Tuple[int, int]

TripleInt = Tuple[int, int, int]

PathInfo = Tuple[
    List[PairInt], PairInt
]  # List(cell id,dir), (steps, distance)

MoleAction = List[Tuple[PairInt, int]]  # (child, face direction), signal

BaseVecList = List[PairInt]

ObjectTypes = List[Dict[str, int | float | bool]]

NumVal = float | int

TupleF4 = Tuple[float, float, float, float]

ColorRGB = TripleInt
ColorRGBA = Tuple[int, int, int, int]

Observations = Tuple[
    NDArray[float32],
    NDArray[float32],
    NDArray[float32],
    NDArray[float32],
    NDArray[float32],
]


class EnvParams(NamedTuple):
    wall_prop: float
    plant_prop: float
    mole_prop: float

    # spawn controll
    alpha: float
    step: float
    xalpha: float

    wall_distribution: TupleF4
