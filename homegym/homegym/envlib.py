from collections import defaultdict
import concurrent.futures
import os
from queue import PriorityQueue
from typing import (
    DefaultDict,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
)
import warnings

import numpy as np
from numpy.typing import NDArray
from homegym._envtypings import MoleAction, PairInt, PathInfo
from homegym._rman import RNDManager, global_rand_manager
from homegym.board import Board
import homegym.constants as C
from homegym.envobjs import (
    BoardObject,
    FacingDirection,
    Mole,
    MovingDirection,
    Plant,
    Shot,
    Trail,
)


class MazeGenerator:
    """
    Well Connected Maze Generator from random noise.

    This class generates a maze by starting with random noise and then
    ensuring connectivity between all accessible cells.
    """

    def __init__(
        self,
        height: int,
        width: int,
        rman: RNDManager = global_rand_manager,
        wall_prob_thres: float = 0.5,
        max_allowed_ways_to_a_tile: int = 3,
    ):
        self.__height = height
        self.__width = width

        self.__rman = rman

        assert (
            0 <= wall_prob_thres < 1
        ), "Wall Selecting threshold is out of bound"
        self.__gamma = wall_prob_thres

        assert (
            2 <= max_allowed_ways_to_a_tile <= 8
        ), "Max Ways Limit is out of bound"

        self.__alpha = max_allowed_ways_to_a_tile

        self.__area = width * height

        if self.__area > 1600:
            warnings.warn(
                'The area is greater than 1600 (40x40), when generating a maze this might result in "RecursionError: maximum recursion depth exceeded" error.\nManually increase the limit using `sys.setrecursionlimit(N)` if already set ignore this message'
            )

        self.__current_round: int = 1
        self.__maze: NDArray[np.int16] = np.zeros(self.__area, dtype=np.int16)

    @property
    def get_steps_took(self) -> int:
        """Get the number of steps taken to generate the maze."""

        if self.__current_round == 1:
            raise Exception("Generator havn't used yet")

        return abs(self.__current_round)

    def __get_all_children(self, tile_id: int) -> Generator[int, None, None]:
        """Get all visitable children cells for a given tile."""

        x, y = divmod(tile_id, self.__width)

        for i, j in C.BASE_DIRECTIONS[0]:
            a, b = (x + i) % self.__height, (y + j) % self.__width
            k = a * self.__width + b
            if self.__maze[k] == 1:
                continue
            yield k

        for i, j in C.BASE_DIRECTIONS[1]:
            a, b = (x + i) % self.__height, (y + j) % self.__width
            k = a * self.__width + b

            # to check for diagonal blockage
            k1 = x * self.__width + b
            k2 = a * self.__width + y

            if self.__maze[k] == 1 or (
                self.__maze[k1] == 1 and self.__maze[k2] == 1
            ):
                continue

            yield k

    def __break_a_wall(self, tile_id: int) -> Optional[int]:
        """Attempt to break a wall adjacent to the given tile."""
        bt = None
        x, y = divmod(tile_id, self.__width)
        for i in range(-1, 2):
            for j in range(-1, 2):
                # if 0 == i == j: the current tile wil be skiped cause it
                # doesn't contain wall
                a, b = (x + i) % self.__height, (y + j) % self.__width
                k = a * self.__width + b
                if self.__maze[k] == 1:
                    bt = k
                    if self.__rman.random.random() < self.__gamma:
                        return bt

        return bt

    def __visit_all_tiles(self, tile_id: int) -> bool:
        """
        Perform a depth-first search to visit all accessible tiles,
        breaking walls as necessary to ensure connectivity.
        """

        self.__maze[tile_id] = self.__current_round

        # to check if we visited any tile from the current tile
        vc = 0  # visit count
        sc = 0  # already visited tile seen count

        for x in self.__get_all_children(tile_id):
            # If the child tile has visited in some other round
            if self.__maze[x] != 0:
                # check if it is visited by a previous round
                # the round will always be decreasing, if it is visted by a
                # previous round we just encountered a well connected network
                # se we can stop the search for this round
                if self.__maze[x] > self.__current_round:
                    return True

                # else it is round_ == maze[x], so we skipping this tile
                # increase the already seen count
                sc += 1
                continue

            vc += 1
            if self.__visit_all_tiles(x):
                return True

        # if we can't visit any child from this tile, the children cell should
        # contain visited marks or walls.
        # we need atleast 1 way to a tile and atmost 3, so we can break a tile
        # this variable is free we can pick any from 2 to 7
        if vc == 0 and sc < self.__alpha:
            bt = self.__break_a_wall(tile_id)
            self.__maze[bt] = 0
            # Try to visit all tiles again
            return self.__visit_all_tiles(tile_id)

        # in this aggressive search round more walls will be breaked
        # so at max a tile could have 8 ways to enter
        # at first round we don't have defined network to reach
        if self.__current_round < -1:
            # aggressively search for previously seen tile with no heuristics
            # we can also employ a heuristics to pick best wall to break.
            # here we using random breaking
            bt = self.__break_a_wall(tile_id)
            while bt is not None:
                self.__maze[bt] = 0
                if self.__visit_all_tiles(tile_id):
                    return True
                bt = self.__break_a_wall(tile_id)

        return False

    def __get_a_free_cell(self) -> int:
        """Find the first free (unvisited) cell in the maze."""
        # can add more randomness to this
        # fc = -1
        for i in range(self.__area):
            if self.__maze[i] == 0:
                # fc = -1
                # if self.rman.random() < .5:
                return i
        return -1

    def __flood_fill(self, tile_id: int):
        self.__maze[tile_id] = -1

        for x in self.__get_all_children(tile_id):
            if self.__maze[x] != 0:
                continue
            self.__flood_fill(x)

    # def plot_maze(self):
    #     plt.imshow(
    #         self.__maze.reshape((self.__height, self.__width)), cmap="gray_r"
    #     )
    #     plt.axis("off")
    #     plt.show()

    def assert_connectivity(self):
        self.__flood_fill(self.__get_a_free_cell())
        assert (
            self.__get_a_free_cell() == -1
        ), "The Maze is not Fully Connected"

        self.__maze[self.__maze < 0] = 0

    def generate_noise_maze(self, threshold: float = 0.3):
        """Generate a well-connected maze from random noise."""
        # generate random noise
        self.__maze[:] = (
            self.__rman.rng.random(self.__area) < threshold
        ).astype(np.int16)
        # the rounds will be decreasing for each round
        # the initial 0th round is the noise gen process
        self.__current_round = 0

        while True:
            curr_free_cell = self.__get_a_free_cell()
            # if there is no free cell, everything is either fully connected
            # or everything is just wall, if threshold set to more than .9
            # and a less area and the seed result in the second scenario
            if curr_free_cell == -1:
                break
            # decreament the count for next round of search
            self.__current_round -= 1
            self.__visit_all_tiles(curr_free_cell)

        # reseting all visited cells
        self.__maze[self.__maze < 0] = 0

        # # check if it is connected or not
        # # self.assert_connectivity()
        return self.__maze.astype(bool)


def calculate_cell_weight(
    pref: float,
    step_dist: float,
    slin_dist: float,
    send_dist: float,
) -> float:
    # *_dist \in (0, 1] small value == very close
    # pref \in [0 to 1] cell preference

    # including steps covered will give chance to unexplored paths
    # and make the movement a little random when the agent is not close
    dweight = np.array([0.0, 0.3, 0.7])
    distances = np.array([step_dist, slin_dist, send_dist])

    distance = dweight @ distances

    # Exp decay to map the distance to a probability
    # distribution: prob = exp(-D) + \eps
    # here exp is not e but 10
    prob = 100 ** (-distance)

    return float(100 * (prob * pref)) + 1e-8


class HiveMind:
    """
    Multi-Threaded A* Star Search and Best Cell Picking Decision Making System.
    """

    def __init__(self, board: Board, moles: List[Mole]):
        self._max_depth: int = C.MOLE_MAX_SEARCH_DEPTH

        self.__executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, os.cpu_count() + 4)  # pyright: ignore
        )

        self.__board = board

        self.__max_dist = max(board._width, board._height)
        self.__moles = moles

    def perfrom_mole_actions(self, target: int, danger_cells: Set[int]):
        """Perform each alive mole actions"""

        mole_actions = self.__get_mole_actions(target, danger_cells)
        attacks: List[Tuple[int, int, float]] = []
        for mole, actions in mole_actions.items():
            for (children, direction), action_flag in actions:
                mole.face_direction = FacingDirection(direction)
                match action_flag:
                    case 1:  # Move
                        self.__board.move_object(
                            mole, children, add_trail=True
                        )
                    case -1:  # Attack
                        attacks.append((children, direction, mole.damage))

                        # add attack trail (only visible to the human)
                        self.__board.add_object(
                            Trail(
                                C.MOLE_ATTACK_TRAIL_UID,
                                direction,
                                id(self.__board),
                                visible=False,
                            ),
                            children,
                        )
                    case 0:  # Nothing
                        pass
                    case _:
                        raise Exception("Got Unexpected Action Flag")
        return attacks

    def __get_mole_actions(
        self, target: int, danger_cells: Set[int]
    ) -> Dict[Mole, List[Tuple[PairInt, int]]]:
        # 0: Target Cell
        # 1: Action (-1: Attack, 0: Nothing, 1: Move)
        mole_actions: Dict[Mole, List[Tuple[PairInt, int]]] = {}

        # Intermediate cell weight holder for checking cell availability
        cells_at_curr_step: DefaultDict[int, int] = defaultdict(int)

        futures_to_args = {
            self.__executor.submit(
                self._astar_search,
                start=mole.cell_id,
                dest=target,
                weight=mole.weight,
                path_size=mole.step_size,
            ): mole
            for mole in self.__moles
            if mole  # only alive moles
        }

        for future in concurrent.futures.as_completed(futures_to_args):
            mole = futures_to_args[future]
            possible_paths = future.result()

            chosen_action = self.__attack_agent(
                target_cell=target,
                possible_paths=possible_paths,
                current_cell=mole.cell_id,
            )

            if chosen_action is None:
                chosen_action = self.__move_mole(
                    mole_weight=mole.weight,
                    possible_paths=possible_paths,
                    cells_at_curr_step=cells_at_curr_step,
                    target_cell=target,
                    danger_cells=danger_cells if mole.dodge_bullets else None,
                )

            if chosen_action:
                mole_actions[mole] = chosen_action

        return mole_actions

    def __attack_agent(
        self,
        target_cell: int,
        possible_paths: List[PathInfo],
        current_cell: int,
    ) -> Optional[MoleAction]:
        # Check if the target is in neighbouring tile
        # If it is send attack signal to the target
        for childrens, _ in possible_paths:
            # mole should not move and attack at the same time
            if len(childrens) > 1 or childrens[0][0] != target_cell:
                continue

            direction = childrens[0][1]

            # Checking for Diagonal facing attack
            if direction % 2 == 1:
                left_face = FacingDirection.from_number(direction - 1)
                right_face = FacingDirection.from_number(direction + 1)
                left_cell = self.__board.get_next_cell(
                    current_cell, left_face, MovingDirection.FRONT
                )
                right_cell = self.__board.get_next_cell(
                    current_cell, right_face, MovingDirection.FRONT
                )

                # if the adjacent tiles are blocked, then the attack will
                # hit on the blocked tile instead of the agent
                if (
                    self.__board[left_cell].is_blocked
                    or self.__board[right_cell].is_blocked
                ):
                    return None  # go and check for other cells to move

            return [
                (
                    (childrens[0][0], direction),
                    -1,
                )
            ]  # target cell, attack signal

        return None

    def __move_mole(
        self,
        mole_weight: int,
        possible_paths: List[PathInfo],
        cells_at_curr_step: DefaultDict[int, int],
        target_cell: int,
        danger_cells: Optional[Set[int]] = None,
        # top_k: int = 5,
    ) -> MoleAction:
        """
        Returns sequence of steps
        """

        mole_action: MoleAction = []
        level: int = 0

        while True:
            tiles_with_prob: List[Tuple[PairInt, float]] = []
            for childrens, (steps, distance) in possible_paths:
                if len(childrens) <= level or (
                    level > 0
                    and childrens[level - 1][0] == mole_action[level - 1][0]
                ):
                    continue

                children, direction = childrens[level]

                future_child_weight = (
                    mole_weight + cells_at_curr_step[children]
                )

                # Can not move to that tile
                if (
                    not self.__board[children].can_hold(future_child_weight)
                    or children == target_cell
                ):
                    continue

                preference = self.__board[children].preference(
                    cells_at_curr_step[children]
                )

                if danger_cells is not None and children in danger_cells:
                    preference = preference * 0.2

                straight_distance = (
                    self.__board.distance_between(children, target_cell)[0]
                    / self.__max_dist
                )

                tiles_with_prob.append(
                    (
                        (children, direction),
                        calculate_cell_weight(
                            pref=preference,
                            step_dist=steps / self._max_depth,
                            slin_dist=straight_distance,
                            send_dist=distance / self.__max_dist,
                        ),
                    )
                )

            if len(tiles_with_prob) == 0:
                break

            # giving a chance to stay on the same tile if all other tiles
            # are suboptimal based on the weights (<50)
            if level > 0:
                tiles_with_prob.append((mole_action[-1][0], 50))

            # This works well in simpler mazes, as moles can quickly reach the
            # agent. In complex mazes, they may get stuck in loops, so adding
            # randomness helps them break free. The agent will explore the
            # maze anyway, so moles will eventually move closer. Complex mazes
            # rarely have more than 5 adjacent free tiles, so moles will likely
            # end up near the agent and start tracking it as it moves.

            # if top_k > 0:
            #     tiles_with_prob = sorted(
            #         tiles_with_prob, key=lambda x: x[1], reverse=True
            #     )[:top_k]

            tiles, weights = zip(*tiles_with_prob)

            chosen_tile = self.__board.rman.random.choices(
                tiles, weights=weights, k=1
            )[0]

            mole_action.append((chosen_tile, 1))  # move signal1
            cells_at_curr_step[chosen_tile[0]] += mole_weight

            level += 1

        return mole_action

    def _astar_search(
        self, start: int, dest: int, weight: int, path_size: int
    ) -> List[PathInfo]:
        """
        Perform a depth-restricted A* search.

        Find possible tiles that can be reached from the current cell.

        Returns possible tile infos
        List[PathInfo] = [
          (
            ChildCell,
            (steps, hitStatus, distance from final_tile to searchend),
            direction
          ),
        ...
        ]
        """

        Q: PriorityQueue[Tuple[float, Tuple[int, int, Tuple[int, ...]]]] = (
            PriorityQueue()
        )
        expanded_cells: Set[int] = {start}

        tile_info: Dict[Tuple[int, ...], PairInt] = dict()
        tile_dirs: Dict[int, int] = dict()

        Q.put((0.0, (0, start, ())))

        while Q.qsize():
            _, node = Q.get()
            steps, curr, ancestors = node

            tile_info[ancestors] = (steps, curr)

            if curr == dest or steps >= self._max_depth:
                break

            expanded_cells.add(curr)

            for children, dir in self.__board.generate_childrens(curr, weight):
                if children in expanded_cells:
                    continue

                if len(ancestors) < path_size:
                    passed_ancestors = (*ancestors, children)
                    tile_dirs[children] = dir
                    # just to register the child presence
                    tile_info[passed_ancestors] = (steps + 1, children)
                else:
                    passed_ancestors = ancestors

                distance, _ = self.__board.distance_between(children, dest)
                # For heuristics considering the steps taken, distance between
                # mole to agent -> heuristics = steps + 1 + distance + p()
                # here a random number is added as a tiebreaker
                heuristics = (
                    steps + 1.0 + distance + self.__board.rman.random.random()
                )

                Q.put(
                    (
                        heuristics,
                        (steps + 1, children, passed_ancestors),
                    )
                )

        possible_paths: List[PathInfo] = []
        for ancestors, (steps, final_tile) in tile_info.items():
            if len(ancestors) == 0:
                continue

            possible_paths.append(
                (
                    [
                        (ancestor, tile_dirs[ancestor])
                        for ancestor in ancestors
                    ],
                    (
                        steps,
                        self.__board.distance_between(final_tile, dest)[0],
                    ),
                )
            )

        return possible_paths


class ShotData(NamedTuple):
    step_size: int
    damage: float
    max_range: int
    face_direction: FacingDirection


class AirControl:
    def __init__(self, board: Board) -> None:
        self.__board = board
        self.__shots_in_air: Set[Shot] = set()

    def reset(self):
        self.__shots_in_air.clear()

    def register_shot(self, shot_data: Optional[ShotData], cell_id: int):
        if shot_data is None:
            return

        new_shot = Shot(
            shot_data.step_size,
            shot_data.damage,
            shot_data.max_range,
            shot_data.face_direction,
        )
        self.__board.add_object(new_shot, cell_id)
        self.__shots_in_air.add(new_shot)

    def __perform(self, shot: Shot, object_hit_list: List[BoardObject]):
        step_size = min(shot.step_size, shot.max_steps - shot.total_steps)
        curr_damage = shot.damage

        prev_cell: int = shot.cell_id

        for cell_id, obj in self.__board.get_all_objects(
            shot.cell_id, step_size, shot.face_direction
        ):
            rem_damage = obj.take_shot(curr_damage)

            if rem_damage is not None:
                obj.apply_shot(curr_damage - rem_damage)
                object_hit_list.append(obj)
                curr_damage = rem_damage

            if curr_damage == 0:
                break

            if prev_cell != cell_id:
                shot.total_steps += 1
                self.__board.move_object(shot, cell_id, add_trail=True)
                prev_cell = cell_id

        if shot.total_steps >= shot.max_steps:
            curr_damage = 0

        shot.take_step(curr_damage)

    def __forecast(self, shot: Shot, danger_cells: Set[int]):
        step_size = min(shot.step_size, shot.max_steps - shot.total_steps)
        for cell_id, obj in self.__board.get_all_objects(
            shot.cell_id, step_size, shot.face_direction
        ):
            if obj._view_blocker:  # pyright: ignore
                break

            danger_cells.add(cell_id)

    def move_shots(self):
        obj_hit_list: List[BoardObject] = []
        danger_cells: Set[int] = set()

        for shot in list(self.__shots_in_air):
            self.__perform(shot, obj_hit_list)
            if not shot:
                self.__board.remove_object(shot)
                self.__shots_in_air.remove(shot)
            else:
                # get  future hit cells
                self.__forecast(shot, danger_cells)

        return obj_hit_list, danger_cells


class ObjectSpawner:
    """Manages the spawning of objects on the game board."""

    def __init__(
        self,
        n: int,
        alpha: float,
        step_size: float,
        extreme_alpha: float,
        object_array: List[Mole] | List[Plant],
        rman: RNDManager = global_rand_manager,
    ):
        self.probs: NDArray[np.float64]
        self.extreme_probs = np.array(
            [1 / ((1 + i) ** extreme_alpha) for i in range(n)]
        )
        self.alpha = alpha
        self.n = n
        self.step_size = (1 - step_size, step_size)

        self.__spwn_objcts = object_array
        self.__rman = rman

    def reset_objects(self):
        """
        Reset the spawner's state and object probabilities.
        """
        self.probs = np.array(
            [1 / ((1 + i) ** self.alpha) for i in range(self.n)]
        )
        # the np choice function barks when the sum of probs is closer to 1
        # but not exactly 1

        self.probs /= sum(self.probs)

        for obj in self.__spwn_objcts:
            obj._truth_value = False  # pyright: ignore
            obj.cell_id = -1

    def spawn(self) -> Optional[BoardObject]:
        nxt_spawner: Optional[BoardObject] = None

        for obj in self.__spwn_objcts:
            if not obj and obj.cell_id == -1:
                nxt_spawner = obj

        if nxt_spawner is None:
            return None

        objtype = self.__rman.rng.choice(self.n, p=self.probs)

        if self.step_size[1] > 0:
            self.probs = (
                self.step_size[0] * self.probs
                + self.step_size[1] * self.extreme_probs
            )
            self.probs /= sum(self.probs)

        return nxt_spawner.spawn(objtype)
