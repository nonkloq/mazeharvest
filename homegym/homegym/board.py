from typing import Generator, List, Optional, OrderedDict, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from pygame import Surface

from homegym._envtypings import ObjectRepr, PairInt
from homegym._rman import RNDManager, global_rand_manager
import homegym.constants as C
from homegym.envobjs import (
    BoardObject,
    FacingDirection,
    MovingDirection,
    NoObj,
    StackableObject,
    Trail,
)


class Node:
    def __init__(self, obj: BoardObject, prev: Optional["Node"]):
        self.obj = obj
        self.prev = prev


class ObjectList:
    """Objects Holder"""

    def __init__(self) -> None:
        self.__head: Node = Node(NoObj(), None)
        self.__size: int = 0

    def __getitem__(self, n: int) -> BoardObject:
        curr = self.__head

        for _ in range(self.__size - n):
            if curr.prev is None:
                raise IndexError(
                    f"List size is {self.__size} and index requested is {n}"
                )
            curr = curr.prev

        return curr.obj

    def remove(self, n: int, return_node: bool = False) -> Optional[Node]:
        curr = self.__head
        prev: Optional[Node] = None

        for _ in range(self.__size - n):
            if curr is None:
                raise IndexError("the list is broken")

            prev = curr
            prev.obj.index_position -= 1
            curr = curr.prev

        if curr is None:
            raise IndexError("The list is broken")

        if prev:
            prev.prev = curr.prev

        else:
            assert curr.prev is not None, "The list is broken"
            self.__head = curr.prev

        self.__size -= 1

        if return_node:
            return curr

    def add_obj(self, obj: BoardObject):
        self.__head = Node(obj, self.__head)
        self.__size += 1
        obj.index_position = self.__size

    def add_node(self, node: Node):
        node.prev = self.__head
        self.__head = node
        self.__size += 1
        node.obj.index_position = self.__size

    def __iter__(self) -> Generator[BoardObject, None, None]:
        curr = self.__head
        while curr.prev is not None:
            yield curr.obj
            curr = curr.prev

    def clear(self):
        curr = self.__head
        while curr.prev is not None:
            tmp = curr
            curr = curr.prev
            del tmp
        self.__head = curr
        self.__size = 0

    def __len__(self):
        return self.__size


class Cell:
    def __init__(self, k: int):
        self.__cell_id = k
        self.__objects = ObjectList()

        # private ray memory
        self.__ray_hit_count: int = 0
        self.__tmp_obj_list: Optional[List[BoardObject]] = None

    def reset_ray_mem(self):
        del self.__tmp_obj_list

        self.__ray_hit_count = 0
        self.__tmp_obj_list = None

    def reset_cell(self):
        self.__objects.clear()
        self.reset_ray_mem()

    @property
    def capacity(self):
        return C.CELL_CAPACITY - sum(
            obj.weight for obj in self.__objects if obj
        )

    @property
    def is_blocked(self) -> bool:
        return self.capacity <= 0

    @property
    def is_view_blocked(self) -> bool:
        return any(
            obj._view_blocker for obj in self.__objects  # pyright: ignore
        )

    def __add_stackable_object(self, st_obj: BoardObject) -> bool:
        if type(st_obj) is not StackableObject:
            return False

        for obj in self.__objects:
            if (type(obj) is StackableObject) and (
                obj.object_id == st_obj.object_id
            ):
                obj.increase_count(st_obj.count)
                return True
        return False

    def can_hold(self, weight: int) -> bool:
        return weight <= self.capacity

    def add(self, obj: BoardObject) -> bool:
        if not self.can_hold(obj.weight):
            return False

        if self.__add_stackable_object(obj):
            return True

        self.__objects.add_obj(obj)
        obj.cell_id = self.__cell_id
        return True

    def add_node(self, node: Node):
        if not self.can_hold(node.obj.weight):
            return False

        self.__objects.add_node(node)
        node.obj.cell_id = self.__cell_id
        return True

    def remove(self, obj: BoardObject):
        self.__objects.remove(obj.index_position)

    def remove_node(self, obj: BoardObject) -> Node:
        node = self.__objects.remove(obj.index_position, True)

        if node is None:
            raise Exception("something went wrong")

        return node

    def preference(self, add_on: int = 0) -> float:
        cap_ = (self.capacity - add_on) / C.CELL_CAPACITY
        obj_counts_ = sum(bool(obj) for obj in self.__objects) / 10
        pref_ = cap_ - obj_counts_

        return max(0.0, pref_)

    def iterate_objects(self) -> Generator[BoardObject, None, None]:
        """Generator that yields all objects in the cell."""
        for obj in self.__objects:
            yield obj

    def hit_ray(self, force_ret: bool) -> Optional[ObjectRepr]:
        if self.__tmp_obj_list is None:
            self.__tmp_obj_list = list(iter(self.__objects))

        if self.__ray_hit_count < len(self.__tmp_obj_list):
            obj = self.__tmp_obj_list[self.__ray_hit_count]
            self.__ray_hit_count += int(
                not obj._view_blocker  # pyright: ignore
            )

            if type(obj) is Trail and not obj:
                return C.EMPTY_CELL_REPR if force_ret else None
            return obj.object_repr
        return C.EMPTY_CELL_REPR if force_ret else None


IGNORE_TYPES = {StackableObject, Trail}


class Board:
    def __init__(
        self, height: int, width: int, rman: RNDManager = global_rand_manager
    ):
        self._width = width
        self._height = height

        self.rman = rman

        self.__board_array: List[Cell] = [
            Cell(k) for k in range(width * height)
        ]

        self.__fast_free_cell_lookup: NDArray[np.int8] = np.zeros(
            width * height, dtype=np.int8
        )

    def __getitem__(self, cell_id: int) -> Cell:
        return self.__board_array[cell_id]

    def get_empty_cell(self, igncell: Optional[int] = None) -> int:
        """Get a random empty cell index."""

        if igncell is not None:
            self.__fast_free_cell_lookup[igncell] += 1

        empty_cells = np.where(self.__fast_free_cell_lookup <= 0)[0]
        if igncell is not None:
            self.__fast_free_cell_lookup[igncell] -= 1

        return self.rman.random.choice(empty_cells)

    def reset(self):
        self.__fast_free_cell_lookup[:] = 0
        for cell in self.__board_array:
            cell.reset_cell()

    def remove_presence(self, obj: BoardObject):
        # The object is be dead but it will remain on the board
        if obj:
            raise Exception("Object is not destroyed")

        self.__fast_free_cell_lookup[obj.cell_id] -= 1

    def add_object(self, obj: BoardObject, target_cell: int):
        assert self.__board_array[target_cell].add(
            obj
        ), "Failed to add an object"

        if type(obj) not in IGNORE_TYPES:
            self.__fast_free_cell_lookup[target_cell] += 1

    def remove_object(self, obj: BoardObject):
        self.__board_array[obj.cell_id].remove(obj)
        if type(obj) not in IGNORE_TYPES:
            self.__fast_free_cell_lookup[obj.cell_id] -= 1

        obj.cell_id = -1

    def move_object(
        self, obj: BoardObject, target_cell: int, add_trail: bool = False
    ):
        if type(obj) is IGNORE_TYPES:
            raise Exception("Can not move stackable & trail objects")

        curr_cell = obj.cell_id

        node = self.__board_array[curr_cell].remove_node(obj)
        assert self.__board_array[target_cell].add_node(
            node
        ), "Failed to move an object"

        if add_trail and obj.face_direction is not None:
            self.__board_array[curr_cell].add(
                Trail(
                    obj.object_id,
                    obj.face_direction.value,
                    id(self),
                    visible=True,
                )
            )

        self.__fast_free_cell_lookup[curr_cell] -= 1
        self.__fast_free_cell_lookup[target_cell] += 1

    def get_next_cell(
        self,
        current_cell: int,
        face_direction: FacingDirection,
        move_direction: MovingDirection,
    ) -> int:
        """
        Get the next cell index based on the current cell and movement direction.

        Returns:
            int: The index of the next cell.
        """
        rotation, base = divmod(face_direction.value, 2)

        i, j = C.BASE_DIRECTIONS[base][(rotation + move_direction.value) % 4]
        x, y = divmod(current_cell, self._width)
        x, y = (x + i) % self._height, (y + j) % self._width
        return x * self._width + y

    def get_all_objects(
        self,
        current_cell: int,
        steps: int,
        face_direction: FacingDirection,
        move_direction: MovingDirection = MovingDirection.FRONT,
    ) -> Generator[Tuple[int, BoardObject], None, None]:
        """Get all Object in a straight path from the given cell"""

        left_face: Optional[FacingDirection] = None
        right_face: Optional[FacingDirection] = None

        if face_direction.value % 2:  # diagonal face
            # in this case we have to consider the adjacent cell objects too
            left_face = FacingDirection.from_number(face_direction.value - 1)
            right_face = FacingDirection.from_number(face_direction.value + 1)

        while steps > 0:
            if left_face is not None and right_face is not None:
                left_cell = self.get_next_cell(
                    current_cell, left_face, move_direction
                )
                right_cell = self.get_next_cell(
                    current_cell, right_face, move_direction
                )

                lblock = self.__board_array[left_cell].is_view_blocked
                rblock = self.__board_array[right_cell].is_view_blocked
                order = [(left_cell, int(lblock)), (right_cell, int(rblock))]

                # we need to first return the view blocking object
                # then the class that using this method will encounter
                # the blocker first and take a valid action
                order: List[Tuple[int, int]] = sorted(
                    order,
                    key=lambda x: x[1],
                    reverse=True,
                )

                # the view ray in the current cell so we returning it instead
                # of the adjacent cell
                for adjacent_cell, _ in order:
                    for obj in self.__board_array[
                        adjacent_cell
                    ].iterate_objects():
                        yield current_cell, obj

            current_cell = self.get_next_cell(
                current_cell, face_direction, move_direction
            )

            for obj in self.__board_array[current_cell].iterate_objects():
                yield current_cell, obj
            else:
                yield current_cell, NoObj()

            steps -= 1

    def generate_childrens(
        self, current_cell: int, weight: int
    ) -> Generator[PairInt, None, None]:
        """Generate accessible neighboring cells with directions for pathfinding."""

        x_cord, y_cord = divmod(current_cell, self._width)
        # Horizontal & Vertical nodes
        for _base_n, (i, j) in enumerate(C.BASE_DIRECTIONS[0]):
            a_cord, b_cord = (x_cord + i) % self._height, (
                y_cord + j
            ) % self._width
            k = a_cord * self._width + b_cord

            if self.__board_array[k].can_hold(weight):
                yield k, _base_n * 2

        # diagonal nodes
        for _base_n, (i, j) in enumerate(C.BASE_DIRECTIONS[1]):
            a_cord, b_cord = (x_cord + i) % self._height, (
                y_cord + j
            ) % self._width

            k = a_cord * self._width + b_cord
            kleft = x_cord * self._width + b_cord
            kright = a_cord * self._width + y_cord

            if self.__board_array[k].can_hold(weight) and (
                not (
                    self.__board_array[kleft].is_blocked
                    and self.__board_array[kright].is_blocked
                )
            ):
                yield k, _base_n * 2 + 1

    def distance_between(
        self,
        source: int,
        destination: int,
        face_direction: FacingDirection = FacingDirection.N,
        ret_direction: bool = False,
    ) -> Tuple[int, Optional[int]]:
        """Calculate the distance and relative direction between two cells."""

        A = divmod(source, self._width)
        B = divmod(destination, self._width)

        # to  map x1 > x2 to positive (top direction)
        atob_row = A[0] - B[0]
        # to map y2 > y1 to positive (right direction)
        atob_col = B[1] - A[1]
        x = abs(atob_row)
        y = abs(atob_col)

        wrapped_x = self._height - x
        wrapped_y = self._width - y

        d1 = min(x, wrapped_x)
        d2 = min(y, wrapped_y)

        # cause of the diagonal paths the min distance is
        # max of a to b row/col (normal or wraped) distane
        distance = max(d1, d2)

        # direction info not needed or
        # if 0 == hord == verd, a single direction doesn't exist
        if not ret_direction or distance == 0:
            return distance, None

        # -1 bottom, 1 top
        row_dir = 2 * (atob_row >= 0) - 1
        # -1 left, 1 right
        col_dir = 2 * (atob_col >= 0) - 1
        # row_dir = -1 if atob_row < 0 else 1
        # col_dir = -1 if atob_col < 0 else 1

        # if the wrapped directions is shorter than actual,
        # flip the current direction signs

        # if a and b in same row, dont divide horizontally
        hord = (x != 0) * row_dir * (2 * (x <= wrapped_x) - 1)
        # (-1 if x > wraped_x else 1)
        # if a and b in same column, dont divide vertically
        verd = (y != 0) * col_dir * (2 * (y <= wrapped_y) - 1)
        # (-1 if y > wraped_y else 1)

        # hord: horizontal divide switch
        # verd: vertical divide switch
        # 0: Nothing, 1: Top/Right, -1: Bottom/Left (switch values)
        # North Face (F to LF): 1 0, 1 1, 0 1, -1 1, -1 0, -1 -1, 0 -1, 1 -1
        # These direction will rotate in clockwise for next FacingDirection

        direction = (C.REL_DIRECTION[(hord, verd)] - face_direction.value) % 8

        return distance, direction

    def get_ray_perception(
        self,
        current_cell: int,
        face_direction: FacingDirection,
        vision_range: float,
        num_rays: int,
        base: int,
        height: int,
    ) -> Tuple[NDArray[np.float32], Set[int], Set[int]]:
        """Get ray perception from the current cell in a specified direction."""

        perceptions: OrderedDict[
            Tuple[int, ObjectRepr], List[Tuple[float, int]]
        ] = OrderedDict()

        x_cord_start, y_cord_start = divmod(current_cell, self._width)

        visible_cells: Set[int] = set()

        half_base = base // 2
        for norm_angle, rad in compute_ray_angles(
            face_direction.value, num_rays, vision_range
        ):
            max_dist = get_max_dist(
                rad, half_base, height, face_direction.value
            )
            ray_x = np.cos(rad)  # indicates left side direction  (x axis)
            ray_y = np.sin(rad)  # indicates right side direction  (y axis)

            # cast ray
            prev_cell: PairInt = (x_cord_start, y_cord_start)
            cell_cord: int = -1
            _left_cell: int = -1
            _right_cell: int = -1

            for curr_dist in range(1, max_dist + 1):
                i_cord = round(x_cord_start - ray_y * curr_dist) % self._height
                j_cord = round(y_cord_start + ray_x * curr_dist) % self._width

                # if its a diagonal, check for view blockers
                prev_x_cord, prev_y_cord = prev_cell
                if (
                    abs(i_cord - prev_x_cord) == abs(j_cord - prev_y_cord)
                    and (_left_cell := prev_x_cord * self._width + j_cord)
                    is not None
                    and (_right_cell := i_cord * self._width + prev_y_cord)
                    is not None
                    and self.__board_array[_left_cell].is_view_blocked
                    and self.__board_array[_right_cell].is_view_blocked
                ):
                    cell_cord = (
                        _left_cell if abs(ray_x) > abs(ray_y) else _right_cell
                    )
                else:
                    cell_cord = i_cord * self._width + j_cord

                prev_cell = (i_cord, j_cord)
                visible_cells.add(cell_cord)

                obj_repr = self.__board_array[cell_cord].hit_ray(
                    curr_dist == max_dist
                )

                # if there is no unseen objects we can continue to next cell
                if obj_repr is None:
                    continue

                un_key = (cell_cord, obj_repr)
                if un_key not in perceptions:
                    perceptions[un_key] = []

                perceptions[un_key].append((norm_angle, curr_dist))
                # if the object is trace or shot we can continue
                # or if the object health is 0 (destroyed) we can continue
                if obj_repr[0] < 0 or obj_repr == 0:
                    continue

                # now this ray can not go further
                break

        # build perceptions
        built_perceptions: List[NDArray[np.float32]] = []
        edge_cells: Set[int] = set()
        for key, values in perceptions.items():
            perc = np.zeros(
                (len(values), len(C.EMPTY_CELL_REPR) + 2), dtype=np.float32
            )
            cell_idx, obj_repr = key
            perc[:, :2] = values
            perc[:, 2:] = obj_repr

            # crunch all ray data for that cell to one
            perc = perc.mean(axis=0, keepdims=True)

            built_perceptions.append(perc)

            # reset the edge cell ray memory
            self.__board_array[cell_idx].reset_ray_mem()
            edge_cells.add(cell_idx)
        return np.vstack(built_perceptions), visible_cells, edge_cells

    def render(self, canva: Surface):
        offs = C.CELL_SIZE // 2
        for cell_idx, cell in enumerate(self.__board_array):
            x, y = divmod(cell_idx, self._width)
            x, y = (
                (x * C.CELL_SIZE) + offs,
                (y * C.CELL_SIZE) + offs,
            )
            obc = 0
            for obj in cell.iterate_objects():
                obc += obj.draw(canva, y, x, obc)


def get_max_dist(
    angle: float,
    base: int,  # required \floor{base / 2}
    height: int,
    face_direction: int,
) -> int:
    # moving the angle back to base angle pi to 0
    angle += face_direction * C.ROTATION_STEP_ANGLE_RAD  # np.pi/4 or 45 degree

    alpha = np.abs(np.cos(angle))
    beta = np.abs(np.sin(angle))
    return int((base * alpha) + (height * beta))


def compute_ray_angles(
    face_direction: int, num_rays: int, vision_range: float
):
    # agent front facing angle in unit circle
    orientation = C.BASE_ORIENTATION - face_direction * C.ROTATION_STEP_ANGLE

    middle = vision_range / 2
    angle_max = orientation + middle
    angle_min = orientation - middle
    angles = np.linspace(
        angle_max,
        angle_min,
        num_rays,
    )

    norm_angles = (angles - angle_min) / (angle_max - angle_min)
    # relative_angle = vision_range * norm_angles

    # scaling 1-0 to -1 to 1
    # the left most angle is -1
    # the right most angle  is 1
    norm_angles = -2 * norm_angles + 1

    return zip(norm_angles, np.radians(angles))
