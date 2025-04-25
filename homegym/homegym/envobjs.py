from enum import Enum
from typing import List, Optional

import numpy as np
from pygame import Surface

import pygame
from homegym._envtypings import ObjectRepr
from homegym._rman import global_rand_manager
import homegym.constants as C


class FacingDirection(Enum):
    """
    Enum representing the facing direction of an object on the board.

    ODD Directions use diagonal bases
    Even Direction use adjacent bases

    Remainder of the values denotes the base rotation,
    base starts at FRONT and goes clockwise
    """

    N = 0
    NE = 1
    E = 2
    SE = 3
    S = 4
    SW = 5
    W = 6
    NW = 7

    @staticmethod
    def from_number(number: int) -> "FacingDirection":
        return FacingDirection(number % 8)


class MovingDirection(Enum):
    """
    Enum representing the moving direction of an object on the board.

    Ordered in clockwise way for correct base direction rotation
    """

    FRONT = 0
    RIGHT = 1
    BACK = 2
    LEFT = 3


class BoardObject:
    """Base Class For all the board Objects"""

    def __init__(self, weight: int, obj_id: int) -> None:
        self.weight: int = weight
        self.object_id: int = obj_id

        self.cell_id: int = -1
        self.index_position: int = -1

        self.face_direction: Optional[FacingDirection] = None
        self.health: float = C.NO_HEALTH

        # object properties
        self._view_blocker: bool = False
        self._truth_value: bool = True

    def __bool__(self):
        return self._truth_value

    def take_shot(self, damage: float) -> Optional[float]:
        """Returns the remaining damage of the shot"""
        if self.health <= 0:
            return None

        return abs(min(self.health - damage, 0.0))

    def apply_shot(self, damage: float):
        if self.health <= 0:
            return
        self.health -= damage
        if self.health <= 0:
            self._truth_value = False
            self._view_blocker = False
            self.health = 0

    @property
    def object_repr(self) -> ObjectRepr:
        raise NotImplementedError(
            "Object representation for this object is unavailable"
        )

    def draw(self, canva: Surface, x: int, y: int, obc: int) -> int:
        raise NotImplementedError(
            "draw function for this object is unavailable"
        )


class NoObj(BoardObject):
    """
    Absence of an object on the board.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NoObj, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            super().__init__(0, 0)
            self._initialized = True
            self._truth_value = False


class StackableObject(BoardObject):
    """
    Base Class for stackable Objets.
    """

    def __init__(self, weight: int, obj_id: int):
        super().__init__(weight, obj_id)

        self.count: int = 0

    def increase_count(self, n: int):
        self.count += n

    def decrease_count(self, n: int):
        self.count -= n

    def __bool__(self):
        return self.count > 0


class Ammo(StackableObject):
    def __init__(self, count: int):
        super().__init__(0, C.AMMO_UID)
        self.count = count

    @property
    def object_repr(self) -> ObjectRepr:
        return (
            self.object_id,
            min(10, self.count) / 10,
            C.NO_HEALTH,
            C.NO_DIRECTION,
        )

    def draw(self, canva: Surface, x: int, y: int, obc: int) -> int:
        bullet_width = C.CELL_SIZE // 4
        bullet_height = bullet_width * 2 // 3

        bullet_rect = pygame.Rect(
            x - bullet_width - 10,
            y - bullet_height,
            bullet_width,
            bullet_height,
        )
        pygame.draw.rect(canva, C.BULLET_COLOR, bullet_rect)
        return 0


def get_face_dir_end(face_dir: int, x: int, y: int, dist: int):
    face_angle = np.radians(90 - face_dir * C.ROTATION_STEP_ANGLE)

    x = x - int(dist * np.sin(face_angle))
    y = y + int(dist * np.cos(face_angle))

    return (y, x)


class TNode:
    def __init__(
        self, trail: Optional["Trail"], nxt: Optional["TNode"], bid: int
    ) -> None:
        self.trail = trail
        self.bid = bid
        self.nxt = nxt


class Trail(BoardObject):
    instances: TNode = TNode(None, None, 0)

    def __init__(
        self, obid: int, dir: int, boardid: int, visible: bool = False
    ) -> None:
        super().__init__(0, C.TRAIL_UID)
        self.representing_object = obid
        self.direction = dir
        self._truth_value = visible
        self.__class__.instances = TNode(
            self, self.__class__.instances, boardid
        )

    @classmethod
    def get_trails(cls, req_id: int):
        """
        Get all trails from requested board,
        it will remove those trails from instances list
        """
        out: List["Trail"] = []

        def rec_func(node: TNode):
            if node.nxt is None:
                return node

            if node.bid == req_id:
                assert node.trail is not None
                out.append(node.trail)
                node = rec_func(node.nxt)
            else:
                node.nxt = rec_func(node.nxt)

            return node

        cls.instances = rec_func(cls.instances)
        return out

    @classmethod
    def discard_all(cls, board_id: int):
        cls.get_trails(board_id)

    @classmethod
    def hard_reset_ilist(cls):
        curr = cls.instances
        while curr.nxt is not None:
            tmp = curr
            curr = curr.nxt
            del tmp

    @property
    def object_repr(self) -> ObjectRepr:
        return (
            self.object_id,
            self.representing_object,
            C.NO_HEALTH,
            self.direction,
        )

    def draw(self, canva: Surface, x: int, y: int, obc: int) -> int:
        dist = abs(x - C.CELL_SIZE // 2)

        dist = C.CELL_SIZE // 2
        end_x, end_y = get_face_dir_end(self.direction, y, x, dist)

        max_offset = C.CELL_SIZE // 2

        trail_color, trail_width = self.get_props(self.representing_object)

        for _ in range(4):
            _x_off = global_rand_manager.random.randint(-10, 10)
            _y_off = global_rand_manager.random.randint(
                -max_offset, max_offset
            )
            start_pos = (x + _x_off, y + _y_off)
            end_pos = (end_x + _x_off, end_y + _x_off)
            pygame.draw.line(
                canva,
                trail_color,
                start_pos,
                end_pos,
                trail_width,
            )

        return 0

    @staticmethod
    def get_props(typ: int):
        match typ:
            case C.MOLE_ATTACK_TRAIL_UID:
                return C.MOLE_ATTACK_TRAIL, 2
            case C.SHOT_UID:
                return C.SHOT_TRAIL, 5
            case C.MOLE_UID:
                return C.MOLE_TRAIL, 3
            case _:
                return (0, 0, 0), 1


class Mole(BoardObject):
    def __init__(self) -> None:
        super().__init__(0, C.MOLE_UID)

        self._truth_value = False

        # Overriding defaults
        self.face_direction: FacingDirection = (  # pyright: ignore
            FacingDirection.S
        )

        # Mole Properties
        self.step_size: int = 1
        self.damage: float = 5.0
        self.bullet_drop_prob: float = 0.3
        self.heal: float = 0.0
        self.dodge_bullets: bool = False
        self._max_health: float = 0

    def spawn(self, mole_type: int):
        self._truth_value = True

        for k, v in C.MOLE_TYPES[mole_type].items():
            setattr(self, k, v)

        # big moles will block the view
        self._view_blocker = self.weight >= 70
        self._max_health = self.health

        return self

    def spawn_ammos(self, random_val: float) -> Optional[Ammo]:
        n = self.mole_ammo_drop_count(self.bullet_drop_prob, random_val)
        if n == 0:
            return None

        return Ammo(n)

    @property
    def object_repr(self) -> ObjectRepr:
        assert self.health >= 0, "repr requested for dead object"

        return (
            self.object_id,
            self.weight / C.CELL_CAPACITY,
            self.health / C.SHOT_POWER,
            self.face_direction.value,
        )

    @staticmethod
    def mole_ammo_drop_count(ammo_prob: float, random_val: float) -> int:
        """Calculate the number of ammo dropped by a mole."""
        n = int(ammo_prob)
        prob = ammo_prob - n
        return n + int(random_val < prob)

    def draw(self, canva: Surface, x: int, y: int, obc: int) -> int:
        max_c = np.ceil(C.CELL_SIZE / self.weight)

        csuf_center = C.CELL_SIZE // 2

        divider = C.CELL_SIZE // 4
        if max_c > 1:
            x += divider * (-1 + obc % 2 * 2)

        if max_c > 2:
            y += divider * (-1 * obc % 2 * 2)

        x += global_rand_manager.random.randint(-10, 10)
        y += global_rand_manager.random.randint(-10, 10)

        radius = int(csuf_center * self.weight / C.CELL_CAPACITY)
        circle_surface = pygame.Surface(
            (C.CELL_SIZE, C.CELL_SIZE), pygame.SRCALPHA
        )
        circ_center = (csuf_center, csuf_center)

        end_x, end_y = get_face_dir_end(
            self.face_direction.value, y, x, radius
        )

        pygame.draw.circle(
            circle_surface, self.get_color(), circ_center, radius
        )

        pygame.draw.line(
            circle_surface,
            (207, 0, 0, 100),
            circ_center,
            (
                end_x - (x - csuf_center),
                end_y - (y - csuf_center),
            ),
            width=4,
        )

        canva.blit(circle_surface, (x - csuf_center, y - csuf_center))

        return 1

    def get_color(self):
        r, g, b = C.MOLE_COLOR
        alpha = max(self.health / self._max_health, 0.04)
        return r, g, b, int(alpha * 255)


class Plant(BoardObject):
    def __init__(self) -> None:
        super().__init__(0, C.PLANT_UID)

        self._truth_value = False

        # Plant Properties
        self.poison_add_on: float = 1.0
        self.poison_reduction: float = 0.1

        self.heal: float = 10.0
        self._spawn_ts = -1

    def spawn(self, plant_type: int):
        self._truth_value = True
        for k, v in C.PLANT_TYPES[plant_type].items():
            setattr(self, k, v)

        return self

    def harvest_plant(self):
        self._truth_value = False

    @property
    def object_repr(self) -> ObjectRepr:
        return (
            self.object_id,
            self.poison_add_on / C.MAX_PLANT_ADDON,
            C.NO_HEALTH,
            C.NO_DIRECTION,
        )

    def draw(self, canva: Surface, x: int, y: int, obc: int) -> int:
        plant_width = plant_height = C.CELL_SIZE // 3

        plant_rect = pygame.Rect(
            x - plant_width,
            y - plant_height,
            plant_width,
            plant_height,
        )
        pygame.draw.rect(canva, self.get_color(self.poison_add_on), plant_rect)
        return 0

    @staticmethod
    def get_color(value: float):
        normalized = (value - 1) / 4

        red = int(255 * normalized)
        green = int(255 * (1 - normalized))

        return (red, green, 0)


class Wall(BoardObject):
    def __init__(self, wall_type: int):
        super().__init__(100, C.WALL_UID)
        self._view_blocker = True
        self._truth_value = True

        self.health: float = C.POS_INF
        self.damage: float = 0

        for k, v in C.WALL_TYPES[wall_type].items():
            setattr(self, k, v)

    @property
    def object_repr(self) -> ObjectRepr:
        health = (
            C.NO_HEALTH
            if self.health == C.POS_INF
            else self.health / C.SHOT_POWER
        )

        return (
            self.object_id,
            self.damage / C.MAX_WALL_DAMAGE,
            health,
            C.NO_DIRECTION,
        )

    def draw(self, canvas: Surface, x: int, y: int, obc: int) -> int:
        wall_width = wall_height = C.CELL_SIZE

        base_dist = C.CELL_SIZE // 2
        plant_rect = pygame.Rect(
            x - base_dist,
            y - base_dist,
            wall_width,
            wall_height,
        )
        transparent_surface = Surface(plant_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(
            transparent_surface,
            self.get_color(self.health, self.damage),
            transparent_surface.get_rect(),
            border_radius=0,
        )

        canvas.blit(transparent_surface, plant_rect.topleft)
        return 0

    @staticmethod
    def get_color(value: float, damage: float):
        alpha = 1 if value == C.POS_INF else value / C.SHOT_POWER - 0.2
        alpha = max(alpha, 0.1)
        r, g, b = C.WALL_COLOR
        r = int((damage > 0) * 88) or r
        return (r, g, b, int(alpha * 255))


class Shot(BoardObject):
    def __init__(
        self,
        step_size: int,
        damage: float,
        max_range: int,
        face_direction: FacingDirection,
    ) -> None:
        super().__init__(0, C.SHOT_UID)

        self.step_size = step_size
        self.damage = damage
        self.max_steps = max_range

        # Overriding default
        self.face_direction: FacingDirection = (  # pyright: ignore
            face_direction
        )
        self.total_steps = 0

    def take_step(self, final_damage: float):
        self.damage = final_damage
        if self.damage <= 0:
            self._truth_value = False

    @property
    def object_repr(self) -> ObjectRepr:
        return (
            self.object_id,
            self.step_size / C.SHOT_STEP_SIZE,
            C.NO_HEALTH,
            self.face_direction.value,
        )

    def draw(self, canvas: Surface, x: int, y: int, obc: int) -> int:
        circle_surface = pygame.Surface(
            (C.CELL_SIZE, C.CELL_SIZE), pygame.SRCALPHA
        )
        shot_size = C.CELL_SIZE // 3
        csuf_center = C.CELL_SIZE // 2
        pygame.draw.circle(
            circle_surface, C.SHOT_COLOR, (csuf_center, csuf_center), shot_size
        )

        canvas.blit(circle_surface, (x - csuf_center, y - csuf_center))

        return 0
