from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from pygame.surface import Surface

import pygame
from homegym._envtypings import (
    ColorRGB,
    EnvParams,
    NumVal,
    Observations,
    TripleInt,
)
from homegym._rman import RNDManager, global_rand_manager
from homegym.board import Board
import homegym.constants as C
from homegym.envlib import (
    AirControl,
    HiveMind,
    MazeGenerator,
    ObjectSpawner,
    ShotData,
)
from homegym.envobjs import (
    Ammo,
    BoardObject,
    FacingDirection,
    Mole,
    MovingDirection,
    Plant,
    Trail,
    Wall,
)

pygame.font.init()


@dataclass
class StepData:
    total_heal: float = 0
    attack_data: List[Tuple[int, int, float]] = field(
        default_factory=list
    )  # cell, dir, damage
    env_air_poison_reduction: float = 0

    kill_count: int = 0
    harvest_count: int = 0
    ammo_pickup: int = 0
    shoot_count: int = 0

    shot_wall_hit: int = 0
    shot_mole_hit: int = 0

    kill_type: float = 0
    harvest_type: float = 0

    prev_cell: int = 0

    def reset(self):
        self.total_heal = 0
        self.attack_data = []
        self.env_air_poison_reduction = 0

        self.kill_count = 0
        self.harvest_count = 0
        self.ammo_pickup = 0
        self.shoot_count = 0

        self.kill_type = 0
        self.harvest_type = 0

        self.shot_wall_hit: int = 0
        self.shot_mole_hit: int = 0


class EnvBackBone:
    def __init__(
        self, height: int, width: int, env_params: EnvParams, rman: RNDManager
    ) -> None:
        self.height = height
        self.width = width

        self.rman = rman

        self.env_params = env_params

        self.maze_generator = MazeGenerator(height, width, rman)

        self.board = Board(height, width, rman)

        total_cells = height * width

        self.moles = [Mole() for _ in range(int(env_params.mole_prop * total_cells))]
        self.plants = [Plant() for _ in range(int(env_params.plant_prop * total_cells))]
        _alpha = env_params.alpha
        _step_size = env_params.step
        _xalpha = env_params.xalpha

        self.mole_spawner = ObjectSpawner(
            len(C.MOLE_TYPES), _alpha, _step_size, _xalpha, self.moles, rman
        )

        self.plant_spawner = ObjectSpawner(
            len(C.PLANT_TYPES), _alpha, _step_size, _xalpha, self.plants, rman
        )

        self.hive_mind = HiveMind(self.board, self.moles)

        self.air_control = AirControl(self.board)

        self.step_data = StepData()

        self._objects_to_clear: List[BoardObject] = []

        self._air_poison_scale: float = total_cells // 2

        self.env_air_poison_level: float = 0

        avs_le = (self.width * self.height) / 2

        self._env_poison_scalar: float = (avs_le / 10) / total_cells

        # additional stats
        self._kill_count = 0
        self._step_count = 0
        self._harvest_count = 0

    def reset(self):
        self.board.reset()

        self.mole_spawner.reset_objects()
        self.plant_spawner.reset_objects()
        self.air_control.reset()

        # clear all objects
        Trail.discard_all(id(self.board))

        self.step_data.reset()
        self._objects_to_clear.clear()
        self.env_air_poison_level = 0

        # generate walls
        wall_map = self.maze_generator.generate_noise_maze(self.env_params.wall_prop)

        wall_types = list(range(len(C.WALL_TYPES)))

        for cell_idx in range(len(wall_map)):
            if not wall_map[cell_idx]:
                continue

            walltype = self.rman.random.choices(
                wall_types, weights=self.env_params.wall_distribution
            )[0]
            self.board.add_object(Wall(walltype), cell_idx)

        self._kill_count = 0
        self._step_count = 0
        self._harvest_count = 0

    def clear_trails(self):
        for obj in Trail.get_trails(id(self.board)):
            try:
                self.board.remove_object(obj)
            except Exception as _:
                # print("at clearance: ")
                # print(
                #     obj,
                #     obj.cell_id,
                #     obj.representing_object,
                #     obj.index_position,
                #     obj.face_direction,
                # )
                # raise e
                pass

    def spawn_objects_randomly(self, agent_tile: int):
        # Spawn Mole
        if self.rman.random.random() < C.MOLE_SPAWN_PROBABILITY:
            mole = self.mole_spawner.spawn()
            if mole is not None:
                randidx = self.board.get_empty_cell(agent_tile)
                try:
                    self.board.add_object(mole, randidx)
                except Exception as e:
                    # print(mole.weight, mole.health)
                    # print(mole, mole.face_direction)
                    # print(randidx, agent_tile)
                    # objs = list(self.board[randidx].iterate_objects())
                    # for obj in objs:
                    #     print(obj)
                    #     print(obj.health)
                    #     print(obj.face_direction)
                    #     print(obj._truth_value)
                    #     print(obj._view_blocker)
                    raise e
        # Spawn Plant
        if self.rman.random.random() < C.PLANT_SPAWN_PROBABILITY:
            plant = self.plant_spawner.spawn()
            if plant is not None:
                randidx = self.board.get_empty_cell(agent_tile)
                self.board.add_object(plant, randidx)

    def get_next_tile_infos(
        self,
        currcell: int,
        facedir: FacingDirection,
        direction: MovingDirection,
        weight: int,
    ) -> Tuple[int, Optional[List[int]]]:
        # check diagonal blockage
        if facedir.value % 2 == 1:
            left_face = FacingDirection.from_number(facedir.value - 1)
            right_face = FacingDirection.from_number(facedir.value + 1)

            left_cell = self.board.get_next_cell(currcell, left_face, direction)
            right_cell = self.board.get_next_cell(currcell, right_face, direction)

            lblock = self.board[left_cell].is_blocked
            rblock = self.board[right_cell].is_blocked
            if lblock and rblock:
                return -1, [left_cell, right_cell]

        nxt_cell = self.board.get_next_cell(currcell, facedir, direction)

        if not self.board[nxt_cell].can_hold(weight):
            return -1, [nxt_cell]

        return nxt_cell, None

    def take_step(self, agent_position: int):
        """Move environment to next state"""
        # clear all dead objects
        for obj in self._objects_to_clear:
            # objects that will stay forever with 0 health
            if obj.object_id == C.WALL_UID:
                self.board.remove_presence(obj)
                continue

            if type(obj) is Mole:
                new_ammos = obj.spawn_ammos(self.rman.random.random())
                if new_ammos is not None:
                    self.board.add_object(new_ammos, obj.cell_id)

            self.board.remove_object(obj)

        del self._objects_to_clear

        # move shots
        hit_objects, danger_cells = self.air_control.move_shots()
        # handle damaged objects
        self._objects_to_clear = []
        for obj in hit_objects:
            if obj:  # not dead
                match obj.object_id:
                    case C.WALL_UID:
                        self.step_data.shot_wall_hit += 1
                    case C.MOLE_UID:
                        self.step_data.shot_mole_hit += 1

                continue

            self._objects_to_clear.append(obj)

            if type(obj) is Mole:
                self.step_data.total_heal += obj.heal
                self._kill_count += 1

                self.step_data.kill_count += 1
                self.step_data.kill_type += obj.object_repr[1]

        # move moles
        attacks = self.hive_mind.perfrom_mole_actions(agent_position, danger_cells)

        self.step_data.attack_data.extend(attacks)

        # update air poison level
        self.env_air_poison_level += (
            sum(plant.poison_add_on for plant in self.plants if plant)
            / self._air_poison_scale
        )

        self.env_air_poison_level = max(
            0,
            self.env_air_poison_level - self.step_data.env_air_poison_reduction,
        )

        self._step_count += 1


@dataclass
class VisionMode:
    angle: float
    length: int
    is_long: bool


class Agent:
    is_alive: bool = False
    face_direction: FacingDirection
    current_cell: int

    _health: float = C.MAX_AGENT_HEALTH
    _weight = C.CELL_CAPACITY
    _ammos_in_inv: int = 0

    def __init__(
        self,
        view_width: int,
        view_length: int,
        max_view_length: int,
        num_rays: int,
        env_backbone: EnvBackBone,
    ) -> None:
        self._view_width = view_width
        self._view_length = view_length
        self._max_view_length = max_view_length
        self._num_rays: int = num_rays

        self._vision_mode: VisionMode = VisionMode(
            C.NORMAL_VISION_ANGLE, self._view_length, False
        )
        # to perform actions in the environment
        self.__env_back_bone = env_backbone

        self.visible_cells: Set[int] = set()
        self.edge_cells: Set[int] = set()

    def reset(self):
        self.is_alive = True
        self._ammos_in_inv = 0
        self._health: float = C.MAX_AGENT_HEALTH

        self._vision_mode.angle = C.NORMAL_VISION_ANGLE
        self._vision_mode.length = self._view_length
        self._vision_mode.is_long = False

        self.visible_cells.clear()
        self.edge_cells.clear()

        self.current_cell = self.__env_back_bone.board.get_empty_cell()

        self.face_direction = self.__env_back_bone.rman.random.choice(
            list(FacingDirection)
        )

    def __pickup_ammo(self, n: int) -> int:
        """Returns the pickup count of the ammo"""
        final_ammoc = min(self._ammos_in_inv + n, C.AGENT_MAX_AMMO_COUNT)
        pickup_count = final_ammoc - self._ammos_in_inv
        self._ammos_in_inv = final_ammoc
        return pickup_count

    def update_health(self):
        damage_by_poison = (
            self.__env_back_bone.env_air_poison_level
            * self.__env_back_bone._env_poison_scalar
        )

        total_damage = sum(
            damage
            for celid, _, damage in self.__env_back_bone.step_data.attack_data
            if celid == self.current_cell
        )

        tot_damage = total_damage + damage_by_poison

        tot_damage -= self.__env_back_bone.step_data.total_heal

        self._health = min(self._health - tot_damage, C.MAX_AGENT_HEALTH)
        # self._health = max(-100, self._health)

        self.is_alive = self._health > 0

    def do_action(self, action: int):
        match action:
            case 0:  # Nothing
                return
            case 1:  # turn left
                self.face_direction = FacingDirection.from_number(
                    self.face_direction.value - 1
                )
            case 2:  # turn right
                self.face_direction = FacingDirection.from_number(
                    self.face_direction.value + 1
                )
            case 3 if not self._vision_mode.is_long:  # flip to hunter vision
                self._vision_mode.angle = C.HUNTER_VISION_ANGLE
                self._vision_mode.length = self._max_view_length
                self._vision_mode.is_long = True
            case 3 if self._vision_mode.is_long:  # flip to normal vision
                self._vision_mode.angle = C.NORMAL_VISION_ANGLE
                self._vision_mode.length = self._view_length
                self._vision_mode.is_long = False

            case 4 if self._ammos_in_inv > 0:  # fire a shot
                self.__env_back_bone.air_control.register_shot(
                    ShotData(
                        C.SHOT_STEP_SIZE,
                        C.SHOT_POWER,
                        self._vision_mode.length,
                        self.face_direction,
                    ),
                    self.current_cell,
                )
                self.__env_back_bone.step_data.shoot_count += 1
                self._ammos_in_inv -= 1
            case 5 | 4:  # attack (including when no ammo)
                self.__env_back_bone.air_control.register_shot(
                    ShotData(
                        C.MAX_FIST_RANGE,
                        C.FIST_POWER,
                        C.MAX_FIST_RANGE,
                        self.face_direction,
                    ),
                    self.current_cell,
                )
            case action if 6 <= action < 10:
                self._move(MovingDirection(action - 6))

            case _:
                raise Exception("Unknown agent action flag")

    def _move(self, direction: MovingDirection):
        """Try to move to the next cell"""

        nxt_cell, blocking_tiles = self.__env_back_bone.get_next_tile_infos(
            self.current_cell, self.face_direction, direction, self._weight
        )

        if blocking_tiles is not None:
            # check for damage causing wall
            for cell_idx in blocking_tiles:
                for obj in self.__env_back_bone.board[cell_idx].iterate_objects():
                    if not obj or type(obj) is not Wall:
                        continue

                    self.__env_back_bone.step_data.attack_data.append(
                        (
                            self.current_cell,
                            # changing the direction relative to fd.North
                            # and flipping it so that the agent will percive
                            # it as damage coming from the direction relative
                            # to it (f:N, m:Left, rd:6 ,frd:2) the agent will
                            # flip the direction again and it will get the rd
                            (direction.value * 2 + self.face_direction.value + 4) % 8,
                            obj.damage,
                        )
                    )
            return

        self.current_cell = nxt_cell
        # collect all items
        for obj in self.__env_back_bone.board[self.current_cell].iterate_objects():
            if not obj:
                continue

            match obj:
                case Plant():
                    self.__env_back_bone.step_data.env_air_poison_reduction += (
                        obj.poison_reduction
                    )

                    self.__env_back_bone.step_data.total_heal += obj.heal
                    obj.harvest_plant()
                    self.__env_back_bone._harvest_count += 1

                    self.__env_back_bone.step_data.harvest_count += 1
                    self.__env_back_bone.step_data.harvest_type += obj.object_repr[1]

                    self.__env_back_bone.board.remove_object(obj)

                case Ammo():
                    pickup_count = self.__pickup_ammo(obj.count)
                    self.__env_back_bone.step_data.ammo_pickup += pickup_count
                    obj.decrease_count(pickup_count)
                    if not obj:
                        self.__env_back_bone.board.remove_object(obj)

                case _:
                    continue

    def _reward_function(self, terminal: bool) -> float:
        if terminal:
            return -(100.0 + self.__env_back_bone.env_air_poison_level * 0.1)

        step_reward: float = 1.0

        step_data = self.__env_back_bone.step_data

        # Here all the constants are  rewards

        # losses
        negative_rewards = np.zeros(1, dtype=np.float32)

        # health_factor
        # moving the last denom closer to 1 make the punishment more aggressive
        # negative_rewards[0] = (1 - self._health / C.MAX_AGENT_HEALTH) / 3.14

        # ammo loss
        negative_rewards[0] = step_data.shoot_count * 0.2

        # same cell penalty
        # negative_rewards[1] = (self.current_cell == step_data.prev_cell) * 1.5

        # poison level concern
        # negative_rewards[1] = self.__env_back_bone.env_air_poison_level * 0.1

        # wall hit
        # negative_rewards[4] = step_data.shot_wall_hit * 0.2

        # Gain
        positive_rewards = np.zeros(4, dtype=np.float32)

        # kill_dopomine
        positive_rewards[0] = 11.0 * step_data.kill_type

        # harvest dopomine
        positive_rewards[1] = 47.0 * step_data.harvest_type

        # Mole Hit Satisfaction
        positive_rewards[2] = step_data.shot_mole_hit * 0.7

        # Ammo pickup Satisfaction
        positive_rewards[3] = step_data.ammo_pickup * 0.5

        # # to discount the rewards when env have high poison level
        # # by doing this the agent will get less rewards for GOOD actions
        # poison_lvl_discount: float = max(
        #     (1 - self.__env_back_bone.env_air_poison_level / 100), 9e-3
        # )
        gains = positive_rewards.sum()

        losses = negative_rewards.sum()

        return float(step_reward - losses + gains)

    def compute_state(self) -> Tuple[float, Observations, bool]:
        perception, visible_cells, edge_cells = (
            self.__env_back_bone.board.get_ray_perception(
                self.current_cell,
                self.face_direction,
                self._vision_mode.angle,
                self._num_rays,
                self._view_width,
                self._vision_mode.length,
            )
        )

        self.visible_cells.clear()
        self.visible_cells.update(visible_cells)
        self.edge_cells.clear()
        self.edge_cells.update(edge_cells)

        # scaling the distance
        perception[:, 1] /= self._max_view_length

        # changing the directions relative to the agent
        # the directions will be shifted from base dir (North faced)
        # to agent facing direction
        mask = perception[:, 5] > -1
        perception[mask, 5] = (perception[mask, 5] - self.face_direction.value) % 8

        loot_heuristics = self._compute_heuristics(self.__env_back_bone.plants)
        mole_heuristics = self._compute_heuristics(
            self.__env_back_bone.moles, max_range=self._max_view_length * 1.5
        )

        damage_dirs = np.zeros(8, dtype=np.float32)
        _face_dir = self.face_direction.value
        for celid, _dir, damage in self.__env_back_bone.step_data.attack_data:
            if celid == self.current_cell:
                # (from_dir- face_fir + 4) % 8 will get the relative
                # attack's from direction (here 4 flipping the dir)
                damage_dirs[(_dir - _face_dir + 4) % 8] += damage

        player_state = np.array(
            [
                self._health / C.MAX_AGENT_HEALTH,
                self.__env_back_bone.env_air_poison_level,
                self._ammos_in_inv / C.AGENT_MAX_AMMO_COUNT,
                int(self._vision_mode.is_long),
                self.face_direction.value,
            ],
            dtype=np.float32,
        )

        terminal: bool = not self.is_alive

        return (
            self._reward_function(terminal),
            (
                perception,
                self._normalize_vector(loot_heuristics),
                self._normalize_vector(mole_heuristics),
                self._normalize_vector(damage_dirs),
                player_state,
            ),
            terminal,
        )

    def _compute_heuristics(
        self,
        objects: List[Mole] | List[Plant],
        max_range: Optional[float] = None,
    ) -> NDArray[np.float32]:
        heuristics = np.zeros(8, dtype=np.float32)
        max_dist = (
            max(self.__env_back_bone.height, self.__env_back_bone.width)
            if max_range is None
            else max_range
        )

        for obj in objects:
            if not obj:
                continue

            dist, _dir = self.__env_back_bone.board.distance_between(
                self.current_cell,
                obj.cell_id,
                self.face_direction,
                ret_direction=True,
            )

            if max_range is None or dist < max_range:
                _, _type, _, _ = obj.object_repr
                # giving high preference to closer objects
                weight = (10 * _type) * (1 - (dist / max_dist))
                heuristics[_dir] += weight
        return heuristics

    @staticmethod
    def _normalize_vector(vector: NDArray[np.float32]) -> NDArray[np.float32]:
        sum_vector = np.sum(vector)
        return vector / sum_vector if sum_vector > 0 else vector

    def draw(self, canvas: Surface):
        board_width = self.__env_back_bone.width

        gx, gy = divmod(self.current_cell, board_width)
        px, py = (gx * C.CELL_SIZE), (gy * C.CELL_SIZE)

        face_angle = np.radians(90 - self.face_direction.value * C.ROTATION_STEP_ANGLE)
        end_x = C.CELL_CENTER - int(C.AGENT_SIZE * np.sin(face_angle))
        end_y = C.CELL_CENTER + int(C.AGENT_SIZE * np.cos(face_angle))

        # agent cell
        center_cord = (C.CELL_CENTER, C.CELL_CENTER)
        agent_surface = pygame.Surface((C.CELL_SIZE, C.CELL_SIZE), pygame.SRCALPHA)

        agent_surface.fill(C.VISIBLE_CELL_COLOR)

        pygame.draw.circle(
            agent_surface,
            C.AGENT_COLOR,
            center_cord,
            C.AGENT_SIZE,
        )

        pygame.draw.line(
            agent_surface,
            (20, 16, 38),
            center_cord,
            (end_y, end_x),
            width=5,
        )

        canvas.blit(agent_surface, (py, px))

        # visible_cells
        for cell in self.visible_cells:
            gx, gy = divmod(cell, board_width)
            px = gx * C.CELL_SIZE
            py = gy * C.CELL_SIZE
            agent_surface = pygame.Surface((C.CELL_SIZE, C.CELL_SIZE), pygame.SRCALPHA)
            agent_surface.fill(C.VISIBLE_CELL_COLOR)
            canvas.blit(agent_surface, (py, px))

        for cell in self.edge_cells:
            x, y = divmod(cell, board_width)
            x = (x * C.CELL_SIZE) + C.CELL_CENTER
            y = (y * C.CELL_SIZE) + C.CELL_CENTER

            pygame.draw.circle(canvas, C.EDGE_CELL_POINT_COLOR, (y, x), 3)


class ActionSpace:
    def __init__(self, num_actions: int):
        self.n = num_actions
        self.action_space = list(range(num_actions))

    def sample(self):
        return global_rand_manager.random.choice(self.action_space)


class Environment:
    def __init__(
        self,
        height: int = 10,
        width: int = 10,
        view_width: NumVal = 0.6,
        view_length: NumVal = 0.75,
        env_mode: str | EnvParams = "easy",
        seed: Optional[int] = None,
        num_rays: int = C.DEFAULT_NUM_RAYS,
        max_steps: int = 1000,
    ) -> None:
        self._width = width
        self._height = height

        self._max_steps = max_steps
        self._step_count: int = 0

        view_width, view_length, view_length_max = _calculate_view_wl(
            height, width, view_length, view_width
        )

        self.action_space = ActionSpace(10)
        self._rman = RNDManager(seed)
        if seed is not None:
            global_rand_manager.set_seed(seed)

        self._backbone = EnvBackBone(
            height,
            width,
            env_params=_get_env_params(env_mode),
            rman=self._rman,
        )

        self._agent = Agent(
            view_width, view_length, view_length_max, num_rays, self._backbone
        )

    def episode_info(self) -> TripleInt:
        return (
            self._backbone._step_count,
            self._backbone._kill_count,
            self._backbone._harvest_count,
        )

    def reset(self, seed: Optional[int] = None) -> Observations:
        if seed is not None:
            self._rman.set_seed(seed)
            global_rand_manager.set_seed(seed)

        self._backbone.reset()
        self._agent.reset()

        self._step_count = 0

        _, state, _ = self._agent.compute_state()
        return state

    def step(self, action: int) -> Tuple[Observations, float, bool, bool]:
        """
        Returns:
            reward: float
            observations: perception,loot_mole_damage indications (3), player state
            terminal: bool
            truncate: bool
        """

        assert self._agent.is_alive, "Reset the Environment before calling step"

        self._step_count += 1

        self._backbone.clear_trails()
        self._backbone.step_data.reset()

        self._backbone.step_data.prev_cell = self._agent.current_cell
        self._agent.do_action(action)

        self._backbone.take_step(self._agent.current_cell)
        self._agent.update_health()

        reward, state, terminal = self._agent.compute_state()
        # state including observations and player state

        truncate = self._step_count >= self._max_steps
        # max steps reached
        if truncate:
            self._agent.is_alive = False

        if not (terminal or truncate):
            self._backbone.spawn_objects_randomly(self._agent.current_cell)

        return state, reward, terminal, truncate

    def render(self):
        if not hasattr(self, "board_canvas"):
            self.board_canvas = pygame.Surface(
                (
                    self._width * C.CELL_SIZE,
                    self._height * C.CELL_SIZE + C.STAT_BAR_HEIGHT,
                )
            )

        canvas = self.board_canvas
        canvas.fill((255, 255, 255))
        self._backbone.board.render(canvas)
        self._agent.draw(canvas)

        # draw stats bar
        stat_bar_start_y: int = self._height * C.CELL_SIZE
        mwidth: int = self._width * C.CELL_SIZE
        stat_bar_bg: ColorRGB = (196, 196, 196)
        loading_bar_bg: ColorRGB = (112, 112, 114)
        text_color: ColorRGB = (0, 0, 0)
        health_color: ColorRGB = (0, 255, 0)
        poison_color: ColorRGB = (255, 0, 0)
        vision_mode_color: ColorRGB = (5, 231, 252)

        stat_bar_rect = pygame.Rect(
            0,
            stat_bar_start_y,
            mwidth,
            C.STAT_BAR_HEIGHT,
        )

        pygame.draw.rect(canvas, stat_bar_bg, stat_bar_rect)

        stat_font = pygame.font.Font(None, 24)

        stat_stats_bar_start = 20

        kill_count = self._backbone._kill_count
        kill_text = stat_font.render(f"Kills: {kill_count}", True, text_color)
        canvas.blit(kill_text, (stat_stats_bar_start, stat_bar_start_y + 10))

        step_count = self._backbone._step_count
        step_text = stat_font.render(f"Steps: {step_count}", True, text_color)
        canvas.blit(step_text, (stat_stats_bar_start, stat_bar_start_y + 40))

        harvest_count = self._backbone._harvest_count
        harvest_text = stat_font.render(f"Harvests: {harvest_count}", True, text_color)
        canvas.blit(harvest_text, (stat_stats_bar_start, stat_bar_start_y + 70))

        health_bar_width = 200
        health_bar_height = 20

        health_bar_start = int(mwidth * 0.3)
        health_bar_y = stat_bar_start_y + 10
        max_health = 100
        current_health = self._agent._health

        pygame.draw.rect(
            canvas,
            loading_bar_bg,
            (
                health_bar_start,
                health_bar_y,
                health_bar_width,
                health_bar_height,
            ),
        )
        health_fill_width = int((current_health / max_health) * health_bar_width)
        pygame.draw.rect(
            canvas,
            health_color,
            (
                health_bar_start,
                health_bar_y,
                health_fill_width,
                health_bar_height,
            ),
        )

        ammo_text = stat_font.render(
            f"Ammo: {self._agent._ammos_in_inv}", True, text_color
        )
        canvas.blit(
            ammo_text,
            (health_bar_start, health_bar_y + health_bar_height + 10),
        )

        vision_mode_x = health_bar_start + health_bar_width + 20
        vision_mode_y = stat_bar_start_y + 15

        if self._agent._vision_mode.is_long:
            triangle_points = [
                (vision_mode_x, vision_mode_y + 20),
                (vision_mode_x - 10, vision_mode_y),
                (vision_mode_x + 10, vision_mode_y),
            ]
            pygame.draw.polygon(canvas, vision_mode_color, triangle_points)
        else:
            bounding_rect = pygame.Rect(vision_mode_x - 10, vision_mode_y, 20, 20)

            pygame.draw.ellipse(canvas, vision_mode_color, bounding_rect)

            pygame.draw.rect(
                canvas,
                stat_bar_bg,
                (
                    vision_mode_x - 10,
                    vision_mode_y + 10,
                    20,
                    10,
                ),
            )

        poison_level = self._backbone.env_air_poison_level / 100
        poison_bar_height = 50
        poison_bar_width = 10
        poison_bar_x = int(mwidth * 0.9)
        poison_bar_y = stat_bar_start_y + 10

        pygame.draw.rect(
            canvas,
            loading_bar_bg,
            (poison_bar_x, poison_bar_y, poison_bar_width, poison_bar_height),
        )
        poison_fill_height = int(
            min(poison_level * poison_bar_height, poison_bar_height)
        )
        pygame.draw.rect(
            canvas,
            poison_color,
            (
                poison_bar_x,
                poison_bar_y + poison_bar_height - poison_fill_height,
                poison_bar_width,
                poison_fill_height,
            ),
        )
        ammo_text = stat_font.render(
            f"{self._backbone.env_air_poison_level * self._backbone._env_poison_scalar:.1f}/dps",
            True,
            text_color,
        )
        canvas.blit(
            ammo_text,
            (poison_bar_x - 20, poison_bar_y + poison_bar_height + 10),
        )

        return np.array(
            pygame.surfarray.pixels3d(self.board_canvas)
        )  # , axes=(1, 0, 2))


def _calculate_view_wl(
    height: int, width: int, view_length: NumVal, view_width: NumVal
) -> Tuple[int, int, int]:
    _min_side = min(width, height)

    _max_view_width = min(_min_side // 2, C.MAX_VIEW_WIDTH)
    _max_view_length = min(_min_side - 2, C.MAX_VIEW_LENGTH)

    # the normal view length is a number that satisfy this:
    # k + k*f = N, here N is the max_view_len and k is the max normal view
    # length, so after hunter view add on the view length will be in range
    # k + k *f = N -> k*(1+f) = N -> k = N/(1+f)
    _max_view_length = int(
        _max_view_length / (1 + C.HUNTER_VIEW_LENGTH_INCREASE_FACTOR)
    )

    # calculating the percentage of the acceptable views if its a float
    if view_width <= 1:
        view_width = int(_max_view_width * view_width)
        view_width -= 1 - view_width % 2  # making it odd
        view_width = max(view_width, C.MIN_VIEW_WIDTH)

    if view_length <= 1:
        view_length = int(_max_view_length * view_length)
        view_length = max(view_length, C.MIN_VIEW_LENGTH)

    # should be odd, then only we can able to divide the width by equal parts
    # fixing one in the center during change of hunter vision
    assert (
        C.MIN_VIEW_WIDTH <= view_width <= _max_view_width
    ) and view_width % 2 == 1, f"View Width {view_width} is invalid"

    assert (
        C.MIN_VIEW_LENGTH <= view_length <= _max_view_length
    ), f"View length {view_length} is invalid"

    _hunter_vision_add_on: int = int(view_length * C.HUNTER_VIEW_LENGTH_INCREASE_FACTOR)

    assert (
        _hunter_vision_add_on + view_length <= _max_view_length
    ), "Hunter vision length exceeds the minimum view length"

    return (
        int(view_width),
        int(view_length),
        int(_hunter_vision_add_on + view_length),
    )


def _get_env_params(env_mode: str | EnvParams) -> EnvParams:
    if isinstance(env_mode, str):
        env_mode = C.ENV_MODES[env_mode]

    assert (
        env_mode.mole_prop + env_mode.wall_prop + env_mode.plant_prop
    ) <= 0.8, "Total Environment Object Proportion is more than 80%"

    assert (
        sum(env_mode.wall_distribution) == 1
    ), "Wall Distribution should sum up to one"

    return env_mode
