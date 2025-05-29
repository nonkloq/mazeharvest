import os
import traceback
from copy import deepcopy
from math import cos, pi, sin
from pathlib import Path

from direct.gui.DirectGui import DirectWaitBar
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    AmbientLight,
    CardMaker,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LColor,
    LineSegs,
    NodePath,
    NodePathCollection,
    Point3,
    PointLight,
    Quat,
    TextNode,
    TransparencyAttrib,
    Vec3,
    WindowProperties,
)

import homegym.constants as C
from homegym.envobjs import BoardObject
from homegym.mazeharvest import Environment

# Configuration
GRID_SIZE = 2.0
WALL_HEIGHT = GRID_SIZE
PLAYER_RADIUS = 0.3 * GRID_SIZE
PLAYER_COLOR = LColor(0.8, 0.1, 0.1, 1.0)
BASE_STONE = LColor(0.58, 0.58, 0.56, 1.0)

WALL_COLOR = BASE_STONE
GROUND_COLOR = LColor(0.522, 0.522, 0.504, 1.0)

BROKEN_WALL_COLOR = LColor(0.66, 0.66, 0.64, 1.0)

DANGER_WALL_COLOR = LColor(0.643, 0.493, 0.476, 1.0)

HALF_HP_WALL_COLOR = LColor(0.68, 0.62, 0.52, 1.0)

PLANT_BASE_COLOR = LColor(0.2, 0.6, 0.2, 1.0)  # Base color for plants
MOLE_COLOR = LColor(0.45, 0.35, 0.25, 1.0)  # Brownish color for moles
AMMO_COLOR = LColor(0.8, 0.8, 0.2, 1.0)
SHOT_COLOR = LColor(0.9, 0.1, 0.1, 1.0)

# Visual Settings
VISIBLE_CELL_HIGHLIGHT_COLOR = LColor(1, 1, 1, 0.3)
VISIBLE_CELL_Z_OFFSET = 0.02
TRAIL_Z_OFFSET = 0.01
BROKEN_WALL_Z_OFFSET = 0.01
SMALL_ITEM_Z_OFFSET = 0.008
PLANT_Z_OFFSET = GRID_SIZE * 0.15
MOLE_Z_OFFSET_FACTOR = 0.05
WALL_CENTER_Z_OFFSET = WALL_HEIGHT / 2.0
PLAYER_Z_OFFSET = PLAYER_RADIUS

# Model Paths
MODEL_DIR = Path(__file__).resolve().parent / "../assets/3dmodels"
MODEL_DIR = MODEL_DIR.resolve()


def make_triangle(vision_shape_np):
    ls = LineSegs()
    ls.setColor(1, 1, 0, 1)
    ls.moveTo(-0.05, 0, -0.05)
    ls.drawTo(0.05, 0, -0.05)
    ls.drawTo(0, 0, 0.05)
    ls.drawTo(-0.05, 0, -0.05)
    return vision_shape_np.attachNewNode(ls.create())


def make_semicircle(vision_shape_np):
    ls = LineSegs()
    ls.setColor(0, 1, 1, 1)
    segments = 20
    radius = 0.05
    angle_step = pi / segments
    for i in range(segments + 1):
        angle = i * angle_step
        x = radius * cos(angle)
        y = radius * sin(angle)
        if i == 0:
            ls.moveTo(x, 0, y)
        else:
            ls.drawTo(x, 0, y)
    return vision_shape_np.attachNewNode(ls.create())


def apply_color(node_path, color):
    node_path.clearTexture()
    if color:
        node_path.setColor(color, 1)


def create_flat(name: str, size: float = GRID_SIZE):
    vformat = GeomVertexFormat.getV3n3t2()
    vdata = GeomVertexData(name + "_vdata", vformat, Geom.UHStatic)
    vdata.setNumRows(4)

    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    texcoord = GeomVertexWriter(vdata, "texcoord")

    s = size / 2.0
    verts_data = [
        (Point3(-s, -s, 0), Vec3(0, 0, 1), (0, 0)),
        (Point3(s, -s, 0), Vec3(0, 0, 1), (1, 0)),
        (Point3(s, s, 0), Vec3(0, 0, 1), (1, 1)),
        (Point3(-s, s, 0), Vec3(0, 0, 1), (0, 1)),
    ]
    for v, n, t in verts_data:
        vertex.addData3(v)
        normal.addData3(n)
        texcoord.addData2(t)

    tris = GeomTriangles(Geom.UHStatic)
    tris.addVertices(0, 1, 2)
    tris.closePrimitive()
    tris.addVertices(0, 2, 3)
    tris.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name + "_geomnode")
    node.addGeom(geom)
    return NodePath(node)


def create_cube(name: str, size: float = GRID_SIZE):
    vformat = GeomVertexFormat.getV3n3t2()
    vdata = GeomVertexData(name + "_vdata", vformat, Geom.UHStatic)
    vdata.setNumRows(24)

    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    texcoord = GeomVertexWriter(vdata, "texcoord")

    s = size / 2.0
    verts_data = [
        # Front (+Y)
        (Point3(-s, s, -s), Vec3(0, 1, 0), (0, 0)),
        (Point3(-s, s, s), Vec3(0, 1, 0), (0, 1)),
        (Point3(s, s, s), Vec3(0, 1, 0), (1, 1)),
        (Point3(s, s, -s), Vec3(0, 1, 0), (1, 0)),
        # Back (-Y)
        (Point3(s, -s, -s), Vec3(0, -1, 0), (0, 0)),
        (Point3(s, -s, s), Vec3(0, -1, 0), (0, 1)),
        (Point3(-s, -s, s), Vec3(0, -1, 0), (1, 1)),
        (Point3(-s, -s, -s), Vec3(0, -1, 0), (1, 0)),
        # Left (-X)
        (Point3(-s, -s, -s), Vec3(-1, 0, 0), (0, 0)),
        (Point3(-s, -s, s), Vec3(-1, 0, 0), (0, 1)),
        (Point3(-s, s, s), Vec3(-1, 0, 0), (1, 1)),
        (Point3(-s, s, -s), Vec3(-1, 0, 0), (1, 0)),
        # Right (+X)
        (Point3(s, s, -s), Vec3(1, 0, 0), (0, 0)),
        (Point3(s, s, s), Vec3(1, 0, 0), (0, 1)),
        (Point3(s, -s, s), Vec3(1, 0, 0), (1, 1)),
        (Point3(s, -s, -s), Vec3(1, 0, 0), (1, 0)),
        # Top (+Z)
        (Point3(-s, s, s), Vec3(0, 0, 1), (0, 0)),
        (Point3(-s, -s, s), Vec3(0, 0, 1), (0, 1)),
        (Point3(s, -s, s), Vec3(0, 0, 1), (1, 1)),
        (Point3(s, s, s), Vec3(0, 0, 1), (1, 0)),
        # Bottom (-Z)
        (Point3(-s, -s, -s), Vec3(0, 0, -1), (0, 0)),
        (Point3(-s, s, -s), Vec3(0, 0, -1), (0, 1)),
        (Point3(s, s, -s), Vec3(0, 0, -1), (1, 1)),
        (Point3(s, -s, -s), Vec3(0, 0, -1), (1, 0)),
    ]
    for v, n, t in verts_data:
        vertex.addData3(v)
        normal.addData3(n)
        texcoord.addData2(t)

    tris = GeomTriangles(Geom.UHStatic)
    for i in range(0, 24, 4):
        tris.addVertices(i + 0, i + 1, i + 2)
        tris.closePrimitive()
        tris.addVertices(i + 0, i + 2, i + 3)
        tris.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name + "_geomnode")
    node.addGeom(geom)
    return NodePath(node)


class RealLifeSim(ShowBase):
    def __init__(
        self,
        env: Environment,
        action_interval: float = 0.1,
        policy_function=None,
    ):
        ShowBase.__init__(self)
        self.win.setClearColor(LColor(0.1, 0.1, 0.1, 1))

        self._env = env
        C.AGENT_CENTER_BOUND_DENOM = 6  # making the bound even shorter
        self.grid_height = self._env._height
        self.grid_width = self._env._width
        self.agent_center = True

        self.action_interval = action_interval
        self.policy_function = policy_function

        # Interpolation state
        self.last_action_time = 0.0
        self.last_interpolation_time = 0.0
        self.interpolation_duration = self.action_interval / 3
        self.dynamic_environment_root = self.render.attachNewNode(
            "DynamicEnvironmentRoot"
        )
        self._tracked_objects = {}
        self._tracked_highlights = NodePathCollection()

        # Player node state for interpolation
        self.player_np = None
        self._player_prev_pos = Point3(0, 0, 0)
        self._player_target_pos = Point3(0, 0, 0)
        self._player_prev_faced = (
            self._env._agent.face_direction.value
            if self._env
            and hasattr(self._env, "_agent")
            and hasattr(self._env._agent, "face_direction")
            else 0.0
        )
        self._player_target_faced = self._player_prev_faced

        props = WindowProperties()
        props.setSize(800, 600)
        props.setTitle("MazeHarvest3d")
        self.win.requestProperties(props)

        self.camera_mode = "TPP"
        self.disableMouse()
        self.setup_lighting()
        self.generate_static_world()
        self.create_player_node()
        self.setup_controls()
        self.update_camera_mode()

        # saving default to restore it later in FPP
        self.default_fov = deepcopy(self.camLens.getFov())
        self._player_prev_cell = self._player_target_cell = 0

        # Bottom-left Text: Kills and Harvests
        self.kill_text = OnscreenText(
            text="Kills: 0",
            pos=(-1.3, -0.88),
            scale=0.05,
            align=TextNode.ALeft,
            fg=(1, 1, 1, 1),  # White text
        )
        self.harvest_text = OnscreenText(
            text="Harvests: 0",
            pos=(-1.3, -0.94),
            scale=0.05,
            align=TextNode.ALeft,
            fg=(1, 1, 1, 1),
        )

        # Center-bottom Health Bar
        self.health_bar = DirectWaitBar(
            text="",
            value=100,
            pos=(0, 0, -0.9),
            scale=0.5,
            barColor=(0.2, 1, 0.1, 0.6),
        )
        self.health_bar["range"] = 100

        # Ammo Text (below health bar, center-bottom)
        self.ammo_text = OnscreenText(
            text="Ammo: 0",
            pos=(0, -0.98),
            scale=0.05,
            fg=(1, 1, 1, 1),
            mayChange=True,
        )

        # Env Poisonlevel bar (bottom-right corner, rotated)
        self.poison_bar = DirectWaitBar(
            text="",
            value=0,
            range=1.0,
            pos=(1.25, 0, -0.9),
            scale=(0.05, 1, 0.5),
            barColor=(1, 0.3, 0.15, 1),
        )
        self.poison_bar.setR(-90)

        self.vision_shape_np = NodePath("vision_shape")
        self.vision_shape_np.reparentTo(self.aspect2d)
        self.vision_shape_np.setPos(0.6, 0, -0.94)

        self.setup_scene_interpolation_targets()
        self.interpolate_scene(1.0)  # Snap to initial state immediately

        self.taskMgr.add(self.update_task, "updateTask")
        self.taskMgr.add(self.interpolation_task, "interpolationTask")

        self.instructions = self.create_instructions()

    def create_instructions(self):
        text = "F1: First Person View\nF2: Bird's Eye View\nF3: Full Grid View\nC: Toggle Agent centric view\nESC: Quit"
        text_node = TextNode("instructions")
        text_node.setText(text)
        text_node.setAlign(TextNode.ALeft)
        text_node.setTextColor(1, 1, 1, 1)
        text_np = self.aspect2d.attachNewNode(text_node)
        text_np.setScale(0.05)
        return text_np

    def setup_lighting(self):
        # Ambient light
        ambient_light = AmbientLight("ambient_light")
        ambient_light.setColor(LColor(0.65, 0.65, 0.65, 1))
        self.ambient_light_np = self.render.attachNewNode(ambient_light)
        self.render.setLight(self.ambient_light_np)
        # Player's Point light
        self.player_light = PointLight("player_light")
        self.player_light.setColor(LColor(0.9, 0.85, 0.88, 1))
        # Attenuation: (constant, linear, quadratic)
        self.player_light.setAttenuation(Vec3(0.7, 0.05, 0.01))

        self.player_light_np = NodePath(self.player_light)
        self.render.setShaderAuto()

    def grid_to_world(self, row, col, z_offset=0.0):
        display_row, display_col = row, col
        if self.agent_center:
            roff, coff = self._env.get_center_offset()
            display_row = (row + roff) % self.grid_height
            display_col = (col + coff) % self.grid_width
        else:
            display_row, display_col = row, col

        x = (display_col + 0.5) * GRID_SIZE
        y = (display_row + 0.5) * GRID_SIZE
        z = z_offset
        return Point3(x, y, z)

    def generate_static_world(self):
        ground_cm = CardMaker("ground_cm")
        ground_cm.setFrame(
            0, self.grid_width * GRID_SIZE, 0, self.grid_height * GRID_SIZE
        )
        ground = self.dynamic_environment_root.attachNewNode(
            ground_cm.generate()
        )
        self.ground_np = ground
        ground.setPos(0, 0, 0)
        ground.lookAt(0, 0, -1)

        apply_color(ground, GROUND_COLOR)
        # ground.setTexScale removed
        # self.ground_ts removed

        ground.setBin("background", 0)
        ground.setDepthWrite(0)

    def create_player_node(self):
        try:
            self.player_np = self.loader.loadModel(
                os.path.join(MODEL_DIR, "Sphere.egg")
            )
            if not self.player_np or self.player_np.isEmpty():
                raise Exception("Loading sphere model failed.")
            self.player_np.setScale(PLAYER_RADIUS * 0.5)
            self.player_np.setZ(PLAYER_RADIUS)
            apply_color(self.player_np, PLAYER_COLOR)
            self.player_np.reparentTo(self.dynamic_environment_root)
            self.player_np.setName("Player")
        except Exception as e:
            print(f"Warning: Could not load sphere model: {e}. Exiting Sim.")
            self.userExit()

        self.player_light_root = self.dynamic_environment_root.attachNewNode(
            "PlayerLightRoot"
        )
        self.player_light_np.reparentTo(self.player_light_root)
        self.player_light_np.setPos(0, 0, PLAYER_RADIUS * 1.8)
        self.player_np.hide()

    def get_object_z_offset(self, obj: BoardObject):
        oid, typ, hlt, _ = obj.object_repr
        if oid == C.WALL_UID:
            return BROKEN_WALL_Z_OFFSET if hlt == 0 else WALL_CENTER_Z_OFFSET
        if oid == C.MOLE_UID:
            return MOLE_Z_OFFSET_FACTOR
        if oid == C.PLANT_UID:
            plant_height = GRID_SIZE * 0.3
            return PLANT_Z_OFFSET + (plant_height / 2.0)
        if oid == C.AMMO_UID or oid == C.SHOT_UID:
            item_radius = GRID_SIZE * 0.1
            return SMALL_ITEM_Z_OFFSET + item_radius
        if oid == C.TRAIL_UID:
            return TRAIL_Z_OFFSET
        return 0.0

    def _create_object_node(self, obj: BoardObject, cell_id: int):
        oid, typ, hlt, _ = obj.object_repr
        name = f"obj_{cell_id}_{oid}_{typ}_{hlt}"

        node_path = None
        object_color = LColor(0.5, 0.5, 0.5, 1.0)  # Default gray

        if oid == C.WALL_UID:
            if hlt == 0:
                node_path = create_flat(name + "_flat", size=GRID_SIZE)
                object_color = BROKEN_WALL_COLOR
            else:
                node_path = create_cube(name + "_cube", size=GRID_SIZE)
                if typ == 1.0:
                    object_color = DANGER_WALL_COLOR
                elif hlt != -1 and hlt <= 0.5:
                    object_color = HALF_HP_WALL_COLOR
                else:
                    object_color = WALL_COLOR
            apply_color(node_path, object_color)

        elif oid == C.MOLE_UID:
            scale = GRID_SIZE * 0.6 * max(typ, 0.15)
            try:
                node_path = self.loader.loadModel(
                    os.path.join(MODEL_DIR, "Mouse.egg")
                )
                apply_color(node_path, MOLE_COLOR)
                node_path.setScale(scale)
            except Exception:
                print(f"Error loading mole model for {cell_id}. Exiting Sim.")
                self.userExit()

        elif oid == C.TRAIL_UID:
            return None

        elif oid == C.AMMO_UID:
            try:
                node_path = self.loader.load_model(
                    os.path.join(MODEL_DIR, "Tetrahedron.egg")
                )
                node_path.setScale(GRID_SIZE * 0.1)
                apply_color(node_path, AMMO_COLOR)
            except Exception:
                print(f"Error loading ammo model for {cell_id}. Exiting Sim.")
                self.userExit()

        elif oid == C.SHOT_UID:
            try:
                node_path = self.loader.loadModel("models/misc/sphere")
                node_path.setScale(GRID_SIZE * 0.1)
                apply_color(node_path, SHOT_COLOR)
            except Exception:
                print(f"Error loading shot model for {cell_id}. Exiting Sim.")
                self.userExit()

        elif oid == C.PLANT_UID:
            object_color = LColor(
                PLANT_BASE_COLOR.getX() + typ * 0.8,
                PLANT_BASE_COLOR.getY() - typ * 0.1,
                PLANT_BASE_COLOR.getZ() + typ * 0.1,
                1.0,
            )
            try:
                node_path = self.loader.load_model(
                    os.path.join(MODEL_DIR, "Icosahedron.egg")
                )
                node_path.setScale(GRID_SIZE * 0.1)
                apply_color(node_path, object_color)
            except Exception:
                print(f"Error loading plant model for {cell_id}. Exiting Sim.")
                self.userExit()

        if node_path:
            node_path.setName(name)
            node_path.setTransparency(TransparencyAttrib.MAlpha)
            node_path.setColorScale(1, 1, 1, 1)
            if node_path.node():
                node_path.node().setFinal(True)

        return node_path

    def setup_scene_interpolation_targets(self):
        keys_to_delete = []
        for logical_id, entry in list(self._tracked_objects.items()):
            if (
                entry["state"] == "disappearing"
                and entry["node"].getColorScale().getW() < 0.01
            ):
                entry["node"].removeNode()
                keys_to_delete.append(logical_id)
            if (
                "old_node_fading" in entry
                and entry["old_node_fading"]
                and entry["old_node_fading"].getColorScale().getW() < 0.01
            ):
                entry["old_node_fading"].removeNode()
                entry["old_node_fading"] = None

        for key in keys_to_delete:
            del self._tracked_objects[key]

        current_env_objects_by_logical_id = {}
        for cell_idx, cell in enumerate(
            self._env._backbone.board._board_array
        ):
            abs_row, abs_col = divmod(cell_idx, self.grid_width)
            for obj in cell.iterate_objects():
                oid, typ, hlt, _ = obj.object_repr
                logical_id = (hash(obj), oid)
                if oid == C.TRAIL_UID:
                    logical_id = (hash(obj), oid, typ)
                elif oid == C.WALL_UID:
                    logical_id = (hash(obj), oid, typ)
                current_env_objects_by_logical_id[logical_id] = (
                    abs_row,
                    abs_col,
                    obj,
                    obj.object_repr,
                )

        next_tracked_objects = {}
        disappearing_logical_ids = set(self._tracked_objects.keys())

        for logical_id, (
            abs_row,
            abs_col,
            obj,
            obj_repr,
        ) in current_env_objects_by_logical_id.items():
            disappearing_logical_ids.discard(logical_id)
            target_pos = self.grid_to_world(
                abs_row, abs_col, self.get_object_z_offset(obj)
            )

            if logical_id in self._tracked_objects:
                entry = self._tracked_objects[logical_id]
                node = entry["node"]
                entry["prev_pos"] = node.getPos(self.render)
                entry["target_pos"] = target_pos
                prev_repr = entry.get("current_repr", None)

                if obj_repr[0] == C.WALL_UID and prev_repr != obj_repr:
                    if "old_node_fading" in entry and entry["old_node_fading"]:
                        entry["old_node_fading"].removeNode()
                    entry["old_node_fading"] = node
                    new_node = self._create_object_node(
                        obj, abs_row * self.grid_width + abs_col
                    )
                    if new_node:
                        new_node.reparentTo(self.dynamic_environment_root)
                        new_node.setPos(target_pos)
                        new_node.setColorScale(1, 1, 1, 0)
                        new_node.setTransparency(TransparencyAttrib.MAlpha)
                        entry["node"] = new_node
                    entry["state"] = "type_changing"
                    entry["current_repr"] = obj_repr
                else:
                    entry["state"] = "stable"
                    entry["node"].setColorScale(1, 1, 1, 1)
                    entry[
                        "node"
                    ].clearTransparency()  # Make fully opaque and solid
                entry["abs_cell"] = abs_row * self.grid_width + abs_col
                entry["prev_hpr"] = entry["faced"]
                entry["faced"] = obj_repr[-1]

                next_tracked_objects[logical_id] = entry
            else:
                node = self._create_object_node(
                    obj, abs_row * self.grid_width + abs_col
                )
                if node:
                    node.reparentTo(self.dynamic_environment_root)
                    node.setPos(target_pos)
                    node.setColorScale(1, 1, 1, 0)
                    node.setTransparency(TransparencyAttrib.MAlpha)
                    next_tracked_objects[logical_id] = {
                        "node": node,
                        "prev_pos": target_pos,
                        "target_pos": target_pos,
                        "current_repr": obj_repr,
                        "abs_cell": (abs_row * self.grid_width + abs_col),
                        "old_node_fading": None,
                        "state": "appearing",
                        "faced": obj_repr[-1],
                        "prev_hpr": obj_repr[-1],
                    }

        for logical_id in disappearing_logical_ids:
            entry = self._tracked_objects[logical_id]
            node = entry["node"]
            if "old_node_fading" in entry and entry["old_node_fading"]:
                entry["old_node_fading"].removeNode()
                entry["old_node_fading"] = None
            entry["prev_pos"] = node.getPos(self.render)
            entry["target_pos"] = node.getPos(self.render)
            entry["state"] = "disappearing"
            entry["current_opacity"] = node.getColorScale().getW()
            next_tracked_objects[logical_id] = entry
        self._tracked_objects = next_tracked_objects

        for highlight in self._tracked_highlights:
            highlight.removeNode()
        self._tracked_highlights = NodePathCollection()

        if self.camera_mode != "FPP":
            visible_cells = self._env._agent.visible_cells.union(
                set([self._env._agent.current_cell])
            )
            for cell_idx in visible_cells:
                abs_row, abs_col = divmod(cell_idx, self.grid_width)
                highlighter = create_flat(
                    f"highlight_{cell_idx}", size=GRID_SIZE
                )
                highlighter.setColor(VISIBLE_CELL_HIGHLIGHT_COLOR)
                highlighter.setTransparency(TransparencyAttrib.MAlpha)
                highlighter.setLightOff(1)
                world_pos = self.grid_to_world(
                    abs_row, abs_col, VISIBLE_CELL_Z_OFFSET
                )
                highlighter.reparentTo(self.dynamic_environment_root)
                highlighter.setPos(world_pos)
                self._tracked_highlights.addPath(highlighter)

        agent_cell = self._env._agent.current_cell
        abs_row, abs_col = divmod(agent_cell, self.grid_width)
        target_player_world_pos = self.grid_to_world(
            abs_row, abs_col, PLAYER_Z_OFFSET
        )

        self._player_prev_pos = self.player_np.getPos(self.render)
        self._player_target_pos = target_player_world_pos
        self._player_prev_faced = self._player_target_faced
        self._player_target_faced = self._env._agent.face_direction.value

        self.player_light_root.setPos(
            target_player_world_pos - Vec3(0, 0, PLAYER_Z_OFFSET)
        )

        visible_cells_indices = self._env._agent.visible_cells.union(
            set([self._env._agent.current_cell])
        )

        for _, entry in self._tracked_objects.items():
            nodes_to_light = []
            if entry["node"]:
                nodes_to_light.append(entry["node"])
            if "old_node_fading" in entry and entry["old_node_fading"]:
                nodes_to_light.append(entry["old_node_fading"])

            for node_to_light in nodes_to_light:
                node_to_light.clearLight()
                node_to_light.setLight(self.ambient_light_np)

                # Check if the object's cell is in the visible set
                obj_abs_cell_idx = entry["abs_cell"]
                if obj_abs_cell_idx in visible_cells_indices:
                    node_to_light.setLight(self.player_light_np)

        # Player model lighting
        if self.player_np:
            self.player_np.clearLight()
            self.player_np.setLight(self.ambient_light_np)
            self.player_np.setLight(self.player_light_np)

    def interpolate_scene(self, change: float):
        """
        Interpolates positions and opacity for all tracked dynamic objects and the player.
        Called every frame while interpolation is active.
        """
        change = max(0.0, min(1.0, change))

        for logical_id, entry in list(self._tracked_objects.items()):
            node = entry["node"]
            prev_pos = entry["prev_pos"]
            target_pos = entry["target_pos"]
            state = entry["state"]
            old_node_fading = entry.get("old_node_fading", None)

            interp_pos = prev_pos * (1.0 - change) + target_pos * change
            node.setPos(interp_pos)
            if old_node_fading:
                old_node_fading.setPos(interp_pos)
            v2c = (
                logical_id[2]
                if logical_id[1] == C.TRAIL_UID
                else logical_id[1]
            )
            is_mole = v2c == C.MOLE_UID
            if entry["faced"] != -1:
                rfacd = entry["faced"]
                pfacd = entry["prev_hpr"]

                effective_dir = pfacd * (1 - change) + rfacd * change
                yaw = (effective_dir * 45 + 180) % 360
                node.setHpr(yaw, 0, 0)

            if state == "appearing":
                node.setColorScale(1, 1, 1, 1 if is_mole else change)
                node.show()

            elif state == "disappearing":
                start_opacity = entry.get("current_opacity", 1.0)
                node.setColorScale(
                    1, 1, 1, 0 if is_mole else start_opacity * (1.0 - change)
                )
                if node.getColorScale().getW() < 0.01:
                    node.hide()

            elif state == "type_changing":
                if node:
                    node.setColorScale(1, 1, 1, change)
                    node.show()
                if old_node_fading:
                    old_node_fading.setColorScale(1, 1, 1, 1.0 - change)
                    old_node_fading.show()
                    if old_node_fading.getColorScale().getW() < 0.01:
                        old_node_fading.hide()

            elif state == "stable":
                node.show()

            # if the shift is drastic don't show the transition
            if (target_pos - prev_pos).length() > 4:
                node.hide()

        interp_player_pos = (
            self._player_prev_pos * (1.0 - change)
            + self._player_target_pos * change
        )

        if self.player_np:
            self.player_np.setPos(interp_player_pos)

        # Player light root position moves with the player (at player's feet level)
        interp_player_light_root_pos = interp_player_pos - Vec3(
            0, 0, PLAYER_Z_OFFSET
        )
        if self.player_light_root:
            self.player_light_root.setPos(interp_player_light_root_pos)

    def setup_controls(self):
        self.accept("escape", self.userExit)
        self.accept("f1", self.set_camera_mode, ["FPP"])
        self.accept("f2", self.set_camera_mode, ["TPP"])
        self.accept("f3", self.set_camera_mode, ["Full"])
        self.accept("c", self.toggle_agent_center)

    def toggle_agent_center(self):
        self.agent_center = not self.agent_center
        self.setup_scene_interpolation_targets()
        self.interpolate_scene(1.0)
        self.update_camera_mode()  # Update camera based on new centered view

    def set_camera_mode(self, mode):
        if mode == self.camera_mode:
            return
        # Reset FOV after using FPP
        if self.camera_mode == "FPP":
            self.camLens.setFov(self.default_fov)

        self.camera_mode = mode
        self.update_camera_mode()
        self.update_player_visibility()

    def update_fpp_camera(self, change: float):
        """
        Updates the First-Person Perspective camera position and orientation.

        If 'change' is None or 1.0, snaps the camera to the agent's
        current state and updates stored previous/target directions.
        If 'change' is a float between 0 and 1, interpolates
        the camera position and orientation.
        """
        is_long_vision = self._env._agent._vision_mode.is_long
        self.camLens.setFov(85 - (is_long_vision * 45))

        if not self.player_np:
            return
        offset = Vec3(0, -PLAYER_RADIUS * 1.5, PLAYER_RADIUS * 0.5)
        player_rendered_pos = self.player_np.getPos(self.render)

        effective_face_dir = (
            self._player_prev_faced * (1 - change)
            + self._player_target_faced * change
        )
        camera_yaw = (effective_face_dir * 45 + 180) % 360
        rotation = Quat()
        rotation.setHpr(Vec3(camera_yaw, 0, 0))
        rotated_offset = rotation.xform(offset)

        cam_world_pos = player_rendered_pos + rotated_offset
        self.camera.setPos(cam_world_pos)
        self.camera.setHpr(camera_yaw, 0, 0)

    def update_camera_mode(self):
        """Detaches, repositions, and re-parents camera based on mode."""
        if not hasattr(self, "player_np") or not self.player_np:
            return

        self.camera.wrtReparentTo(self.render)

        if self.camera_mode == "FPP":
            self.camera.reparentTo(self.render)
            self.update_fpp_camera(1.0)  # Snap to current FPP pos
        elif self.camera_mode == "TPP":
            self.update_camera_position()  # Set initial TPP position
        elif self.camera_mode == "Full":
            self.update_camera_position()  # Set Full view position

        self.update_player_visibility()

    def update_player_visibility(self):
        """Hides player model in FPP, shows otherwise."""
        if not hasattr(self, "player_np") or not self.player_np:
            return
        if self.camera_mode == "FPP":
            self.player_np.hide()
        else:
            self.player_np.show()

    def update_camera_position(self):
        """Calculates and sets camera position/orientation for TPP and Full modes."""
        if not hasattr(self, "player_np") or not self.player_np:
            return

        # Use the player node's current (interpolated) position for camera target
        player_world_pos = self.player_np.getPos(self.render)

        if self.camera_mode == "TPP":
            # Offset relative to player: behind and above Y(-) and Z(+)
            offset = Vec3(0, -GRID_SIZE * 10.0, GRID_SIZE * 8.0)

            target_cam_pos = player_world_pos + offset

            self.camera.setPos(target_cam_pos)
            # Look slightly above player center
            look_at_pos = player_world_pos + Vec3(0, 0, PLAYER_RADIUS * 0.7)
            self.camera.lookAt(look_at_pos)
            self.camera.setR(0)

        elif self.camera_mode == "Full":
            # Center camera over the grid (using display coordinates)
            center_x = (self.grid_width * GRID_SIZE) / 2.0
            center_y = (self.grid_height * GRID_SIZE) / 2.0
            # Height based on grid size - adjust multiplier for zoom
            height = max(self.grid_width, self.grid_height) * GRID_SIZE * 1.8
            self.camera.setPos(
                center_x, center_y, height
            )  # Position above grid
            self.camera.lookAt(
                center_x, center_y, 0
            )  # Look down at the center

    def update_task(self, task):
        """Task for continuous updates (UI, TPP/Full Camera)."""
        if self.instructions:
            aspect_ratio = self.getAspectRatio()
            if aspect_ratio > 0:  # Avoid division by zero if window is weird
                self.instructions.setPos(-aspect_ratio + 0.05, 0, 0.9)

        if self.camera_mode in ["TPP", "Full"]:
            self.update_camera_position()

        return Task.cont

    def interpolation_task(self, task):
        """
        Task to handle smooth interpolation and trigger environment steps
        when the interpolation cycle completes.
        """
        current_time = task.time

        time_since_last_action = current_time - self.last_action_time
        interpolation_factor = min(
            time_since_last_action / self.action_interval, 1.0
        )
        self.interpolate_scene(interpolation_factor)

        if self.camera_mode == "FPP":
            self.update_fpp_camera(interpolation_factor)
        if time_since_last_action >= self.action_interval:
            try:
                self._player_prev_cell = self._env._agent.current_cell
                if self.policy_function is None:
                    action = self._env.action_space.sample()
                    obs, reward, done, trunc = self._env.step(action)
                    eoe = done or trunc
                else:
                    eoe = self.policy_function()

                self._player_target_cell = self._env._agent.current_cell
                self.setup_scene_interpolation_targets()

                self.last_action_time = current_time
                if eoe:
                    print("Episode finished.")
                    print(self._env.episode_info())
                    self.userExit()

                hrc = self._env._backbone._harvest_count
                kic = self._env._backbone._kill_count
                is_hunter = self._env._agent._vision_mode.is_long
                poison_lvl = (
                    self._env._backbone.env_air_poison_level
                    * self._env._backbone._env_poison_scalar
                )
                health = self._env._agent._health
                ammos = self._env._agent._ammos_in_inv

                self.kill_text.setText(f"Kills: {kic}")
                self.harvest_text.setText(f"Harvests: {hrc}")
                self.ammo_text.setText(f"Ammo: {ammos}")

                self.health_bar["value"] = health
                self.poison_bar["value"] = poison_lvl

                self.vision_shape_np.node().removeAllChildren()
                if is_hunter:
                    make_triangle(self.vision_shape_np)
                else:
                    make_semicircle(self.vision_shape_np)
            except Exception:
                print("\n!!! Error during environment step or setup !!!")
                traceback.print_exc()
                print("Exiting due to error.")
                self.userExit()
        return Task.cont


if __name__ == "__main__":
    try:
        env_width, env_height = 20, 20
        print(f"Initializing environment ({env_width}x{env_height})...")
        env = Environment(env_width, env_height, env_mode="hard", seed=50)
        env.reset()
        print("Environment reset complete.")
        print("Initializing Panda3D application...")
        app = RealLifeSim(env=env, action_interval=0.3)

        print("Starting Panda3D main loop (ESC to quit)...")
        app.run()
    except Exception:
        print("\n!!! An error occurred during setup or execution !!!")
        traceback.print_exc()
