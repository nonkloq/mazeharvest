from homegym._envtypings import ObjectRepr
from homegym.board import (
    Board,
    Cell,
    FacingDirection,
    MovingDirection,
    ObjectList,
    StackableObject,
    BoardObject,
)
import pytest


def test_objectlist():
    mylist = ObjectList()
    n = 12
    for _ in range(n):
        mylist.add_obj(BoardObject(69, 420))

    man_obj = BoardObject(8, -8)

    mylist.add_obj(man_obj)
    n += 1
    for obj in mylist:
        assert obj.index_position == n
        n -= 1

    for x in range(4, 7):
        mylist.remove(x)

    assert len(mylist) == 10

    node = mylist.remove(3, return_node=True)
    assert node is not None
    assert node.obj.index_position == 3

    prev = 10
    for obj in mylist:
        assert obj.index_position < prev and obj.index_position != prev
        prev = obj.index_position

    assert prev == 1

    mylist.add_node(node)

    assert len(mylist) == 10 == sum(1 for _ in mylist)

    # it should also delete the objects
    mylist.clear()

    assert len(mylist) == 0


def test_cell():
    mycell1 = Cell(8)
    myobj1 = BoardObject(50, 1)
    myobj2 = BoardObject(50, 1)

    mysobj1 = StackableObject(0, 2)
    mysobj1.count = 1
    mysobj2 = StackableObject(0, 2)
    mysobj2.count = 21

    assert mycell1.capacity == 100
    assert mycell1.add(myobj1)
    assert mycell1.capacity == 50
    assert mycell1.add(myobj2)
    assert mycell1.capacity == 0
    assert mycell1.add(BoardObject(1, 1)) is False

    assert mycell1.is_blocked

    assert myobj1.cell_id == myobj2.cell_id == 8

    # adding StackableObject
    assert mycell1.add(mysobj1)
    assert mycell1.add(mysobj2)

    assert myobj1.index_position == 1
    assert myobj2.index_position == 2
    assert mysobj1.index_position == 3
    assert mysobj2.index_position == -1

    mycell1.remove(myobj1)
    assert mycell1.is_blocked is False
    assert mycell1.can_hold(25)
    assert mycell1.can_hold(75) is False

    assert myobj2.index_position == 1
    assert mysobj1.index_position == 2

    assert mycell1.capacity == 50
    i = 0
    for obj in mycell1.iterate_objects():
        if type(obj) is StackableObject:
            assert obj.count == 22
        i += 1

    assert i == 2
    assert mycell1.preference(50) == 0
    assert mycell1.preference() == 0.3
    mysobj3 = StackableObject(0, 4)
    mysobj3.count = 12

    assert mycell1.add(mysobj3)
    assert mycell1.preference() == 0.2

    assert len(list(mycell1.iterate_objects())) == 3

    mycell1.reset_cell()

    assert len(list(mycell1.iterate_objects())) == 0
    assert mycell1.preference() == 1
    assert mycell1.capacity == 100


def test_board():
    board = Board(10, 10)

    objects = [BoardObject(10, 1) for _ in range(10)]

    for i, obj in enumerate(objects):
        board.add_object(obj, i)

    for i in range(10):
        assert objects[i].cell_id == i

    board.remove_object(objects[4])

    board.add_object(objects[4], 50)
    assert objects[4].cell_id == 50 and objects[4].index_position == 1

    for x in range(5, 10):
        board.move_object(objects[x], 50)

    for i in range(4, 10):
        obj = objects[i]
        assert obj.cell_id == 50 and obj.index_position == i - 3

        assert board[i].capacity == 100

    assert board[50].capacity == 40
    assert board[50].is_blocked is False

    bigobj = BoardObject(50, 1)
    mbigobj = BoardObject(40, 1)

    with pytest.raises(AssertionError):
        board.add_object(bigobj, 50)

    assert board.add_object(mbigobj, 50) is None
    assert board[50].capacity == 0
    assert board[50].is_blocked

    # check the distance & direction calculator
    target1 = 55
    target2 = 99

    # fmt: off
    mat1 = [
        [5,5,5,5,5,5,5,5,5,5],
        [5,4,4,4,4,4,4,4,4,4],
        [5,4,3,3,3,3,3,3,3,4],
        [5,4,3,2,2,2,2,2,3,4],
        [5,4,3,2,1,1,1,2,3,4],
        [5,4,3,2,1,0,1,2,3,4],
        [5,4,3,2,1,1,1,2,3,4],
        [5,4,3,2,2,2,2,2,3,4],
        [5,4,3,3,3,3,3,3,3,4],
        [5,4,4,4,4,4,4,4,4,4]
    ]

    dir_mat1 =[
        ['↘','↘','↘','↘','↘','↓','↙','↙','↙','↙'],
        ['↘','↘','↘','↘','↘','↓','↙','↙','↙','↙'],
        ['↘','↘','↘','↘','↘','↓','↙','↙','↙','↙'],
        ['↘','↘','↘','↘','↘','↓','↙','↙','↙','↙'],
        ['↘','↘','↘','↘','↘','↓','↙','↙','↙','↙'],
        ['→','→','→','→','→','x','←','←','←','←'],
        ['↗','↗','↗','↗','↗','↑','↖','↖','↖','↖'],
        ['↗','↗','↗','↗','↗','↑','↖','↖','↖','↖'],
        ['↗','↗','↗','↗','↗','↑','↖','↖','↖','↖'],
        ['↗','↗','↗','↗','↗','↑','↖','↖','↖','↖'],
    ]

    mat2 = [
        [1,2,3,4,5,4,3,2,1,1],
        [2,2,3,4,5,4,3,2,2,2],
        [3,3,3,4,5,4,3,3,3,3],
        [4,4,4,4,5,4,4,4,4,4],
        [5,5,5,5,5,5,5,5,5,5],
        [4,4,4,4,5,4,4,4,4,4],
        [3,3,3,4,5,4,3,3,3,3],
        [2,2,3,4,5,4,3,2,2,2],
        [1,2,3,4,5,4,3,2,1,1],
        [1,2,3,4,5,4,3,2,1,0]
    ]
    dir_mat2 = [
        ['↖','↖','↖','↖','↗','↗','↗','↗','↗','↑'],
        ['↖','↖','↖','↖','↗','↗','↗','↗','↗','↑'],
        ['↖','↖','↖','↖','↗','↗','↗','↗','↗','↑'],
        ['↖','↖','↖','↖','↗','↗','↗','↗','↗','↑'],
        ['↙','↙','↙','↙','↘','↘','↘','↘','↘','↓'],
        ['↙','↙','↙','↙','↘','↘','↘','↘','↘','↓'],
        ['↙','↙','↙','↙','↘','↘','↘','↘','↘','↓'],
        ['↙','↙','↙','↙','↘','↘','↘','↘','↘','↓'],
        ['↙','↙','↙','↙','↘','↘','↘','↘','↘','↓'],
        ['←','←','←','←','→','→','→','→','→','x'],
    ]
    # fmt: on
    dir2str = {
        None: "x",
        0: "↑",  # Front
        1: "↗",
        2: "→",
        3: "↘",
        4: "↓",
        5: "↙",
        6: "←",
        7: "↖",  # FrontLeft
    }

    for i in range(10):
        for j in range(10):
            k = i * 10 + j
            dist, dir = board.distance_between(k, target1, ret_direction=True)
            assert dist == mat1[i][j] and dir2str[dir] == dir_mat1[i][j]
            dist, dir = board.distance_between(k, target2, ret_direction=True)
            assert dist == mat2[i][j] and dir2str[dir] == dir_mat2[i][j]
    assert board.distance_between(9, 9)[0] == 0

    for i in range(9):
        assert (
            board.get_next_cell(
                i,
                face_direction=FacingDirection.N,
                move_direction=MovingDirection.RIGHT,
            )
            == i + 1
        )


class MBoardObj(BoardObject):
    def __init__(self, x: int, y: int):
        super().__init__(x, y)

    @property
    def object_repr(self) -> ObjectRepr:
        return (self.object_id, 0, -1, -1)


def test_rayperception():
    # Create a 10x10 board
    board = Board(10, 10)

    # Place some objects on the board
    obj1 = MBoardObj(10, 1)
    obj2 = MBoardObj(10, 2)
    obj3 = MBoardObj(10, 3)

    board.add_object(obj1, 22)  # Place obj1 at (2, 2)
    board.add_object(obj2, 85)  # Place obj2 at (5, 5)
    board.add_object(obj3, 77)  # Place obj3 at (7, 7)

    # Test ray perception from cell 0 (0, 0) facing North
    perceptions, visible_cells, _ = board.get_ray_perception(
        current_cell=94,
        face_direction=FacingDirection.N,
        vision_range=180,
        num_rays=9,
        base=3,
        height=6,
    )

    # Check if the correct number of perceptions are returned
    assert len(perceptions) == 9

    # Check if the perceptions are in the correct format
    for perception in perceptions:
        assert len(perception) == 6  # angle, dist, obid, obt, hp, fd

    # Check if the visible cells are returned
    assert isinstance(
        visible_cells, set
    ), f"Visible cells should be a set, got {type(visible_cells)}"

    assert len(visible_cells) >= 6 * 3

    perceptions, _, _ = board.get_ray_perception(
        current_cell=55,  # Start from the center
        face_direction=FacingDirection.E,
        vision_range=60,
        num_rays=18,
        base=5,
        height=7,
    )

    assert len(perceptions) <= 9
