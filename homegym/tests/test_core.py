import numpy as np

from homegym.envlib import MazeGenerator, RNDManager
from homegym.mazeharvest import Environment, EnvParams, C
import pytest
import pygame


def test_maze_generator_connectivity():
    height = 49
    width = 49
    seed = 91223
    wall_prob_thres = 0.5
    max_allowed_ways_to_a_tile = 3
    noise_threshold = 0.9

    rman = RNDManager(seed)

    maze_gen = MazeGenerator(
        height, width, rman, wall_prob_thres, max_allowed_ways_to_a_tile
    )

    maze = maze_gen.generate_noise_maze(noise_threshold)

    maze_gen.assert_connectivity()

    assert maze_gen.get_steps_took < 300
    assert maze.shape == (height * width,), "Maze shape is incorrect"
    assert maze.dtype == np.bool_, "Maze dtype is incorrect"
    assert np.sum(maze) > 0, "Maze has no open cells"


def test_env():
    # Test environment initialization
    env = Environment(width=10, height=10, env_mode="easy", num_rays=3)
    assert env._agent is not None

    # Test reset
    state = env.reset()
    for x in state:
        assert isinstance(x, np.ndarray)

    assert state[0].shape == (
        env._agent._num_rays,
        6,
    )

    # Test step
    next_state, reward, terminal, trunc = env.step(0)  # No-op action
    assert isinstance(reward, float)

    for y, x in zip(state, next_state):
        assert isinstance(x, np.ndarray)
        assert np.array_equal(x, y)

    assert next_state[0].shape == (
        env._agent._num_rays,
        6,
    )

    assert isinstance(terminal, int)

    # Test multiple steps
    for _ in range(10):
        action1 = np.random.randint(0, 9)
        next_state, reward, terminal, trunc = env.step(action1)
        if terminal or trunc:
            break

    env_hard = Environment(width=20, height=20, env_mode="hard")
    _ = env_hard.reset()

    # Test invalid actions
    with pytest.raises(Exception):
        env.step(10)

    with pytest.raises(Exception):
        env.step(11)

    # Test environment behavior
    env = Environment(width=10, height=10, env_mode="easy", seed=12)
    env.reset()

    # Test agent movement
    initial_pos = env._agent.current_cell
    env.step(7)  # Move somewhere
    assert env._agent.current_cell != initial_pos

    # Test shooting
    env.reset()
    env.step(4)  # Shoot
    env.step(4)  # Shoot
    env.step(4)  # Shoot
    env.step(4)  # Shoot
    env.step(4)  # Shoot

    # random test
    env.reset()
    terminal = False
    steps = 0
    while not terminal and steps < 100:
        _, _, done, trunc = env.step(np.random.randint(0, 9))
        terminal = done or trunc
        steps += 1
    assert terminal in [0, 1]  # 0: ongoing, 1: agent died

    # test random seed

    env1 = Environment(width=10, height=10, env_mode="easy", seed=42)
    env2 = Environment(width=10, height=10, env_mode="easy", seed=42)

    state1 = env1.reset()
    state2 = env2.reset()
    for x, y in zip(state1, state2):
        assert np.array_equal(x, y)

    # shouldn't be so random for the first few steps
    for _ in range(10):
        action1 = np.random.randint(0, 9)
        next_state1, _, _, _ = env1.step(action1)
        next_state2, _, _, _ = env2.step(action1)

        for x, y in zip(next_state1, next_state2):
            assert np.array_equal(x, y)


def test_headless():
    w, h = 10, 10
    max_steps = 100  # 500
    env = Environment(
        w,
        h,
        view_length=4,
        view_width=3,
        env_mode=EnvParams(
            0, 0.2, 0.3, -1.3, 0.01, -2.17, C.high_risk_dist
        ),  # "insane",
        seed=1239012,
        max_steps=max_steps,
        num_rays=21,
    )
    env.reset()
    steps = 0
    while True:
        action = env.action_space.sample()
        steps += 1
        observation, reward, done, trunc = env.step(action)
        if done or trunc:
            break
    assert done, steps < max_steps  # agent died

    setattr(env._agent, "update_health", lambda: None)
    env.reset(seed=1239012)
    steps = 0
    while True:
        action = env.action_space.sample()
        steps += 1

        observation, reward, done, trunc = env.step(action)
        if done or trunc:
            break

    assert trunc, steps == max_steps  # max steps reached


def test_render():
    h, w = 20, 40
    max_steps = 500
    env = Environment(
        h,
        w,
        view_length=4,
        view_width=3,
        env_mode="insane",
        seed=1239012,
        max_steps=max_steps,
        num_rays=21,
    )
    setattr(env._agent, "update_health", lambda: None)

    env.reset()
    pygame.init()

    d_height, d_width, _ = env.render().shape
    screen = pygame.display.set_mode((d_height, d_width))

    running = True
    count = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        count += 1

        action = env.action_space.sample()

        observation, reward, done, trunc = env.step(action)

        # Render the observation
        rgb_array = env.render()
        surf = pygame.surfarray.make_surface(rgb_array)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        # time.sleep(0.01)

        if done or trunc:
            break

    pygame.quit()
