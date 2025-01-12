import time
from typing import Any

from homegym.mazeharvest import Environment
import pygame


def play(
    env: Environment,
    agent: Any,
    title: str = "MazeHarvest",
    frame_delay: float = 0.03,
    wait_for_quit: bool = True,
    no_head: bool = False,
):
    observation = env.reset()

    if no_head:
        while True:
            action = agent(observation)
            observation, _, done, trunc = env.step(action)

            if done or trunc:
                return env.episode_info()

    try:
        pygame.init()
        pygame.display.set_caption(title)

        d_height, d_width, _ = env.render().shape
        screen = pygame.display.set_mode((d_height, d_width))

        clock = pygame.time.Clock()
        running = True
        reward = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action = agent(observation)

            observation, reward, done, trunc = env.step(action)
            # print("Step Reward:", r)

            rgb_array = env.render()
            surf = pygame.surfarray.make_surface(rgb_array)
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            time.sleep(frame_delay)

            if done or trunc:
                running = False

            clock.tick(60)

        running = wait_for_quit
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
    except Exception as e:
        raise e
    finally:
        pygame.quit()

    return env.episode_info()
