from typing import Any

import cv2
import numpy as np
import pygame

from homegym.mazeharvest import Environment


def play(
    env: Environment,
    agent: Any,
    title: str = "MazeHarvest",
    fps: int = 30,
    wait_for_quit: bool = True,
    no_head: bool = False,
    record_vid: bool = False,
    video_name: str = "output_video",
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

        if record_vid:
            video_filename = f"{video_name}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_filename, fourcc, fps, (d_height, d_width))

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

            if record_vid:
                frame = pygame.surfarray.array3d(screen)
                frame = np.transpose(frame, (1, 0, 2))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)

            if done or trunc:
                running = False

            clock.tick(fps)

        running = wait_for_quit
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
    except Exception as e:
        raise e
    finally:
        pygame.quit()
        if record_vid:
            writer.release()
            cv2.destroyAllWindows()

    return env.episode_info()
