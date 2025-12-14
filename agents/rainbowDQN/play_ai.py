import pygame
import sys
import time
import os
from constants import *
from environment.game import Game
from rainbow_agent import RainbowDQNAgent
from utils import get_state  # Ensure this matches your utils file name

# --- CONFIGURATION ---
VIEW_RANGE = 5
STATE_DIM = (VIEW_RANGE * 2 + 1) ** 2  # 169 inputs for Range 6

# CHANGE THIS to 'easy', 'medium', or 'medium-hard' to watch that specific win
LEVEL_TO_WATCH = 'easy'


def play_ai():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"Rainbow DQN - Watching '{LEVEL_TO_WATCH.upper()}' Agent")
    clock = pygame.time.Clock()

    # 1. Initialize Game with the correct difficulty level
    try:
        game = Game(level=LEVEL_TO_WATCH)
    except:
        print(f"Level '{LEVEL_TO_WATCH}' not found, defaulting to medium.")
        game = Game(level='easy')

    agent = RainbowDQNAgent(
        agent_name="RainbowBot",
        state_size=STATE_DIM,
        action_size=4
    )

    # 2. Construct the specific checkpoint path for that level
    # We look for the file created when the agent graduated (e.g., rainbow_cleared_easy.pth)
    base_dir = "checkpoints/RainbowBot"
    specific_checkpoint = os.path.join(base_dir, f"rainbow_cleared_{LEVEL_TO_WATCH}.pth")
    fallback_checkpoint = os.path.join(base_dir, "rainbow_best_agent.pth")

    if os.path.exists(specific_checkpoint):
        print(f"Loading cleared level checkpoint: {specific_checkpoint}")
        agent.load(specific_checkpoint)
    elif os.path.exists(fallback_checkpoint):
        print(f"Specific level checkpoint not found. Loading best agent instead: {fallback_checkpoint}")
        print("Note: This might be an agent trained on a different difficulty.")
        agent.load(fallback_checkpoint)
    else:
        print(f"File not found: {specific_checkpoint}")
        print("Please run train.py first until the agent beats a level!")
        return

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not game.game_over and not game.won:
            # USE LOCAL STATE
            # Make sure this function name matches your utils.py (get_state vs get_local_state)
            state = get_state(game, view_range=VIEW_RANGE)

            # Select action without noise (pure exploitation)
            action_idx = agent.select_action(state, training=False)

            directions = ['up', 'down', 'left', 'right', None]
            game.move_player(directions[action_idx])
            game.update()
        else:
            # Draw the final frame so we can see the Win/Loss
            game.draw(screen)
            pygame.display.flip()

            # Pause briefly to admire the victory
            time.sleep(2)
            game.reset()

        game.draw(screen)
        pygame.display.flip()
        clock.tick(10)  # 10 FPS for slow-motion viewing

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    play_ai()