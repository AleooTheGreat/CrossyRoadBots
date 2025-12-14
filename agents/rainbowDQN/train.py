import numpy as np
import sys
import os
import pygame
import time
from environment.game import Game
from rainbow_agent import RainbowDQNAgent
from constants import GRID_ROWS, GRID_COLS
from utils import get_state

# --- Configuration ---
MAX_STEPS = 1000
BATCH_SIZE = 64
VIEW_RANGE = 5
CHECKPOINT_DIR = "checkpoints/RainbowBot"
MAIN_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "rainbow_best_agent.pth")

# Curriculum Settings
LEVEL_ORDER = ['easy', 'medium', 'medium-hard']
MASTERY_SCORE = 450.0
MIN_EPISODES_TO_GRADUATE = 15


def compute_reward(env, r_before, c_before, level_best_row, prev_ep_best_row, curr_ep_best_row):
    """
    Calculates reward including the 'Better than Previous Round' bonus.
    """
    r_after = env.player_row
    c_after = env.player_col

    # 1. Terminal States
    if env.game_over:
        return -2.0, True  # Death
    if env.won:
        return +2.0, True  # Victory

    # 2. Step Rewards
    reward = -0.05  # Time penalty

    # Vertical progress
    if r_after < r_before:
        reward += 0.1  # Moved up
    elif r_after > r_before:
        reward -= 0.075  # Moved down

    # 3. GLOBAL HIGH SCORE BONUS (+0.5)
    if r_after < level_best_row:
        reward += 0.5

    # 4. BEAT PREVIOUS ROUND BONUS (+0.5)
    # Logic: If we are in new territory for THIS episode (curr_ep_best_row)
    # AND that territory is deeper than where we died last time (prev_ep_best_row).
    if r_after < curr_ep_best_row:
        reward += 0.25

    # Increased clip to 2.0 so bonuses can stack (Global + Prev Round)
    return float(np.clip(reward, -2.0, 2.0)), False


def train():
    total_steps = 0
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    env = Game(level='easy')

    # Input dimensions
    side_len = (VIEW_RANGE * 2 + 1)
    state_dim = side_len * side_len
    action_dim = 4

    agent = RainbowDQNAgent(
        agent_name="RainbowBot",
        state_size=state_dim,
        action_size=action_dim,
        memory_size=50000,
        batch_size=BATCH_SIZE,
        lr=0.0005
    )

    if os.path.exists(MAIN_CHECKPOINT):
        print(f"Loading existing brain from {MAIN_CHECKPOINT}...")
        try:
            agent.load(MAIN_CHECKPOINT)
        except RuntimeError:
            print("ERROR: Architecture mismatch! Delete the old .pth file and restart.")
            return
    else:
        print("Starting fresh training.")

    # --- CURRICULUM LOOP ---
    current_level_idx = 0

    # Use a local mutable copy of the global mastery threshold
    mastery_score = MASTERY_SCORE

    while current_level_idx < len(LEVEL_ORDER):
        level_name = LEVEL_ORDER[current_level_idx]
        env.set_level(level_name)

        # Record Tracking
        level_best_row = float('inf')  # Best row ever in this level
        prev_ep_best_row = float('inf')  # Best row of the PREVIOUS episode

        print(f"\n{'=' * 50}")
        print(f" TRAINING LEVEL: {level_name.upper()}")
        print(f" Goal: Avg Score > {mastery_score}")
        print(f"{'=' * 50}")

        episode = 1
        scores = []
        wins = []  # <-- NEW: track whether each episode was a victory
        best_avg_score = -float('inf')
        victory_flag = False

        while True:
            env.reset()
            state = get_state(env, view_range=VIEW_RANGE)
            total_reward = 0
            done = False
            steps = 0

            # Current Episode Tracking
            curr_ep_best_row = env.player_row

            # Initialize Level/Prev records on first run if needed
            if level_best_row == float('inf'):
                level_best_row = env.player_row
                prev_ep_best_row = env.player_row  # Start baseline at bottom

            start_time = time.time()

            while not done and steps < MAX_STEPS:

                # Action
                action_idx = agent.select_action(state, training=True)
                move_cmd = ['up', 'down', 'left', 'right'][action_idx]

                # Step Preparation
                r_before, c_before = env.player_row, env.player_col

                # Execute Move
                env.move_player(move_cmd)
                env.update()

                next_state = get_state(env, view_range=VIEW_RANGE)
                total_steps += 1

                # Reward Calculation
                reward, done = compute_reward(
                    env,
                    r_before,
                    c_before,
                    level_best_row,
                    prev_ep_best_row,
                    curr_ep_best_row
                )

                # --- RECORD UPDATES ---
                # 1. Update Current Episode Record
                if env.player_row < curr_ep_best_row:
                    curr_ep_best_row = env.player_row

                # 2. Update Global Level Record (for logging mostly)
                if env.player_row < level_best_row:
                    print(f"  > New Level Record! Reached Row {env.player_row}")
                    level_best_row = env.player_row

                # --- 30 SECOND SUICIDE TIMER ---
                if not done:
                    if time.time() - start_time > 30.0:
                        reward = -2.0
                        done = True

                # Train
                agent.train_step(state, action_idx, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps += 1

            if env.won:
                victory_flag = True

            # End of Episode: Update Previous Episode Record for next time
            prev_ep_best_row = curr_ep_best_row

            # Record score and whether this episode was a win
            scores.append(env.score)
            wins.append(bool(env.won))  # <-- NEW: store win flag for this episode

            # compute average over last 5 (same behavior as before)
            avg_score = 0
            if len(scores) >= 5:
                avg_score = np.mean(scores[-5:])

            # --- MASTERY CHECK ---
            # Graduation condition changed so that at least one of the episodes used
            # in Avg5 must be a victory.
            # We require:
            #  - avg_score >= mastery_score
            #  - at least MIN_EPISODES_TO_GRADUATE episodes total
            #  - at least one win among the episodes included in the average (last 5)
            recent_win_in_avg = False
            if len(wins) >= 1:
                recent_win_in_avg = any(wins[-5:])  # check wins among last up to 5 episodes

            if avg_score > mastery_score and len(scores) > MIN_EPISODES_TO_GRADUATE and recent_win_in_avg:
                print(f"\n!!! LEVEL MASTERED !!!")
                print(
                    f"Agent beat level '{level_name}' with Avg Score {avg_score:.1f} and at least one win in the averaging window!")

                trophy_path = os.path.join(CHECKPOINT_DIR, f"rainbow_cleared_{level_name}.pth")
                agent.save(trophy_path)
                agent.save(MAIN_CHECKPOINT)

                current_level_idx += 1
                mastery_score -= 50.0
                break

            # --- BEST MODEL SAVING ---
            if avg_score >= best_avg_score:
                best_avg_score = avg_score
                agent.save(MAIN_CHECKPOINT)

            # --- LOGGING ---
            # Added "PrevBest" and show whether there was a recent win in Avg window
            recent_win_marker = "Y" if recent_win_in_avg else "N"
            print(
                f"[{level_name.upper()}] Ep {episode} | Score: {env.score} | LvlBest: {level_best_row} | PrevBest: {prev_ep_best_row} | Avg5: {avg_score:.1f} | BestAvg: {best_avg_score:.1f} | WinInAvg5: {recent_win_marker}")

            episode += 1

    print("\nALL LEVELS MASTERED! TRAINING COMPLETE.")

if __name__ == "__main__":
    train()