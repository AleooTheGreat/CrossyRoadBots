import os
import csv
import pickle
from environment.game import Game
from agents.SARSA.sarsa_agent import SARSAAgent

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "NOOP"]

CHECKPOINT_DIR = "agents/SARSA/checkpoints"
CSV_LOG_PATH = "agents/SARSA/sarsa_training_log.csv"


def get_state(game, prev_car_cols):
    """
    Augmented discrete state (TABULAR, temporal):
    (
        lane_type,
        car_ahead, car_ahead_moving,
        car_left, car_left_moving,
        car_right, car_right_moving
    )
    """
    pr, pc = game.player_row, game.player_col

    lane_type = "ROAD" if any(v.row == pr for v in game.cars) else "SAFE"

    car_ahead = car_left = car_right = 0
    car_ahead_moving = car_left_moving = car_right_moving = 0

    for v in game.cars:
        r = v.row
        c = int(v.col)
        prev_c = prev_car_cols.get(id(v), c)

        dc = c - prev_c  # movement since last frame

        # AHEAD
        if r == pr - 1 and c == pc:
            car_ahead = 1
            if dc != 0:
                car_ahead_moving = 1

        # LEFT
        if r == pr and c == pc - 1:
            car_left = 1
            if dc != 0:
                car_left_moving = 1

        # RIGHT
        if r == pr and c == pc + 1:
            car_right = 1
            if dc != 0:
                car_right_moving = 1

        prev_car_cols[id(v)] = c

    return (
        lane_type,
        car_ahead, car_ahead_moving,
        car_left, car_left_moving,
        car_right, car_right_moving
    )


def train(episodes=5000, checkpoint_interval=100):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    game = Game(level="medium")
    agent = SARSAAgent(actions=ACTIONS)

    # CSV init
    with open(CSV_LOG_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["episode", "total_reward", "epsilon"])

    for ep in range(episodes):
        game.reset()

        # store previous car positions (TIMING MEMORY)
        prev_car_cols = {id(v): int(v.col) for v in game.cars}

        state = get_state(game, prev_car_cols)
        action = agent.select_action(state)

        total_reward = 0.0
        done = False
        prev_highest_row = game.highest_row

        while not done:
            if action != "NOOP":
                game.move_player(action.lower())

            game.update()

            reward = -0.01  # step penalty

            # reward only for real forward progress
            if game.highest_row < prev_highest_row:
                reward += 1.0
                prev_highest_row = game.highest_row

            if game.game_over:
                reward = -100.0
                done = True
            elif game.won:
                reward = 200.0
                done = True

            next_state = get_state(game, prev_car_cols)
            next_action = agent.select_action(next_state)

            agent.update(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action
            total_reward += reward

        # epsilon decay
        agent.epsilon = max(0.02, agent.epsilon * 0.995)

        # CSV log
        with open(CSV_LOG_PATH, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([ep, round(total_reward, 3), round(agent.epsilon, 4)])

        if ep % 100 == 0:
            print(f"Episode {ep}, total reward: {total_reward:.2f}, epsilon: {agent.epsilon:.3f}")

        # checkpoint save
        if ep > 0 and ep % checkpoint_interval == 0:
            path = os.path.join(CHECKPOINT_DIR, f"sarsa_ep_{ep}.pkl")
            with open(path, "wb") as f:
                pickle.dump(agent.Q, f)

    # final save
    with open(os.path.join(CHECKPOINT_DIR, "sarsa_final.pkl"), "wb") as f:
        pickle.dump(agent.Q, f)

    print("\nTraining finished.")
    print(f"CSV log saved to: {CSV_LOG_PATH}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    train()
