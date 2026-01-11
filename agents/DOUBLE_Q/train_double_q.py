import sys
import os
import csv
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from environment.game import Game
from agents.DOUBLE_Q.double_q_agent import DoubleQAgent
from constants import GRID_ROWS, GRID_COLS

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTIONS_MAP = {
    "UP": "up",
    "DOWN": "down",
    "LEFT": "left",
    "RIGHT": "right",
}


def get_next_run_folder(difficulty="medium-hard"):
    checkpoints_dir = "agents/DOUBLE_Q/checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    existing_runs = [d for d in os.listdir(checkpoints_dir) if d.startswith("run_") and d.endswith(f"_{difficulty}")]
    if not existing_runs:
        run_num = 1
    else:
        run_nums = [int(d.split("_")[1]) for d in existing_runs]
        run_num = max(run_nums) + 1

    run_folder = os.path.join(checkpoints_dir, f"run_{run_num}_{difficulty}")
    os.makedirs(run_folder, exist_ok=True)
    return run_folder, run_num


def _car_velocity(car) -> float:
    sp = float(getattr(car, "speed", 0.0))
    d = getattr(car, "direction", None)
    if d is None:
        return sp
    d = float(d)
    if sp < 0:
        return sp
    return sp * d


def _ttc_bin_for_cell(game, target_row: int, target_col: int, horizon: int = 3) -> int:
    if target_row < 0 or target_row >= GRID_ROWS or target_col < 0 or target_col >= GRID_COLS:
        return 0

    earliest = None
    for car in game.cars:
        if int(car.row) != int(target_row):
            continue

        vel = _car_velocity(car)
        width = float(getattr(car, "width", 1.0))

        for k in range(0, horizon + 1):
            col_k = float(car.col) + vel * k
            if col_k <= target_col < col_k + width:
                if earliest is None or k < earliest:
                    earliest = k
                break

    if earliest is None:
        return 3
    if earliest <= 0:
        return 0
    if earliest == 1:
        return 1
    if earliest == 2:
        return 2
    return 3


def _lane_is_road(game, row: int) -> int:
    return 1 if any(int(c.row) == int(row) for c in game.cars) else 0


def state_to_tabular(game, horizon: int = 3) -> tuple:
    """
    Returns:
      (
        pc,
        lane_cur, lane_up,
        in_up, in_down, in_left, in_right,
        ttc_up, ttc_up_l, ttc_up_r,
        ttc_cur, ttc_cur_l, ttc_cur_r,
        ttc_down
      )
    """
    pr, pc = int(game.player_row), int(game.player_col)

    in_up = 1 if pr - 1 >= 0 else 0
    in_down = 1 if pr + 1 < GRID_ROWS else 0
    in_left = 1 if pc - 1 >= 0 else 0
    in_right = 1 if pc + 1 < GRID_COLS else 0

    ttc_up = _ttc_bin_for_cell(game, pr - 1, pc, horizon=horizon) if in_up else 0
    ttc_up_l = _ttc_bin_for_cell(game, pr - 1, pc - 1, horizon=horizon) if (in_up and in_left) else 0
    ttc_up_r = _ttc_bin_for_cell(game, pr - 1, pc + 1, horizon=horizon) if (in_up and in_right) else 0

    ttc_cur = _ttc_bin_for_cell(game, pr, pc, horizon=horizon)
    ttc_cur_l = _ttc_bin_for_cell(game, pr, pc - 1, horizon=horizon) if in_left else 0
    ttc_cur_r = _ttc_bin_for_cell(game, pr, pc + 1, horizon=horizon) if in_right else 0

    ttc_down = _ttc_bin_for_cell(game, pr + 1, pc, horizon=horizon) if in_down else 0

    lane_cur = _lane_is_road(game, pr)
    lane_up = _lane_is_road(game, pr - 1) if in_up else 0

    return (
        pc,
        lane_cur, lane_up,
        in_up, in_down, in_left, in_right,
        ttc_up, ttc_up_l, ttc_up_r,
        ttc_cur, ttc_cur_l, ttc_cur_r,
        ttc_down,
    )


def compute_reward(env, r_before, c_before, level_best_row, prev_ep_best_row, curr_ep_best_row):
    r_after = int(env.player_row)

    if env.game_over:
        return -2.0, True, level_best_row, prev_ep_best_row, curr_ep_best_row
    if env.won:
        return +2.0, True, level_best_row, prev_ep_best_row, curr_ep_best_row

    reward = -0.05

    if r_after < r_before:
        reward += 0.1
    elif r_after > r_before:
        reward -= 0.075

    if r_after < curr_ep_best_row:
        if r_after < prev_ep_best_row:
            reward += 0.25
        curr_ep_best_row = r_after

    if r_after < level_best_row:
        reward += 0.5
        level_best_row = r_after

    reward = float(np.clip(reward, -2.0, 2.0))
    return reward, False, level_best_row, prev_ep_best_row, curr_ep_best_row


def train_double_q(
    episodes=100000,
    max_steps=800,
    difficulty="medium-hard",
    checkpoint_interval=1000,
    horizon=3,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=0.9995,
    alpha=0.25,
    alpha_min=0.08,
    alpha_decay=0.9997,
    gamma=0.99,
    optimistic_init=0.1,
    no_progress_limit=140,
):
    run_folder, run_num = get_next_run_folder(difficulty)

    print("=" * 60)
    print("Double Q-Learning (TABULAR) Training Configuration")
    print(f"Run: #{run_num} | Difficulty: {difficulty}")
    print(f"Episodes: {episodes} | Max steps: {max_steps}")
    print(f"horizon(TTC): {horizon}")
    print(f"epsilon: start={eps_start} end={eps_end} decay={eps_decay}")
    print(f"alpha: start={alpha} min={alpha_min} decay={alpha_decay}")
    print(f"gamma: {gamma}")
    print(f"optimistic_init: {optimistic_init}")
    print(f"no_progress_limit: {no_progress_limit}")
    print(f"Run folder: {run_folder}")
    print("=" * 60)

    env = Game(level=difficulty)
    agent = DoubleQAgent(actions=ACTIONS, alpha=alpha, gamma=gamma, epsilon=eps_start, optimistic_init=optimistic_init)

    csv_path = os.path.join(run_folder, "doubleq_training_log.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "score", "ep_reward", "steps", "epsilon", "alpha", "won", "avg_score_100"])

    scores = []
    best_avg = -1e9

    level_best_row = float("inf")
    prev_ep_best_row = float("inf")

    for ep in range(episodes):
        env.reset()

        s = state_to_tabular(env, horizon=horizon)
        ep_reward = 0.0
        steps = 0
        won = 0

        curr_ep_best_row = int(env.player_row)
        steps_since_progress = 0

        done = False
        while not done and steps < max_steps:
            steps += 1

            r_before = int(env.player_row)
            c_before = int(env.player_col)

            a = agent.select_action(s)

            if a != "NOOP":
                env.move_player(ACTIONS_MAP[a])

            env.update()

            if env.won:
                won = 1

            reward, done, level_best_row, prev_ep_best_row, curr_ep_best_row = compute_reward(
                env, r_before, c_before, level_best_row, prev_ep_best_row, curr_ep_best_row
            )
            ep_reward += reward

            if int(env.player_row) == curr_ep_best_row:
                steps_since_progress += 1
            else:
                steps_since_progress = 0

            if (not done) and steps_since_progress >= no_progress_limit:
                reward = float(np.clip(reward - 1.0, -2.0, 2.0))
                done = True

            s_next = state_to_tabular(env, horizon=horizon)
            agent.update(s, a, reward, s_next, done)
            s = s_next

        prev_ep_best_row = curr_ep_best_row
        scores.append(env.score)

        agent.set_epsilon(max(eps_end, agent.epsilon * eps_decay))
        agent.set_alpha(max(alpha_min, agent.alpha * alpha_decay))

        avg_score_100 = float(np.mean(scores[-100:])) if len(scores) >= 100 else float(np.mean(scores))

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, env.score, round(ep_reward, 3), steps, round(agent.epsilon, 4),
                                    round(agent.alpha, 4), won, round(avg_score_100, 3)])

        if ep % 100 == 0:
            print(f"Ep {ep:6d} | Score {env.score:6.1f} | Avg100 {avg_score_100:7.2f} | "
                  f"R {ep_reward:7.2f} | eps {agent.epsilon:.3f} | alpha {agent.alpha:.3f} | won {won}")

            if avg_score_100 > best_avg:
                best_avg = avg_score_100
                best_path = os.path.join(run_folder, "doubleq_best.pkl")
                agent.save(best_path, meta={
                    "difficulty": difficulty,
                    "horizon": horizon,
                    "episode": ep,
                    "best_avg_score_100": best_avg
                })
                print(f"  New BEST avg100={best_avg:.2f} -> saved {best_path}")

        if ep > 0 and ep % checkpoint_interval == 0:
            ckpt_path = os.path.join(run_folder, f"doubleq_checkpoint_{ep}.pkl")
            agent.save(ckpt_path, meta={"difficulty": difficulty, "horizon": horizon, "episode": ep})
            print(f"  Checkpoint -> {ckpt_path}")

    final_path = os.path.join(run_folder, "doubleq_final.pkl")
    agent.save(final_path, meta={"difficulty": difficulty, "horizon": horizon, "episode": episodes - 1, "best_avg100": best_avg})

    print("\nTraining complete.")
    print(f"Run folder: {run_folder}")
    print(f"Best avg100 score: {best_avg:.2f}")
    print(f"Best model: doubleq_best.pkl | Final model: doubleq_final.pkl")
    print(f"CSV log: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TABULAR Double Q-Learning for Crossy Road (PPO-like structure)")
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--difficulty", type=str, default="medium-hard", choices=["easy", "medium", "medium-hard"])
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=800)

    parser.add_argument("--horizon", type=int, default=3, help="TTC prediction horizon (0..3 recommended)")

    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=float, default=0.9995)

    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--alpha-min", type=float, default=0.08)
    parser.add_argument("--alpha-decay", type=float, default=0.9997)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--optimistic-init", type=float, default=0.1)

    parser.add_argument("--no-progress-limit", type=int, default=140)

    args = parser.parse_args()

    train_double_q(
        episodes=args.episodes,
        max_steps=args.max_steps,
        difficulty=args.difficulty,
        checkpoint_interval=args.checkpoint_interval,
        horizon=args.horizon,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        alpha=args.alpha,
        alpha_min=args.alpha_min,
        alpha_decay=args.alpha_decay,
        gamma=args.gamma,
        optimistic_init=args.optimistic_init,
        no_progress_limit=args.no_progress_limit,
    )
