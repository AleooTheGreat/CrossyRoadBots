import sys
import os
import argparse
import glob
import pickle
import time
import pygame

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from constants import WINDOW_WIDTH, WINDOW_HEIGHT, BLACK, GRID_ROWS, GRID_COLS
from environment.game import Game
from agents.DOUBLE_Q.double_q_agent import DoubleQAgent


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTIONS_MAP = {
    "UP": "up",
    "DOWN": "down",
    "LEFT": "left",
    "RIGHT": "right",
}



def _car_velocity(car) -> float:
    sp = float(getattr(car, "speed", 0.0))
    d = getattr(car, "direction", None)
    if d is None:
        return sp
    d = float(d)
    return sp * d


def _ttc_bin_for_cell(game, target_row, target_col, horizon=3):
    if target_row < 0 or target_row >= GRID_ROWS or target_col < 0 or target_col >= GRID_COLS:
        return 0

    earliest = None
    for car in game.cars:
        if int(car.row) != int(target_row):
            continue

        vel = _car_velocity(car)
        width = float(getattr(car, "width", 1.0))

        for k in range(horizon + 1):
            col_k = float(car.col) + vel * k
            if col_k <= target_col < col_k + width:
                earliest = k if earliest is None else min(earliest, k)
                break

    if earliest is None:
        return 3
    return min(earliest, 3)


def _lane_is_road(game, row):
    return 1 if any(int(c.row) == int(row) for c in game.cars) else 0


def state_to_tabular(game, horizon=3):

    pr, pc = int(game.player_row), int(game.player_col)

    in_up = int(pr - 1 >= 0)
    in_down = int(pr + 1 < GRID_ROWS)
    in_left = int(pc - 1 >= 0)
    in_right = int(pc + 1 < GRID_COLS)

    ttc_up = _ttc_bin_for_cell(game, pr - 1, pc, horizon) if in_up else 0
    ttc_up_l = _ttc_bin_for_cell(game, pr - 1, pc - 1, horizon) if (in_up and in_left) else 0
    ttc_up_r = _ttc_bin_for_cell(game, pr - 1, pc + 1, horizon) if (in_up and in_right) else 0

    ttc_cur = _ttc_bin_for_cell(game, pr, pc, horizon)
    ttc_cur_l = _ttc_bin_for_cell(game, pr, pc - 1, horizon) if in_left else 0
    ttc_cur_r = _ttc_bin_for_cell(game, pr, pc + 1, horizon) if in_right else 0

    ttc_down = _ttc_bin_for_cell(game, pr + 1, pc, horizon) if in_down else 0

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

def find_latest_model(difficulty):
    base = "agents/DOUBLE_Q/checkpoints"
    runs = [d for d in os.listdir(base) if d.startswith("run_") and d.endswith(f"_{difficulty}")]
    if not runs:
        return None

    run = max(runs, key=lambda x: int(x.split("_")[1]))
    folder = os.path.join(base, run)

    for name in ["doubleq_best.pkl", "doubleq_final.pkl"]:
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return p

    ckpts = glob.glob(os.path.join(folder, "doubleq_checkpoint_*.pkl"))
    return max(ckpts, key=lambda p: int(p.split("_")[-1].replace(".pkl", ""))) if ckpts else None


def run_watch(difficulty="medium-hard", model_path=None, horizon=3, fps=60):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"Double Q-Learning TABULAR ({difficulty})")
    clock = pygame.time.Clock()

    if model_path is None:
        model_path = find_latest_model(difficulty)

    if not model_path:
        print("No model found.")
        return

    with open(model_path, "rb") as f:
        ckpt = pickle.load(f)

    agent = DoubleQAgent(actions=ckpt.get("actions", ACTIONS))
    agent.Q1 = ckpt["Q1"]
    agent.Q2 = ckpt["Q2"]
    agent.epsilon = 0.0 

    game = Game(level=difficulty)

    DECISION_DELAY = 0.15   
    last_decision = 0.0

    playing = True
    step_once = False
    running = True

    font = pygame.font.Font(None, 24)

    while running:
        now = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_n:
                    step_once = True
                elif event.key == pygame.K_r:
                    game.reset()
                    playing = True

        if playing and not game.game_over and not game.won:
            if step_once or (now - last_decision >= DECISION_DELAY):
                s = state_to_tabular(game, horizon)

                a = agent.select_action(s)

                if a != "NOOP":
                    game.move_player(ACTIONS_MAP[a])

                game.update()

                last_decision = now
                step_once = False

        game.draw(screen)

        info = font.render(
            "Double Q TABULAR | SPACE pause | N step | R restart",
            True,
            BLACK
        )
        screen.blit(info, (10, WINDOW_HEIGHT - 30))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "medium-hard"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()

    run_watch(
        difficulty=args.difficulty,
        model_path=args.model_path,
        horizon=args.horizon,
        fps=args.fps
    )
