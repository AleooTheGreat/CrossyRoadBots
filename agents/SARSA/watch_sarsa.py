import pickle
import pygame
from constants import WINDOW_WIDTH, WINDOW_HEIGHT
from environment.game import Game
from agents.SARSA.sarsa_agent import SARSAAgent

# IMPORTANT: scoatem NOOP ca să nu stea pe loc
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

CHECKPOINT = "agents/SARSA/checkpoints/sarsa_ep_4800.pkl"


def get_state(game, prev_car_cols):
    pr, pc = game.player_row, game.player_col

    lane_type = "ROAD" if any(v.row == pr for v in game.cars) else "SAFE"

    car_ahead = car_left = car_right = 0
    car_ahead_moving = car_left_moving = car_right_moving = 0

    for v in game.cars:
        r = v.row
        c = int(v.col)
        prev_c = prev_car_cols.get(id(v), c)
        dc = c - prev_c

        if r == pr - 1 and c == pc:
            car_ahead = 1
            if dc != 0:
                car_ahead_moving = 1

        if r == pr and c == pc - 1:
            car_left = 1
            if dc != 0:
                car_left_moving = 1

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


def main():
    pygame.init()

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("SARSA WATCH")

    clock = pygame.time.Clock()

    game = Game(level="medium")

    # epsilon mic doar pentru vizualizare (altfel ar sta blocat)
    agent = SARSAAgent(actions=ACTIONS, epsilon=0.05)

    # load Q-table
    with open(CHECKPOINT, "rb") as f:
        agent.Q = pickle.load(f)

    # eliminăm NOOP din Q-table dacă există
    for s in agent.Q:
        agent.Q[s].pop("NOOP", None)

    game.reset()
    prev_car_cols = {id(v): int(v.col) for v in game.cars}

    running = True
    while running:
        clock.tick(10)  # încetinește jocul

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state = get_state(game, prev_car_cols)

        # fallback dacă starea nu există
        if state not in agent.Q:
            action = "UP"
        else:
            action = agent.select_action(state)

        # EXECUTĂ ACȚIUNEA (asta lipsea înainte)
        game.move_player(action.lower())

        game.update()

        game.draw(screen)
        pygame.display.flip()

        if game.game_over or game.won:
            pygame.time.delay(2000)
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
