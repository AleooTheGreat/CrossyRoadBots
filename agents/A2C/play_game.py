import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pygame
from environment.game import *
from utilities import Utilities
from agents.A2C.parameters import *
from agents.A2C.easy.a2c import A2C # Import the A2C agent class

if __name__ == "__main__":
    pygame.init()

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Crossy Road Bots")
    clock = pygame.time.Clock()

    LEVEL = 'medium'

    agent = A2C(action_dim=4, radius=4, lr=1e-4, n_steps=20)
    game = Game(level=LEVEL)

    for episode in range(EPISODES):
        game.reset()

        state = Utilities.get_game_state(game, radius=RADIUS)
        done = False
        episode_reward = 0
        counter = 0
        max_reached_row = game.player_row
        score = 0

        rewards, logps, values = [], [], []


        while not done:
            counter += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action, logp, value = agent.select_action(state)

            prev_row = game.player_row
            
            # Take action in the game
            if action == 0:
                game.move_player("up")
            elif action == 1:
                game.move_player("down")
            elif action == 2:
                game.move_player("left")
            elif action == 3:
                game.move_player("right")

            game.update()

            if RENDER:
                game.draw(screen)
                pygame.display.flip()
                clock.tick(60)

            next_state = Utilities.get_game_state(game, radius=RADIUS)

            reward = 0.0

            # Compute reward
            if game.won:
                reward = GAME_WON_REWARD
                done = True
            elif game.game_over:
                reward = GAME_OVER_PENALTY
                done = True
            elif counter >= EPISODE_LENGTH:
                done = True
                reward = EXCEEDED_TIME_PENALTY
            else:
                reward = STEP_PENALTY

                if game.player_row < max_reached_row:
                    reward += ADVANCEMENT_REWARD
                    max_reached_row = game.player_row
                elif game.player_row > prev_row:
                    reward += REGRESSION_PENALTY


            rewards.append(reward)
            logps.append(logp)
            values.append(value)

            episode_reward += reward
            state = next_state

            if len(rewards) >= agent.n_steps or done:
                agent.train_step(rewards, logps, values, next_state, done)
                rewards, logps, values = [], [], []

        score += game.score

        score_history.append(score)
        reward_history.append(episode_reward)

        if len(score_history) >= SAVE_WINDOW:
            avg_score = np.mean(score_history[-SAVE_WINDOW:])
        else:
            avg_score = np.mean(score_history)

        if avg_score > best_avg_score:
            best_avg_score = avg_score
            agent.save(f"a2c_{LEVEL}_best.pth")
            log = f"[SAVE] New best avg score: {best_avg_score:.2f}"
            print(log)
            logs.append(log)

        if int(best_avg_score) >= 490:
            print("Environment solved!")
            break 


        log = f"Episode {episode + 1}/{EPISODES}, Score: {score}, Reward: {episode_reward}, Avg: {avg_score}, Actor loss: {agent.list_actor_losses[-1]:.4f}, Critic loss: {agent.list_critic_losses[-1]:.4f}, Total loss: {agent.list_losses[-1]:.4f}"
        print(log)
        logs.append(log)

    pygame.quit()

    # Save training logs
    with open(f"a2c_{LEVEL}_training_log.txt", "w") as f:
        for log in logs:
            f.write(log + "\n")


    # Execute plotting
    plot_results(score_history, reward_history, agent)
