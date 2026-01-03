import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import pygame
import numpy as np
import torch.nn.functional as F

from environment.game import *

from collections import deque
from actor import Actor
from critic import Critic
from environment.game import Game
from utilities import Utilities
from constants import WINDOW_WIDTH, WINDOW_HEIGHT

class A2C(torch.nn.Module):
    def __init__(self):
        super(A2C, self).__init__()

        self.RADIUS = 5
        self.input_dim = (2 * self.RADIUS + 1) ** 2

        self.actor = Actor(input_dim=self.input_dim, action_dim=5)
        self.critic = Critic(input_dim=self.input_dim)

        self.gamma = 0.99
        self.logs = []
        

    def forward(self, x: torch.Tensor):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value


    def train(self, episodes: int = 1000, level: str = 'easy', render: bool = False):
        pygame.init()
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("A2C Agent Training")
        clock = pygame.time.Clock()

        game = Game(level=level)
        print(f"Training A2C agent on {level} level for {episodes} episodes.")

        N_STEPS = 20 # Mărit puțin pentru a prinde secvențe mai lungi
        ENTROPY_BETA = 0.01 # Redus drastic! 0.1 e prea haotic, 0.01 îl lasă să învețe
        MAX_STEPS = 500 # Mărit ca să aibă timp să traverseze

        last_scores = deque(maxlen=100) # Media pe 100 de episoade e mai relevantă
        best_avg_score = -float('inf')
        
        for episode in range(episodes):
            game.reset()

            state = Utilities.get_game_state(game, radius=self.RADIUS)
            state = torch.FloatTensor(state).unsqueeze(0)

            episode_reward = 0
            done = False

            values = []
            log_probs = []
            rewards = []
            masks = []
            entropies = []

            step_count = 0
            
            # Ținem minte cel mai bun rând atins LOCAL în episod pentru recompensă
            current_episode_highest_row = game.player_row 

            while not done:
                step_count += 1

                action, log_prob, entropy = self.actor.select_action(state, train=True)
                entropies.append(entropy)
                
                prev_row = game.player_row

                if action == 0: game.move_player("up")
                elif action == 1: game.move_player("down")
                elif action == 2: game.move_player("left")
                elif action == 3: game.move_player("right")
                elif action == 4: pass

                game.update()

                reward = -0.05 # Penalizare mică de existență

                if game.won:
                    reward = 50.0
                    done = True
                elif game.game_over:
                    reward = -10.0
                    done = True
                elif step_count >= MAX_STEPS:
                    done = True
                else:
                    # RECOMPENSA CRITICĂ PENTRU PROGRES REAL
                    if game.player_row < current_episode_highest_row:
                        reward += 5.0 # BRAVO! Ai avansat.
                        current_episode_highest_row = game.player_row
                    elif game.player_row > prev_row:
                        reward -= 0.5 # Nu te întoarce
                    
                    # Penalizare dacă stă în zona periculoasă (nu e safe zone)
                    is_safe = game.player_row % game.level_config.safe_zone_spacing == 0 or game.player_row >= GRID_ROWS - 3
                    if not is_safe and action == 4:
                         reward -= 0.1 # Nu sta pe stradă!

                episode_reward += reward

                next_state = Utilities.get_game_state(game, radius=self.RADIUS)
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                state_value = self.critic(state)

                log_probs.append(log_prob)
                values.append(state_value)
                rewards.append(torch.FloatTensor([reward]))
                masks.append(torch.FloatTensor([1 - int(done)]))

                if len(values) == N_STEPS or done:
                    if done:
                        R = torch.tensor([0.0])
                    else:
                        R = self.critic(next_state).detach()

                    actor_loss = 0
                    critic_loss = 0
                    
                    for step in reversed(range(len(rewards))):
                        R = rewards[step] + self.gamma * R * masks[step]
                        advantage = R - values[step].detach()
                        critic_loss += F.mse_loss(values[step], R)
                        actor_loss += -log_probs[step] * advantage - (ENTROPY_BETA * entropies[step])

                    optimizer_steps = len(rewards)
                    
                    self.actor.optimizer.zero_grad()
                    (actor_loss / optimizer_steps).backward()
                    self.actor.optimizer.step() # CLIP GRADIENT ar ajuta aici, dar lăsăm simplu

                    self.critic.optimizer.zero_grad()
                    (critic_loss / optimizer_steps).backward()
                    self.critic.optimizer.step()

                    values = []
                    log_probs = []
                    rewards = []
                    masks = []
                    entropies = []

                state = next_state

                if render:
                    game.draw(screen)
                    pygame.display.flip()
                    clock.tick(60)

            last_scores.append(game.score)
            
            # Logare la fiecare 10 episoade sau dacă e scor bun
            if episode % 10 == 0 or game.score > 50:
                avg = np.mean(last_scores)
                log = f"Episode {episode + 1}/{episodes} - Reward: {episode_reward:.2f} | Score: {game.score} | Avg: {avg:.1f}"
                print(log)
                self.logs.append(log)

            if len(last_scores) == last_scores.maxlen and np.mean(last_scores) >= best_avg_score:
                best_avg_score = np.mean(last_scores)
                self.save(f'a2c_best_{level}.pth')
                print(f"New best model saved: a2c_best_{level}.pth")

        pygame.quit()


    def save(self, filepath: str):
        torch.save(self.state_dict(), filepath) 
        with open(filepath + '_logs.txt', 'w') as f:
            for log in self.logs:
                f.write(log + '\n')

    def load(self, filepath: str):
        self.load_state_dict(torch.load(filepath))


if __name__ == "__main__":
    agent = A2C()

    # Train for easy level
    agent.train(episodes=2500, level='easy', render=False)
    agent.save('a2c_easy.pth')
    print("Saved A2C agent trained on easy level.")

    # # Train for medium level
    # agent.train(episodes=1200, level='medium', render=True)
    # agent.save('a2c_medium.pth')
    # print("Saved A2C agent trained on medium level.")
    
    # # Train for hard level
    # agent.train(episodes=1400, level='medium-hard', render=True)
    # agent.save('a2c_hard.pth')
    # print("Saved A2C agent trained on hard level.")
