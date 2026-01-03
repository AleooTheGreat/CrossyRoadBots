import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import pygame
import torch.nn.functional as F

from actor import Actor
from critic import Critic
from environment.game import Game
from utilities import Utilities
from constants import WINDOW_WIDTH, WINDOW_HEIGHT

class A2C(torch.nn.Module):
    def __init__(self):
        super(A2C, self).__init__()

        self.actor = Actor(input_dim=49, action_dim=5)
        self.critic = Critic(input_dim=49)

        self.gamma = 0.99
        

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

        MAX_STEPS = 200

        for episode in range(episodes):
            print(f"Starting episode {episode+1}/{episodes}")
            game.reset()

            # initial state
            game_state = Utilities.get_game_state(game)
            t_game_state = torch.tensor(game_state, dtype=torch.float32).unsqueeze(0)

            max_reached_row = game.player_row
            total_reward = 0
            done = False

            step = 0

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                action, log_prob = self.actor.select_action(t_game_state)

                prev_row = game.player_row
                step += 1

                if action == 0:
                    game.move_player("up")
                elif action == 1:
                    game.move_player("down")
                elif action == 2:
                    game.move_player("left")
                elif action == 3:
                    game.move_player("right")
                elif action == 4:
                    pass  # do nothing


                game.update()

                reward = 0
        
                if game.won:
                    reward = 100
                    done = True
                elif step >= MAX_STEPS:
                    done = True
                    reward = -10
                elif game.game_over:
                    reward = -10
                    done = True
                else:
                    reward = -0.3

                    if game.player_row < max_reached_row: # player advanced
                        reward += 1.0
                        max_reached_row = game.player_row
                    elif game.player_row > prev_row: # player moved back
                        reward -= 1.0

                total_reward += reward
                
                game_next_state = Utilities.get_game_state(game)
                t_game_next_state = torch.tensor(game_next_state, dtype=torch.float32).unsqueeze(0)


                t_reward = torch.tensor([reward], dtype=torch.float32)
                t_done = torch.tensor([done], dtype=torch.float32)


                value = self.critic(t_game_state)
                next_value = self.critic(t_game_next_state)

                target = t_reward + self.gamma * next_value.detach() * (1 - t_done)

                critic_loss = F.mse_loss(value, target)

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()


                advantage = (target - value).detach()

                probs = self.actor(t_game_state)
                dist = torch.distributions.Categorical(probs)
                entropy = dist.entropy()

                actor_loss = -log_prob * advantage - 0.01 * entropy

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                t_game_state = t_game_next_state

                if render:
                    game.draw(screen)
                    pygame.display.flip()
                    clock.tick(60)

            print(f"Episode {episode+1} finished with total reward: {int(total_reward)}, total score: {game.score}")


    def save(self, filepath: str):
        torch.save(self.state_dict(), filepath) 

    def load(self, filepath: str):
        self.load_state_dict(torch.load(filepath))


if __name__ == "__main__":
    agent = A2C()

    # Train for easy level
    agent.train(episodes=1000, level='easy', render=True)
    agent.save('a2c_easy.pth')
    print("Saved A2C agent trained on easy level.")

    # Train for medium level
    agent.train(episodes=1200, level='medium', render=True)
    agent.save('a2c_medium.pth')
    print("Saved A2C agent trained on medium level.")
    
    # Train for hard level
    agent.train(episodes=1400, level='medium-hard', render=True)
    agent.save('a2c_hard.pth')
    print("Saved A2C agent trained on hard level.")
