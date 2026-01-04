import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import pygame
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from agents.A2C.policy import PolicyNet
from agents.A2C.value import ValueNet
from environment.game import *
from base_agent import BaseAgent
from utilities import Utilities
from agents.A2C.parameters import *

class A2C(BaseAgent):
    def __init__(
            self, 
            n_steps: int = 20,
            lr: float = 3e-4,
            gamma: float = 0.99,
            radius: int = 3,
            action_dim: int = 5
    ):
        self.input_dim = (2 * radius + 1) ** 2
        super(A2C, self).__init__(agent_name="A2C", state_size=self.input_dim, action_size=action_dim)

        self.gamma = gamma
        self.n_steps = n_steps

        self.policy = PolicyNet(self.input_dim, action_dim)
        self.value = ValueNet(self.input_dim)

        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), 
            lr=lr
        )

        self.list_actor_losses = []
        self.list_critic_losses = []
        self.list_losses = []
        
    # Override
    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        logits = self.policy(state_tensor)
        dist = torch.distributions.Categorical(logits=logits)

        action = dist.sample()
        logp = dist.log_prob(action)

        value = self.value(state_tensor).squeeze(-1)  # FIX SHAPE
        entropy = dist.entropy()

        return action.item(), logp, value, entropy

    

    # Override
    def train_step(self, rewards, logps, values, entropies, next_state, masks):
        values = torch.stack(values)
        logps = torch.stack(logps)
        entropies = torch.stack(entropies)


        with torch.no_grad():
            next_value = self.value(
                torch.tensor(next_state, dtype=torch.float32)
            ).squeeze(-1).detach()



        # 1. Compute n-step return
        returns = torch.zeros_like(values)
        G = next_value

        for step in reversed(range(len(rewards))):
            G = rewards[step] + self.gamma * G * masks[step]
            returns[step] = G

        advantages = returns.detach() - values


        # 4. Loss-uri
        ENTROPY_BETA = 0.05
        actor_loss = -(logps * advantages.detach() + ENTROPY_BETA * entropies).mean()
        self.list_actor_losses.append(actor_loss.item())
        
        # Critic: Folosește Mean Squared Error pe valorile reale (Returns vs Values)
        # NU folosi varianta normalizată aici!
        critic_loss = 0.5 * advantages.pow(2).mean()
        self.list_critic_losses.append(critic_loss.item())

        # Total loss
        loss = actor_loss + critic_loss
        self.list_losses.append(loss.item())


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    # Override
    def save(self, filepath: str):
        checkpoint = {
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict()
        }
        torch.save(checkpoint, filepath)

    # Override
    def load(self, filepath: str):
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])

def plot_results(score_history, reward_history, agent):
    # Parameters for smoothing
    window_size = 50  
    
    # Helper to smooth data
    def smooth(data):
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    # Data Preparation
    actor_losses = agent.list_actor_losses
    critic_losses = agent.list_critic_losses
    total_losses = agent.list_losses
    
    # 1. General Loss
    plt.figure(figsize=(10, 5))
    plt.plot(total_losses, color='gray', alpha=0.3, label='Raw Loss')
    plt.plot(smooth(total_losses), color='red', linewidth=2, label=f'Avg Loss (Window={window_size})')
    plt.title('1. General Loss Evolution')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('1_general_loss.png')
    plt.close()

    # 2. Actor vs Critic Loss
    plt.figure(figsize=(10, 5))
    plt.plot(smooth(actor_losses), label='Actor Loss', color='blue', linewidth=1.5)
    plt.plot(smooth(critic_losses), label='Critic Loss', color='orange', linewidth=1.5)
    plt.title('2. Actor and Critic Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('2_actor_critic_loss.png')
    plt.close()

    # 3. Score Evolution
    plt.figure(figsize=(10, 5))
    plt.plot(score_history, color='lightblue', alpha=0.4, label='Raw Score')
    plt.plot(smooth(score_history), color='blue', linewidth=2, label='Avg Score')
    plt.title('3. Score Evolution')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('3_score_evolution.png')
    plt.close()

    # 4. Reward Evolution
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, color='lightgreen', alpha=0.4, label='Raw Reward')
    plt.plot(smooth(reward_history), color='green', linewidth=2, label='Avg Reward')
    plt.title('4. Reward Evolution')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('4_reward_evolution.png')
    plt.close()

    # 5. Score (Foreground) vs Reward (Background)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Score on Left Axis
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Score', color='blue')
    ax1.plot(smooth(score_history), color='blue', linewidth=2, label='Score')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Reward on Right Axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Reward', color='gray')
    # Use fill_between for background effect
    smoothed_rewards = smooth(reward_history)
    x_axis = range(len(smoothed_rewards))
    ax2.fill_between(x_axis, smoothed_rewards, color='gray', alpha=0.2, label='Reward')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    plt.title('5. Score vs Reward')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('5_score_vs_reward.png')
    plt.close()
    
    print("All plots saved successfully.")


if __name__ == "__main__":
    pygame.init()

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("A2C Agent Training")
    clock = pygame.time.Clock()

    RENDER = False
    EPISODES = 2000
    LEVEL = 'easy'
    EPISODE_LENGTH = 1000
    RADIUS = 4

    best_avg_score = -float("inf")
    SAVE_WINDOW = 20
    
    reward_history = []
    score_history = []
    logs = []

    agent = A2C(action_dim=4, radius=RADIUS, lr=1e-4)
    game = Game(level=LEVEL)

    for episode in range(EPISODES):
        game.reset()

        state = Utilities.get_game_state(game, radius=RADIUS)
        done = False
        episode_reward = 0
        counter = 0
        max_reached_row = game.player_row
        score = 0

        rewards, logps, values, entropies, masks = [], [], [], [], []


        while not done:
            counter += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action, logp, value, entropy = agent.select_action(state)

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
                    reward -= REGRESSION_PENALTY


            rewards.append(reward)
            logps.append(logp)
            values.append(value)
            entropies.append(entropy)
            masks.append(1.0 - float(done))

            episode_reward += reward
            state = next_state

            if len(rewards) >= agent.n_steps or done:
                agent.train_step(rewards, logps, values, entropies, next_state, masks)
                rewards, logps, values, entropies, masks = [], [], [], [], []

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
            print(f"[SAVE] New best avg score: {best_avg_score:.2f}")

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
