import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import torch
from environment.game import Game
from agents.PPO.PPO import PPO
import matplotlib.pyplot as plt
from constants import GRID_ROWS, GRID_COLS

def get_next_run_folder(difficulty = 'medium'):
    
    checkpoints_dir = 'agents/PPO/checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    existing_runs = [d for d in os.listdir(checkpoints_dir) if d.startswith(f'run_') and d.endswith(f'_{difficulty}')]
    
    if not existing_runs:
        run_num = 1
    else:
        run_nums = [int(d.split('_')[1]) for d in existing_runs]
        run_num = max(run_nums) + 1
    
    run_folder = os.path.join(checkpoints_dir, f'run_{run_num}_{difficulty}')
    os.makedirs(run_folder, exist_ok=True)
    
    return run_folder, run_num

def state_to_vector(game):
    player_pos = [game.player_row / GRID_ROWS, game.player_col / GRID_COLS]
    
    nearby_cars = []
    for row_offset in range(-3, 4):
        for col_offset in range(-3, 4):
            check_row = game.player_row + row_offset
            check_col = game.player_col + col_offset
            
            if 0 <= check_row < GRID_ROWS and 0 <= check_col < GRID_COLS:
                has_car = 0
                for car in game.cars:
                    if car.row == check_row and car.col <= check_col < car.col + car.width:
                        has_car = 1
                        break
                nearby_cars.append(has_car)
            else:
                nearby_cars.append(0)
    
    state = player_pos + nearby_cars
    return np.array(state, dtype=np.float32)

def train_ppo(episodes = 10000, max_steps = 500, update_frequency = 2048, difficulty = 'medium', checkpoint_interval = 1000):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    run_folder, run_num = get_next_run_folder(difficulty)
    
    print(f"\nStarting Run #{run_num} - Difficulty: {difficulty.upper()}")
    print(f"Checkpoints will be saved to: {run_folder}")
    print(f"Checkpoint interval: every {checkpoint_interval} episodes")
    print(f"Training PPO agent for {episodes} episodes\n")

    game = Game(level=difficulty)
    
    state_dim = 51
    action_dim = 4
    
    agent = PPO(state_dim, action_dim, lr = 3e-4, gamma = 0.99, eps_clip = 0.2, K = 4, hidden_dim = 512)
    agent.policy.to(device)
    agent.old_policy.to(device)
    
    scores = []
    avg_scores = []
    best_avg_score = -float('inf')
    
    
    total_steps = 0
    
    for episode in range(episodes):
        
        game.reset()
        
        state = state_to_vector(game)
        state = torch.FloatTensor(state).to(device)
        
        episode_reward = 0
        
        for step in range(max_steps):
            
            action = agent.select_action(state)
            
            actions_map = ['up', 'down', 'left', 'right']
            if action < 4:
                game.move_player(actions_map[action])
            
            game.update()
            
            reward = 0
            if game.game_over:
                reward = -10
            elif game.won:
                reward = 100
            elif game.player_row < game.highest_row:
                reward = 5
            else:
                reward = -0.1
            
            next_state = state_to_vector(game)
            next_state = torch.FloatTensor(next_state).to(device)
            
            done = game.game_over or game.won
            
            agent.store_transition(reward, done)
            
            episode_reward += reward
            total_steps += 1
            
            if total_steps % update_frequency == 0:
                agent.update()
            
            state = next_state
            
            if done:
                break
        
        scores.append(game.score)
        
        if episode % 100 == 0:
            
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            avg_scores.append(avg_score)
            
            print(f"Episode {episode}/{episodes} | Score: {game.score} | Avg: {avg_score:.4f} | Reward: {episode_reward:.4f}")
            
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_path = os.path.join(run_folder, 'ppo_best.pth')
                agent.save(best_path)
                print(f"New best average score: {best_avg_score:.4f} - Model saved!")
            
            if len(avg_scores) >= 10:
                mean_of_last_10_avgs = np.mean(avg_scores[-10:])
                if mean_of_last_10_avgs >= 490:
                    print(f"\n Early stopping")
                    print(f"Mean of last 10 averages: {mean_of_last_10_avgs:.4f}")
                    print(f"Training completed at episode {episode}")
                    break
        
        if episode % checkpoint_interval == 0 and episode > 0:
            checkpoint_path = os.path.join(run_folder, f'ppo_checkpoint_{episode}.pth')
            agent.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    final_path = os.path.join(run_folder, 'ppo_final.pth')
    agent.save(final_path)
    
    print(f"\nTraining done! Models saved to {run_folder}")
    print(f"Best average score achieved: {best_avg_score:.4f}")
    print(f"Best model: ppo_best.pth | Final model: ppo_final.pth")
    
    plot_path = os.path.join(run_folder, 'training_progress.png')
    
    plt.figure(figsize=(10, 5))
    plt.plot(scores, alpha=0.3, label='Score per episode')
    
    if len(avg_scores) > 0:
        x_avg = [i * 100 for i in range(len(avg_scores))]
        plt.plot(x_avg, avg_scores, linewidth=2, label='Average (100 episodes)')
    
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f'PPO Training Progress - Run {run_num} ({difficulty.capitalize()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_path)
    
    print(f"Training plot saved to {plot_path}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train PPO agent for Crossy Road')
    
    parser.add_argument('--episodes', type=int, default=100000,
                        help='Number of training episodes (default: 100000)')
    
    parser.add_argument('--difficulty', type=str, default='medium', 
                        choices=['easy', 'medium', 'medium-hard'],
                        help='Game difficulty level (default: medium)')
    
    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                        help='Episodes between checkpoint saves (default: 1000)')
    
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per episode (default: 500)')
    
    parser.add_argument('--update-frequency', type=int, default=2048,
                        help='Steps between policy updates (default: 2048)')
    
    args = parser.parse_args()
    
    train_ppo(
        episodes=args.episodes,
        max_steps=args.max_steps,
        update_frequency=args.update_frequency,
        difficulty=args.difficulty,
        checkpoint_interval=args.checkpoint_interval
    )
