import pygame
import sys
import os
import argparse
import glob
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from constants import *
from environment.game import Game
from agents.PPO.PPO import PPO
import numpy as np

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

def find_latest_model(difficulty = 'medium'):
    
    checkpoints_dir = 'agents/PPO/checkpoints'
    
    if not os.path.exists(checkpoints_dir):
        return None
    
    runs = [d for d in os.listdir(checkpoints_dir) if d.startswith('run_') and d.endswith(f'_{difficulty}')]
    
    if not runs:
        return None
    
    run_nums = [int(d.split('_')[1]) for d in runs]
    latest_run = f'run_{max(run_nums)}_{difficulty}'
    
    model_path = os.path.join(checkpoints_dir, latest_run, 'ppo_best.pth')
    if os.path.exists(model_path):
        return model_path
    
    model_path = os.path.join(checkpoints_dir, latest_run, 'ppo_final.pth')
    if os.path.exists(model_path):
        return model_path
    
    checkpoint_pattern = os.path.join(checkpoints_dir, latest_run, 'ppo_checkpoint_*.pth')
    checkpoints = glob.glob(checkpoint_pattern)
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].replace('.pth', '')))
        return latest_checkpoint
    
    return None

def run_watch(difficulty = 'medium', model_path = None):
    
    pygame.init()

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"Crossy Road Bots - PPO Agent ({difficulty.capitalize()})")

    clock = pygame.time.Clock()

    state_dim = 51
    action_dim = 4
    agent = PPO(state_dim, action_dim, hidden_dim=512)

    if model_path:
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return
        
        final_model_path = model_path
    else:
        final_model_path = find_latest_model(difficulty)
    
    if final_model_path:
        
        agent.load(final_model_path)
        agent.policy.to(agent.device)
        agent.old_policy.to(agent.device)
        
        print(f"Loaded trained PPO agent from: {final_model_path}")
        print(f"Difficulty: {difficulty.capitalize()}")
        print(f"Running on: {agent.device}")
    else:
        print(f"No trained model found for difficulty '{difficulty}'. Please train first with:")
        print(f"  python train_ppo.py --difficulty {difficulty}")
        return

    game = Game(level=difficulty)
    running = True
    playing = True

    while running:
        
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset()
                    playing = True
                elif event.key == pygame.K_SPACE:
                    playing = not playing
        
        if playing and not game.game_over and not game.won:
            
            state = state_to_vector(game)
            action = agent.select_action(state)
            
            actions_map = ['up', 'down', 'left', 'right', 'stay']
            if action < 4:
                game.move_player(actions_map[action])
            
            game.update()
        
        game.draw(screen)
        
        font = pygame.font.Font(None, 24)
        info_text = font.render(f'PPO Agent ({difficulty.upper()}) | Press SPACE to pause | Press R to restart', True, BLACK)
        screen.blit(info_text, (10, WINDOW_HEIGHT - 30))
        
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Watch trained PPO agent play Crossy Road')
    
    parser.add_argument('--difficulty', type=str, default='medium',
                        choices=['easy', 'medium', 'medium-hard'],
                        help='Game difficulty level (default: medium)')
    
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to specific model file (optional, defaults to latest for difficulty)')
    
    args = parser.parse_args()
    
    run_watch(difficulty=args.difficulty, model_path=args.model_path)
