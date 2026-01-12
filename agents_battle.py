import pygame
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from constants import *
from environment.game import Game

from agents.A2C.a2c import A2C
from agents.A2C.utilities import Utilities
from agents.PPO.PPO import PPO
from agents.rainbowDQN.rainbow_agent import RainbowDQNAgent
from agents.rainbowDQN.utils import get_state
from agents.DOUBLE_Q.double_q_agent import DoubleQAgent


class AgentsBattle:
    
    def __init__(self, difficulty = 'medium', game_mode = 'statistical'):
        self.difficulty = difficulty
        self.game_mode = game_mode
        
        infinite_mode = (game_mode == 'survival')
        self.games = {
            'A2C': Game(level = difficulty, infinite_mode = infinite_mode),
            'PPO': Game(level = difficulty, infinite_mode = infinite_mode),
            'Rainbow': Game(level = difficulty, infinite_mode = infinite_mode),
            'DoubleQ': Game(level = difficulty, infinite_mode = infinite_mode)
        }
        
        self.agents = {}
        self._load_agents()
        
        self.scores = {
            'A2C': {'wins': 0, 'max_progress': 0, 'deaths': 0, 'alive': True, 'survival_time': 0},
            'PPO': {'wins': 0, 'max_progress': 0, 'deaths': 0, 'alive': True, 'survival_time': 0},
            'Rainbow': {'wins': 0, 'max_progress': 0, 'deaths': 0, 'alive': True, 'survival_time': 0},
            'DoubleQ': {'wins': 0, 'max_progress': 0, 'deaths': 0, 'alive': True, 'survival_time': 0}
        }
        
        self.frame_count = 0
        self.winner = None
        
        self.agent_colors = {
            'A2C': (100, 150, 200),
            'PPO': (200, 100, 100),
            'Rainbow': (160, 120, 180),
            'DoubleQ': (100, 200, 100)
        }
        
    def _load_agents(self):
        
        print("Loading A2C")
        a2c_checkpoint_paths = {
            "easy": "agents/A2C/easy/a2c_easy_best.pth",
            "medium": "agents/A2C/medium/a2c_medium_best.pth",
            "medium-hard": "agents/A2C/medium-hard/a2c_medium-hard_best.pth"
        }
        self.agents['A2C'] = A2C()
        self.agents['A2C'].load(filepath = a2c_checkpoint_paths[self.difficulty])
        print(f"A2C loaded from {a2c_checkpoint_paths[self.difficulty]}")
        
        print("Loading PPO")
        ppo_checkpoint_path = f"agents/PPO/checkpoints/run_1_{self.difficulty}/ppo_best.pth"
        state_dimension = 51
        action_dimension = 4
        self.agents['PPO'] = PPO(state_dimension, action_dimension, hidden_dim = 512)
        self.agents['PPO'].load(ppo_checkpoint_path)
        self.agents['PPO'].policy.to(self.agents['PPO'].device)
        self.agents['PPO'].old_policy.to(self.agents['PPO'].device)
        print(f"PPO loaded from {ppo_checkpoint_path}")
        
        print("Loading RainbowDQN")
        view_range = 5
        state_dimension = (view_range * 2 + 1) ** 2
        self.agents['Rainbow'] = RainbowDQNAgent(
            agent_name = "RainbowBot",
            state_size = state_dimension,
            action_size = 4
        )
        rainbow_checkpoint_path = f"agents/rainbowDQN/checkpoints/RainbowBot/rainbow_cleared_{self.difficulty}.pth"
        if not os.path.exists(rainbow_checkpoint_path):
            rainbow_checkpoint_path = "agents/rainbowDQN/checkpoints/RainbowBot/rainbow_best_agent.pth"
        self.agents['Rainbow'].load(rainbow_checkpoint_path)
        print(f"Rainbow loaded from {rainbow_checkpoint_path}")
        
        print("Loading DoubleQ")
        doubleq_checkpoint_path = f"agents/DOUBLE_Q/checkpoints/run_1_{self.difficulty}/doubleq_best.pkl"
        if not os.path.exists(doubleq_checkpoint_path):
            doubleq_checkpoint_path = "agents/DOUBLE_Q/checkpoints/doubleq_best.pkl"
        actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.agents['DoubleQ'] = DoubleQAgent(actions=actions, alpha=0.25, gamma=0.99, epsilon=0.0)
        self.agents['DoubleQ'].load(doubleq_checkpoint_path)
        print(f"DoubleQ loaded from {doubleq_checkpoint_path}")
    
    def get_a2c_action(self, game):
        
        state = Utilities.get_game_state(game)
        action = self.agents['A2C'].select_action(state)[0]
        return action
    
    def get_ppo_action(self, game):
        
        player_position = [game.player_row / GRID_ROWS, game.player_col / GRID_COLS]
        nearby_cars_list = []
        
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
                    nearby_cars_list.append(has_car)
                else:
                    nearby_cars_list.append(0)
        
        state = player_position + nearby_cars_list
        state = np.array(state, dtype = np.float32)
        
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.agents['PPO'].device)
            action, _, _ = self.agents['PPO'].old_policy.act(state)
        
        return action
    
    def get_rainbow_action(self, game):
        
        state = get_state(game, view_range = 5)
        action = self.agents['Rainbow'].select_action(state, training = False)
        return action
    
    def get_doubleq_action(self, game):
        
        state = self._state_to_tabular(game)
        action_str = self.agents['DoubleQ'].select_action(state)
        action_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
        return action_map[action_str]
    
    def _state_to_tabular(self, game, horizon=3):

        pr, pc = int(game.player_row), int(game.player_col)

        in_up = int(pr - 1 >= 0)
        in_down = int(pr + 1 < GRID_ROWS)
        in_left = int(pc - 1 >= 0)
        in_right = int(pc + 1 < GRID_COLS)

        ttc_up = self._ttc_bin_for_cell(game, pr - 1, pc, horizon) if in_up else 0
        ttc_up_l = self._ttc_bin_for_cell(game, pr - 1, pc - 1, horizon) if (in_up and in_left) else 0
        ttc_up_r = self._ttc_bin_for_cell(game, pr - 1, pc + 1, horizon) if (in_up and in_right) else 0

        ttc_cur = self._ttc_bin_for_cell(game, pr, pc, horizon)
        ttc_cur_l = self._ttc_bin_for_cell(game, pr, pc - 1, horizon) if in_left else 0
        ttc_cur_r = self._ttc_bin_for_cell(game, pr, pc + 1, horizon) if in_right else 0

        ttc_down = self._ttc_bin_for_cell(game, pr + 1, pc, horizon) if in_down else 0

        lane_cur = self._lane_is_road(game, pr)
        lane_up = self._lane_is_road(game, pr - 1) if in_up else 0

        return (
            pc,
            lane_cur, lane_up,
            in_up, in_down, in_left, in_right,
            ttc_up, ttc_up_l, ttc_up_r,
            ttc_cur, ttc_cur_l, ttc_cur_r,
            ttc_down,
        )

    def _ttc_bin_for_cell(self, game, target_row, target_col, horizon=3):
        
        if target_row < 0 or target_row >= GRID_ROWS or target_col < 0 or target_col >= GRID_COLS:
            return 0
        earliest = None
        for car in game.cars:
            if int(car.row) != int(target_row):
                continue
            
            vel = self._car_velocity(car)
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
    
    def _car_velocity(self, car):
        
        sp = float(getattr(car, "speed", 0.0))
        d = getattr(car, "direction", None)
        if d is None:
            return sp
        d = float(d)
        if sp < 0:
            return sp
        return sp * d
    
    def _lane_is_road(self, game, row):
        
        return 1 if any(int(c.row) == int(row) for c in game.cars) else 0
    
    def execute_action(self, game, action, agent_name):
        
        action_directions = ['up', 'down', 'left', 'right']
        
        if action < len(action_directions) and action_directions[action] is not None:
            game.move_player(action_directions[action])
    
    def update_scores(self):
        
        for agent_name, game in self.games.items():
            
            if game.player_row > self.scores[agent_name]['max_progress']:
                self.scores[agent_name]['max_progress'] = game.player_row
            
            if game.won and not hasattr(game, '_win_counted'):
                self.scores[agent_name]['wins'] += 1
                game._win_counted = True
                
                if self.game_mode == 'survival':
                    self.scores[agent_name]['survival_time'] = self.frame_count
            
            if game.game_over and not hasattr(game, '_death_counted'):
                self.scores[agent_name]['deaths'] += 1
                game._death_counted = True
                
                if self.game_mode == 'survival':
                    self.scores[agent_name]['alive'] = False
                    self.scores[agent_name]['survival_time'] = self.frame_count
    
    def reset_if_needed(self):
        
        if self.game_mode == 'statistical':
            
            for agent_name, game in self.games.items():
                if game.game_over or game.won:
                    game.reset()
                    
                    if hasattr(game, '_win_counted'):
                        delattr(game, '_win_counted')
                    if hasattr(game, '_death_counted'):
                        delattr(game, '_death_counted')
        else:
            
            alive_agents = [agent_name for agent_name, agent_stats in self.scores.items() if agent_stats['alive']]
            
            if len(alive_agents) == 1 and not self.winner:
                self.winner = alive_agents[0]
                print(f"\n{self.winner} WINS THE SURVIVAL BATTLE")
                print(f"Survived for {self.scores[self.winner]['survival_time']} frames")
            elif len(alive_agents) == 0 and not self.winner:
                
                best_survival_time = max(self.scores[agent_name]['survival_time'] for agent_name in self.scores)
                winner_agents = [agent_name for agent_name, agent_stats in self.scores.items() if agent_stats['survival_time'] == best_survival_time]
                if len(winner_agents) == 1:
                    self.winner = winner_agents[0]
                    print(f"\n{self.winner} SURVIVED THE LONGEST")
                else:
                    self.winner = "TIE"
                    print(f"\nTIE - Multiple agents survived {best_survival_time} frames")
    
    def draw_scoreboard(self, screen, game_width):
        
        pygame.font.init()
        main_font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 18)
        
        scoreboard_height = 80
        y_position = WINDOW_HEIGHT
        
        pygame.draw.rect(screen, (30, 30, 30), (0, y_position, game_width * 4, scoreboard_height))
        
        x_positions = [game_width // 2, game_width + game_width // 2, 2 * game_width + game_width // 2, 3 * game_width + game_width // 2]
        
        for agent_index, (agent_name, agent_color) in enumerate(self.agent_colors.items()):
            x_center = x_positions[agent_index]
            
            name_text = main_font.render(agent_name, True, agent_color)
            name_rect = name_text.get_rect(center = (x_center, y_position + 15))
            screen.blit(name_text, name_rect)
            
            if self.game_mode == 'statistical':
                wins_count = self.scores[agent_name]['wins']
                deaths_count = self.scores[agent_name]['deaths']
                max_progress = self.scores[agent_name]['max_progress']
                
                stats_text = small_font.render(f"Wins: {wins_count} | Deaths: {deaths_count} | Best: {max_progress}", True, (200, 200, 200))
            else:
                survival_time = self.scores[agent_name]['survival_time']
                max_progress = self.scores[agent_name]['max_progress']
                is_alive = self.scores[agent_name]['alive']
                
                alive_status = "ALIVE" if is_alive else "DEAD"
                stats_text = small_font.render(f"{alive_status} | Frames: {survival_time} | Best: {max_progress}", True, (200, 200, 200))
            
            stats_rect = stats_text.get_rect(center = (x_center, y_position + 40))
            screen.blit(stats_text, stats_rect)
            
            current_game = self.games[agent_name]
            
            if self.game_mode == 'survival' and not self.scores[agent_name]['alive']:
                status_message = "ELIMINATED"
                status_color = (100, 100, 100)
            elif current_game.won:
                status_message = "WON"
                status_color = (100, 255, 100)
            elif current_game.game_over:
                status_message = "DEAD"
                status_color = (255, 100, 100)
            else:
                status_message = f"Row {current_game.player_row}"
                status_color = (200, 200, 200)
            
            status_text = small_font.render(status_message, True, status_color)
            status_rect = status_text.get_rect(center = (x_center, y_position + 60))
            screen.blit(status_text, status_rect)
    
    def draw_game_with_overlay(self, screen, game, x_offset, agent_name, agent_color):
        
        game_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        
        game.draw(game_surface)
        
        border_width = 3
        pygame.draw.rect(game_surface, agent_color, (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT), border_width * 2)
        
        pygame.font.init()
        label_font = pygame.font.Font(None, 32)
        label_text = label_font.render(agent_name, True, agent_color)
        label_background = pygame.Surface((label_text.get_width() + 20, label_text.get_height() + 10))
        label_background.fill((0, 0, 0))
        label_background.set_alpha(180)
        game_surface.blit(label_background, (WINDOW_WIDTH // 2 - label_text.get_width() // 2 - 10, 10))
        game_surface.blit(label_text, (WINDOW_WIDTH // 2 - label_text.get_width() // 2, 15))
        
        screen.blit(game_surface, (x_offset, 0))
    
    def run(self):
        
        pygame.init()
        
        screen_width = WINDOW_WIDTH * 4
        screen_height = WINDOW_HEIGHT + 80
        screen = pygame.display.set_mode((screen_width, screen_height))
        
        mode_display_name = "STATISTICAL" if self.game_mode == 'statistical' else "SURVIVAL"
        pygame.display.set_caption(f"Agents Battle 1v1v1v1 - {self.difficulty.upper()} - {mode_display_name}")
        
        game_clock = pygame.time.Clock()
        
        print(f"\n{'=' * 60}")
        print(f"AGENTS BATTLE - {self.difficulty.upper()} DIFFICULTY")
        print(f"MODE: {mode_display_name}")
        print(f"{'=' * 60}")
        print("Press R to reset all games")
        print("Press M to switch mode (Statistical/Survival)")
        print("Press 1/2/3 to change difficulty (easy/medium/medium-hard)")
        print("Press ESC or close window to quit")
        print(f"{'=' * 60}\n")
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        
                        self.__init__(self.difficulty, self.game_mode)
                        print("All games reset")
                    elif event.key == pygame.K_m:
                        
                        new_game_mode = 'survival' if self.game_mode == 'statistical' else 'statistical'
                        self.__init__(self.difficulty, new_game_mode)
                        mode_display_name = "STATISTICAL" if new_game_mode == 'statistical' else "SURVIVAL"
                        print(f"Switched to {mode_display_name} mode")
                    elif event.key == pygame.K_1:
                        self.__init__('easy', self.game_mode)
                        print("Switched to EASY difficulty")
                    elif event.key == pygame.K_2:
                        self.__init__('medium', self.game_mode)
                        print("Switched to MEDIUM difficulty")
                    elif event.key == pygame.K_3:
                        self.__init__('medium-hard', self.game_mode)
                        print("Switched to MEDIUM-HARD difficulty")
            
            self.frame_count += 1
            
            for agent_name, game in self.games.items():
                
                if self.game_mode == 'survival' and not self.scores[agent_name]['alive']:
                    continue
                
                if not game.game_over and not game.won:
                    
                    if agent_name == 'A2C':
                        action = self.get_a2c_action(game)
                    elif agent_name == 'PPO':
                        action = self.get_ppo_action(game)
                    elif agent_name == 'Rainbow':
                        action = self.get_rainbow_action(game)
                    else:
                        action = self.get_doubleq_action(game)
                    
                    self.execute_action(game, action, agent_name)
                    
                    game.update()
            
            self.update_scores()
            
            self.reset_if_needed()
            
            screen.fill((0, 0, 0))
            
            self.draw_game_with_overlay(screen, self.games['A2C'], 0, 'A2C', self.agent_colors['A2C'])
            self.draw_game_with_overlay(screen, self.games['PPO'], WINDOW_WIDTH, 'PPO', self.agent_colors['PPO'])
            self.draw_game_with_overlay(screen, self.games['Rainbow'], WINDOW_WIDTH * 2, 'Rainbow', self.agent_colors['Rainbow'])
            self.draw_game_with_overlay(screen, self.games['DoubleQ'], WINDOW_WIDTH * 3, 'DoubleQ', self.agent_colors['DoubleQ'])
            
            self.draw_scoreboard(screen, WINDOW_WIDTH)
            
            pygame.display.flip()
            game_clock.tick(60)
        
        print(f"\n{'=' * 60}")
        print("FINAL RESULTS")
        print(f"{'=' * 60}")
        for agent_name in ['A2C', 'PPO', 'Rainbow', 'DoubleQ']:
            agent_stats = self.scores[agent_name]
            print(f"{agent_name:10} - Wins: {agent_stats['wins']:3} | Deaths: {agent_stats['deaths']:3} | Best Row: {agent_stats['max_progress']:3}")
        print(f"{'=' * 60}\n")
        
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    import argparse

    argument_parser = argparse.ArgumentParser(description = 'Run 4-agent battle (A2C vs PPO vs Rainbow vs DoubleQ)')
    argument_parser.add_argument('--difficulty', type = str, default = 'medium', 
                        choices = ['easy', 'medium', 'medium-hard'],
                        help = 'Difficulty level (default: medium)')
    argument_parser.add_argument('--mode', type = str, default = 'statistical',
                        choices = ['statistical', 'survival'],
                        help = 'Game mode: statistical (auto-restart) or survival (one-shot)')
    
    parsed_args = argument_parser.parse_args()
    
    agents_battle = AgentsBattle(difficulty = parsed_args.difficulty, game_mode = parsed_args.mode)
    agents_battle.run()
