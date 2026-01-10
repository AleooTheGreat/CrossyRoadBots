import pygame
import random
import sys
sys.path.append('..')
from constants import *
from environment.car import Car
from environment.truck import Truck
from environment.level_config import LevelManager

class Game:
    def __init__(self, level='medium', infinite_mode=False):
        self.level_config = LevelManager.get_level(level)
        self.infinite_mode = infinite_mode
        self.reset()
    
    def reset(self):
        self.player_row = GRID_ROWS - 1
        self.player_col = GRID_COLS // 2
        self.score = 0
        self.highest_row = GRID_ROWS - 1
        self.game_over = False
        self.won = False
        self.cars = []
        self.rows_generated = GRID_ROWS  # Track how many rows we've generated
        
        vehicle_pattern = 0
        for row in range(3, GRID_ROWS - 3):
          
            is_safe_zone = (row % self.level_config.safe_zone_spacing == 0)
            if not is_safe_zone:  
                direction = random.choice([-1, 1])
                speed = random.uniform(self.level_config.min_speed, self.level_config.max_speed)
                color = random.choice([RED, BLUE, YELLOW, PURPLE])
                
                vehicle_pattern += 1
                use_trucks = (random.randint(0, 100) < self.level_config.truck_frequency)
                
                if use_trucks:
                    num_vehicles = random.randint(max(1, self.level_config.min_vehicles - 1), 
                                                 max(2, self.level_config.max_vehicles - 2))
                else:
                    num_vehicles = random.randint(self.level_config.min_vehicles, 
                                                 self.level_config.max_vehicles)
                
                lane_vehicles = []
                for i in range(num_vehicles):
                    base_col = (GRID_COLS // num_vehicles) * i
                    
                    max_attempts = 10
                    for attempt in range(max_attempts):
                        col = base_col + random.randint(-2, 2)
                        
                        if use_trucks:
                            temp_vehicle = Truck(row, col, speed, direction, color)
                        else:
                            temp_vehicle = Car(row, col, speed, direction, color)
                        
                        overlaps = False
                        for existing_vehicle in lane_vehicles:
                            min_spacing = temp_vehicle.width + existing_vehicle.width
                            if abs(col - existing_vehicle.col) < min_spacing:
                                overlaps = True
                                break
                        
                        if not overlaps:
                            lane_vehicles.append(temp_vehicle)
                            self.cars.append(temp_vehicle)
                            break
    
    def set_level(self, level_name):
        self.level_config = LevelManager.get_level(level_name)
        self.reset()
    
    def move_player(self, direction):
        if self.game_over or self.won:
            return
        
        new_row = self.player_row
        new_col = self.player_col
        
        if direction == 'up':
            new_row = max(0, self.player_row - 1)
        elif direction == 'down':
            new_row = min(GRID_ROWS - 1, self.player_row + 1)
        elif direction == 'left':
            new_col = max(0, self.player_col - 1)
        elif direction == 'right':
            new_col = min(GRID_COLS - 1, self.player_col + 1)
        
        self.player_row = new_row
        self.player_col = new_col
        
        if self.player_row < self.highest_row:
            self.score += (self.highest_row - self.player_row) * 10
            self.highest_row = self.player_row
        
        # In infinite mode, never win, just keep generating rows
        if not self.infinite_mode and self.player_row == 0:
            self.won = True
        
        # Generate new rows if player is advancing in infinite mode
        if self.infinite_mode and self.player_row < 10:
            self._generate_new_rows()
    
    def update(self):
        if self.game_over or self.won:
            return
        
        for car in self.cars:
            car.update()
        
        for car in self.cars:
            if car.collides_with(self.player_row, self.player_col):
                self.game_over = True
    
    def draw(self, screen):
        screen.fill(WHITE)
        
        for row in range(GRID_ROWS):
            if row < 3 or row >= GRID_ROWS - 3 or row % self.level_config.safe_zone_spacing == 0:
                color = DARK_GREEN
            else:
                color = DARK_GRAY
            
            pygame.draw.rect(screen, color, (0, row * CELL_SIZE, WINDOW_WIDTH, CELL_SIZE))
        
        self.draw_grid(screen)
        
        for car in self.cars:
            car.draw(screen)
        
        self.draw_player(screen)
        
        font = pygame.font.Font(None, 24)
        score_text = font.render(f'Score: {self.score}', True, BLACK)
        screen.blit(score_text, (10, 10))
        
        level_text = font.render(f'Level: {self.level_config.name}', True, BLACK)
        screen.blit(level_text, (10, 35))
        
        font_large = pygame.font.Font(None, 36)
        if self.game_over:
            text = font_large.render('GAME OVER! Press R to Restart', True, RED)
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            pygame.draw.rect(screen, WHITE, text_rect.inflate(20, 20))
            screen.blit(text, text_rect)
        elif self.won:
            text = font_large.render('YOU WIN! Press R to Restart', True, GREEN)
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            pygame.draw.rect(screen, WHITE, text_rect.inflate(20, 20))
            screen.blit(text, text_rect)
    
    def draw_grid(self, screen):
        for row in range(GRID_ROWS + 1):
            pygame.draw.line(screen, GRAY, (0, row * CELL_SIZE), (WINDOW_WIDTH, row * CELL_SIZE), 1)
        for col in range(GRID_COLS + 1):
            pygame.draw.line(screen, GRAY, (col * CELL_SIZE, 0), (col * CELL_SIZE, WINDOW_HEIGHT), 1)
    
    def draw_player(self, screen):
        center_x = self.player_col * CELL_SIZE + CELL_SIZE // 2
        center_y = self.player_row * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 2 - 2
        pygame.draw.circle(screen, GREEN, (center_x, center_y), radius)
        pygame.draw.circle(screen, BLACK, (center_x, center_y), radius, 2)
    
    def _generate_new_rows(self):
        """Generate new rows at the top for infinite mode."""
        # Shift all cars down
        shift_amount = 10
        for car in self.cars:
            car.row += shift_amount
        
        # Remove cars that are too far down
        self.cars = [car for car in self.cars if car.row < GRID_ROWS + 20]
        
        # Adjust player and tracking
        self.player_row += shift_amount
        self.highest_row += shift_amount
        
        # Generate new rows at the top
        for row in range(3, 3 + shift_amount):
            is_safe_zone = (row % self.level_config.safe_zone_spacing == 0)
            if not is_safe_zone:
                direction = random.choice([-1, 1])
                speed = random.uniform(self.level_config.min_speed, self.level_config.max_speed)
                color = random.choice([RED, BLUE, YELLOW, PURPLE])
                
                use_trucks = (random.randint(0, 100) < self.level_config.truck_frequency)
                
                if use_trucks:
                    num_vehicles = random.randint(max(1, self.level_config.min_vehicles - 1), 
                                                 max(2, self.level_config.max_vehicles - 2))
                else:
                    num_vehicles = random.randint(self.level_config.min_vehicles, 
                                                 self.level_config.max_vehicles)
                
                lane_vehicles = []
                for i in range(num_vehicles):
                    base_col = (GRID_COLS // num_vehicles) * i
                    col = base_col + random.randint(-2, 2)
                    
                    if use_trucks:
                        temp_vehicle = Truck(row, col, speed, direction, color)
                    else:
                        temp_vehicle = Car(row, col, speed, direction, color)
                    
                    overlaps = False
                    for existing_vehicle in lane_vehicles:
                        min_spacing = temp_vehicle.width + existing_vehicle.width
                        if abs(col - existing_vehicle.col) < min_spacing:
                            overlaps = True
                            break
                    
                    if not overlaps:
                        lane_vehicles.append(temp_vehicle)
                        self.cars.append(temp_vehicle)
        
        self.rows_generated += shift_amount

