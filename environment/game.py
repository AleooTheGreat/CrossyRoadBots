import pygame
import random
import sys
sys.path.append('..')
from constants import *
from environment.car import Car
from environment.truck import Truck

class Game:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.player_row = GRID_ROWS - 1
        self.player_col = GRID_COLS // 2
        self.score = 0
        self.highest_row = GRID_ROWS - 1
        self.game_over = False
        self.won = False
        self.cars = []
        

        vehicle_pattern = 0
        for row in range(3, GRID_ROWS - 3):
            if row % 5 != 0:  
                direction = random.choice([-1, 1])
                speed = random.uniform(0.05, 0.15)
                color = random.choice([RED, BLUE, YELLOW, PURPLE])
                
                vehicle_pattern += 1
                use_trucks = (vehicle_pattern % 12) > 9
                
                if use_trucks:
                    num_vehicles = random.randint(1, 2)
                else:
                    num_vehicles = random.randint(2, 4)
                
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
        
        if self.player_row == 0:
            self.won = True
    
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
            if row < 3 or row >= GRID_ROWS - 3 or row % 5 == 0:
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
