import pygame
import random
import sys
sys.path.append('..')
from constants import *

class Truck:
    def __init__(self, row, col, speed, direction, color):
        self.row = row
        self.col = col
        self.speed = speed
        self.direction = direction  
        self.color = color
        self.width = random.choice([3, 4, 5])
    
    def update(self):
        self.col += self.speed * self.direction
        
        if self.direction > 0 and self.col > GRID_COLS:
            self.col = -self.width
        elif self.direction < 0 and self.col < -self.width:
            self.col = GRID_COLS
    
    def draw(self, screen):
        x = self.col * CELL_SIZE
        y = self.row * CELL_SIZE
        width = self.width * CELL_SIZE
        height = CELL_SIZE
        darker_color = tuple(max(0, c - 30) for c in self.color)
        pygame.draw.rect(screen, darker_color, (x, y, width, height))
        if self.direction > 0:
            pygame.draw.rect(screen, self.color, (x + width - CELL_SIZE, y, CELL_SIZE, height))
        else:
            pygame.draw.rect(screen, self.color, (x, y, CELL_SIZE, height))
        pygame.draw.rect(screen, BLACK, (x + 2, y + 2, width - 4, height - 4))
    
    def collides_with(self, player_row, player_col):
        if self.row != player_row:
            return False
        return self.col <= player_col <= self.col + self.width
