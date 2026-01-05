import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pygame
from constants import *
from environment.game import Game
from agents.A2C.a2c import A2C
from agents.A2C.utilities import Utilities


pygame.init()

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Crossy Road Bots")

clock = pygame.time.Clock()

file_paths = {
    "easy": "./easy/a2c_easy_best.pth",
    "medium": "./medium/a2c_medium_best.pth",
    "medium-hard": "./medium-hard/a2c_medium-hard_best.pth"
}

LEVEL = 'easy'
game = Game(level=LEVEL)
agent = A2C()

agent.load(filepath=file_paths[LEVEL])

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                game.set_level('easy')
                agent.load(filepath=file_paths['easy'])
            if event.key == pygame.K_2:
                game.set_level('medium')
                agent.load(filepath=file_paths['medium'])
            if event.key == pygame.K_3:
                game.set_level('medium-hard')
                agent.load(filepath=file_paths['medium-hard'])
            if event.key == pygame.K_r:
                game.reset()

    game_state = Utilities.get_game_state(game)
    action = agent.select_action(game_state)[0]
    
    if action == 0:
        game.move_player('up')
    elif action == 1:
        game.move_player('down')
    elif action == 2:
        game.move_player('left')
    elif action == 3:
        game.move_player('right')
    
    game.update()
    
    game.draw(screen)
    
    pygame.display.flip()
    
    clock.tick(60)

pygame.quit()
sys.exit()
