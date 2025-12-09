import pygame
import sys
from constants import *
from environment.game import Game

pygame.init()

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Crossy Road Bots")

clock = pygame.time.Clock()

# Start with medium difficulty (can be changed to 'easy', 'medium', or 'medium-hard')
game = Game(level='medium')

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP or event.key == pygame.K_w:
                game.move_player('up')
            elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                game.move_player('down')
            elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                game.move_player('left')
            elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                game.move_player('right')
            elif event.key == pygame.K_r:
                game.reset()

            elif event.key == pygame.K_1:
                game.set_level('easy')
            elif event.key == pygame.K_2:
                game.set_level('medium')
            elif event.key == pygame.K_3:
                game.set_level('medium-hard')
    
    game.update()
    
    game.draw(screen)
    
    pygame.display.flip()
    
    clock.tick(60)

pygame.quit()
sys.exit()
