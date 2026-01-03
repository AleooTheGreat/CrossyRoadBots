import numpy as np
from environment.game import *

class Utilities:
    
    @staticmethod
    def get_game_state(game: Game, radius: int = 3):
        side_length = radius * 2 + 1
        state_grid = np.full((side_length, side_length), -1.0, dtype=np.float32)

        for row in range(-radius, radius + 1):
            for col in range(-radius, radius + 1):
                grid_row = game.player_row + row
                grid_col = game.player_col + col


                if 0 <= grid_row < GRID_ROWS and 0 <= grid_col < GRID_COLS: # if inside the grid
                    # check if it's a safe zone or road
                    is_safe = grid_row < 3 or grid_row >= GRID_ROWS - 3 or grid_row % game.level_config.safe_zone_spacing == 0
                    if is_safe:
                        state_grid[row + radius, col + radius] = 1.0
                    else:
                        state_grid[row + radius, col + radius] = 0.0

                    # if player collides with a car
                    for car in game.cars:
                        if car.collides_with(grid_row, grid_col):
                            state_grid[row + radius, col + radius] = 2.0
                            break

        return state_grid.flatten()