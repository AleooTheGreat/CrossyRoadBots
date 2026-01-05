import numpy as np
from environment.game import *
from agents.A2C.parameters import *

class Utilities:
    
    @staticmethod
    def get_game_state(game: Game, radius: int = 4):
        side_length = radius * 2 + 1
        state_grid = np.zeros((side_length, side_length), dtype=np.float32)

        for row in range(-radius, radius + 1):
            for col in range(-radius, radius + 1):
                grid_row = game.player_row + row
                grid_col = game.player_col + col


                if 0 <= grid_row < GRID_ROWS and 0 <= grid_col < GRID_COLS: # if inside the grid

                    is_safe = grid_row < 3 or grid_row >= GRID_ROWS - 3 or grid_row % game.level_config.safe_zone_spacing == 0
                    if is_safe: # safe zone
                        state_grid[row + radius, col + radius] = SAFE_ZONE_REWARD
                    else:
                        state_grid[row + radius, col + radius] = ROAD_PENALTY

                    # if player collides with a car
                    for car in game.cars:
                        if car.collides_with(grid_row, grid_col):
                            state_grid[row + radius, col + radius] = CAR_COLLISION_PENALTY
                            break
                else:
                    state_grid[row + radius, col + radius] = WALL_PENALTY

        return state_grid.flatten()