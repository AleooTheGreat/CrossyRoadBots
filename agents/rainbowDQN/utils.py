import numpy as np
from constants import GRID_ROWS, GRID_COLS


def get_state(game, view_range=5):
    
    window_size = view_range * 2 + 1
    local_view = np.zeros((window_size, window_size), dtype=np.float32)
    
    for car in game.cars:
        car_row = car.row
        
        start_check = int(car.col) - 1
        end_check = int(car.col + car.width) + 2

        for check_col in range(start_check, end_check):
            if 0 <= check_col < GRID_COLS:
                if car.collides_with(car_row, check_col):
                    
                    row_offset = car_row - game.player_row + view_range
                    col_offset = check_col - game.player_col + view_range
                    
                    if 0 <= row_offset < window_size and 0 <= col_offset < window_size:
                        local_view[row_offset][col_offset] = 1.0
    
    for r_offset in range(window_size):
        for c_offset in range(window_size):
            actual_col = game.player_col - view_range + c_offset
            
            if not (0 <= actual_col < GRID_COLS):
                local_view[r_offset][c_offset] = 2.0

    return local_view.flatten()