import numpy as np
from constants import GRID_ROWS, GRID_COLS


def get_state(game, view_range=5):  # Rename to get_local_state to match your other files
    """
    Returns a flattened grid centered on the player.
    Uses exact collision logic to ensure no 'invisible' cars.
    """
    # 1. Create full grid
    full_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

    # 2. Mark Cars using EXACT collision logic
    for car in game.cars:
        r = car.row
        # Optimization: Only check columns near the car instead of 0..24
        # We look a bit wider (width+1) to be safe against float boundaries
        start_check = int(car.col) - 1
        end_check = int(car.col + car.width) + 2

        for c in range(start_check, end_check):
            if 0 <= c < GRID_COLS:
                # If the game says this column kills the player, mark it as 1.0
                if car.collides_with(r, c):
                    full_grid[r][c] = 1.0

    # 3. Padding (Walls = 2.0)
    padded_grid = np.pad(full_grid, pad_width=view_range, mode='constant', constant_values=2.0)

    # 4. Extract Window
    p_r = game.player_row + view_range
    p_c = game.player_col + view_range

    r_start = p_r - view_range
    r_end = p_r + view_range + 1
    c_start = p_c - view_range
    c_end = p_c + view_range + 1

    local_view = padded_grid[r_start:r_end, c_start:c_end]

    return local_view.flatten()