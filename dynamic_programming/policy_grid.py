import numpy as np

def generate_very_large_connected_grid(size):
    """
    Generates a large, connected grid with a "rooms and corridors" structure.
    Goal 'G' is placed near the bottom-right corner.
    """
    if size < 10: # Ensure size is reasonable for this generation logic
        raise ValueError("Size must be at least 10 for this grid generation logic.")

    policy_grid = np.full((size, size), 'W', dtype='<U1') # Start with all walls

    # Define room characteristics
    # room_interior_size: how many 'U' cells across/down in a room's open space
    # wall_buffer: how many 'W' cells for the wall segment before the next room starts
    # block_size: total size of one room + its wall leading to the next
    
    # Aim for rooms that are not too tiny, e.g., 3x3 or 4x4 interior
    if size < 30:
        room_interior_size = 2
        wall_buffer = 1 # Wall is 1 cell thick
    elif size < 100:
        room_interior_size = 3
        wall_buffer = 1
    else: # For 200x200 or larger
        room_interior_size = 4
        wall_buffer = 1
        
    block_size = room_interior_size + wall_buffer

    num_rooms_across = (size - wall_buffer) // block_size
    num_rooms_down = (size - wall_buffer) // block_size

    for r_idx in range(num_rooms_down):
        for c_idx in range(num_rooms_across):
            # Calculate top-left corner of the 'U' interior of the room
            r_start_U = wall_buffer + r_idx * block_size
            c_start_U = wall_buffer + c_idx * block_size

            # Carve out the room's interior 'U' space
            # Ensure we don't go out of bounds (especially for last room if size isn't perfect multiple)
            r_end_U = min(r_start_U + room_interior_size, size - wall_buffer)
            c_end_U = min(c_start_U + room_interior_size, size - wall_buffer)
            
            policy_grid[r_start_U : r_end_U, c_start_U : c_end_U] = 'U'

            # Carve "door" to the room on the right (East)
            if c_idx < num_rooms_across - 1:
                door_r_E = r_start_U + room_interior_size // 2
                door_c_E = c_start_U + room_interior_size # This is the wall cell
                if door_r_E < size - wall_buffer and door_c_E < size - wall_buffer : # Check bounds
                    policy_grid[door_r_E, door_c_E] = 'U'

            # Carve "door" to the room below (South)
            if r_idx < num_rooms_down - 1:
                door_r_S = r_start_U + room_interior_size # This is the wall cell
                door_c_S = c_start_U + room_interior_size // 2
                if door_r_S < size - wall_buffer and door_c_S < size - wall_buffer: # Check bounds
                    policy_grid[door_r_S, door_c_S] = 'U'
    
    # Ensure outer border is solid wall (might be redundant but safe)
    policy_grid[0, :] = 'W'
    policy_grid[size-1, :] = 'W'
    policy_grid[:, 0] = 'W'
    policy_grid[:, size-1] = 'W'

    # Place Goal 'G' in the interior of the last room (bottom-right)
    # Calculate center of the last conceptual room's U-space
    goal_r = wall_buffer + (num_rooms_down - 1) * block_size + room_interior_size // 2
    goal_c = wall_buffer + (num_rooms_across - 1) * block_size + room_interior_size // 2
    
    # Ensure goal is within bounds and on a 'U' cell
    goal_r = min(max(goal_r, 1), size-2) # Clamp within playable area
    goal_c = min(max(goal_c, 1), size-2)

    policy_grid[goal_r, goal_c] = 'G'
    
    # Ensure a path to the goal if it landed awkwardly due to clamping/rounding
    # by making its neighbors U (if they are walls within the last room's block)
    # This logic is simplified; the room generation should make goal cell accessible.
    if policy_grid[goal_r, goal_c-1] == 'W' and goal_c-1 > 0: policy_grid[goal_r, goal_c-1] = 'U'
    if policy_grid[goal_r-1, goal_c] == 'W' and goal_r-1 > 0: policy_grid[goal_r-1, goal_c] = 'U'
    if goal_c+1 < size-1 and policy_grid[goal_r, goal_c+1] == 'W': policy_grid[goal_r, goal_c+1] = 'U'
    if goal_r+1 < size-1 and policy_grid[goal_r+1, goal_c] == 'W': policy_grid[goal_r+1, goal_c] = 'U'


    # A common starting point, ensure it's 'U'
    policy_grid[1,1] = 'U'
    if policy_grid[1,2] == 'W': policy_grid[1,2] = 'U' # path out
    if policy_grid[2,1] == 'W': policy_grid[2,1] = 'U' # path out


    return policy_grid

import numpy as np

# A new, very challenging 50x50 grid, inspired by the complex maze image
# Goal 'G' at (48, 48)
policy_grid = np.array([
    ['W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W'], #0
    ['W','U','U','U','W','U','U','U','W','U','U','U','U','U','W','U','U','U','W','U','U','U','U','U','W','U','U','U','W','U','U','U','U','U','W','U','U','U','W','U','U','U','U','U','W','U','U','U','U','W'], #1
    ['W','U','W','W','W','U','W','W','W','U','W','W','W','U','W','U','W','W','W','U','W','W','W','U','W','U','W','W','W','U','W','W','W','U','W','U','W','W','W','U','W','W','W','U','W','U','W','W','U','W'], #2
    ['W','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','W','U','U','U','W','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','W','U','U','U','W','U','U','U','U','W'], #3
    ['W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','U','W'], #4
    ['W','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','W','U','W'], #5
    ['W','U','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','U','W','U','W'], #6
    ['W','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','W','U','U','U','W'], #7
    ['W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W'], #8
    ['W','U','U','U','U','W','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','W','U','U','U','W'], #9
    ['W','U','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','U','U','W'], #10
    ['W','U','U','W','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','W'], #11
    ['W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','U','W'], #12
    ['W','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','W'], #13
    ['W','U','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','U','W'], #14
    ['W','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','W'], #15
    ['W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W'], #16
    ['W','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','W'], #17
    ['W','U','W','W','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','U','W','U','W'], #18
    ['W','U','U','U','U','W','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','W'], #19
    ['W','W','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','U','W'], #20
    ['W','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','W'], #21
    ['W','U','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','U','W'], #22
    ['W','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','W','U','W'], #23
    ['W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','U','W'], #24
    ['W','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','W','U','W'], #25
    ['W','U','W','W','U','W','W','W','W','W','W','W','W','U','W','W','W','W','U','W','W','W','W','W','U','W','W','U','U','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','U','W','U','W'], #26
    ['W','U','U','W','U','U','U','U','U','U','U','U','W','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','W','U','U','U','W'], #27
    ['W','W','U','W','W','W','W','W','W','W','W','U','U','W','W','W','U','W','W','W','U','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W'], #28
    ['W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','W','U','U','U','W'], #29
    ['W','U','W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','U','W','W','W','W','W','W','W','U','W','U','W','W','W','U','W'], #30
    ['W','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','W'], #31
    ['W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W','U','W','W','U','W','W','W','W','W','W','U','W','W','W','W','W','U','W'], #32
    ['W','U','U','U','U','U','U','U','W','W','W','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W'], #33
    ['W','U','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','U','W','W','U','W','W','W','U','W','W','W','W','W','W','W','W','W','W','U','W'], #34
    ['W','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','W','W','W','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W'], #35
    ['W','W','W','U','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W'], #36
    ['W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W'], #37
    ['W','U','W','W','W','W','U','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','U','W','U','U','W','W','U','W','U','W','W','U','U','W','W','W','W','W','W','W','W','W','W','U','G','W'], #38
    ['W','U','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','U','W','W','U','W','U','U','W','U','U','W','W','W','W','W','W','W','W','W','W','U','U','W'],  #39
    ['W','U','W','W','W','W','W','U','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','U','U','U','W','U','W','W','U','W','W','U','U','W','W','W','W','W','W','W','W','U','W','U','W'], #14
    ['W','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','W'], #15
    ['W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W'], #16
    ['W','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','W'], #17
    ['W','U','W','W','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','U','W'], #18
    ['W','U','U','U','U','W','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','W'], #19
    ['W','W','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','U','W'], #20
    ['W','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','W'], #21
    ['W','U','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','U','W'], #22
    ['W','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','W','U','W'], #23
    ['W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','U','W','U','W'], #24
    ['W','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','W','U','W'], #25
      ['W','U','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','U','W'], #14
    ['W','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','W'], #15
    ['W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','U','W','W','W','U','W','W','W','U','W','W','W'], #16
    ['W','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','W'], #17
    ['W','U','W','W','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','U','W'], #18
    ['W','U','U','U','U','W','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','W'], #19
    ['W','W','W','W','U','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','U','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','U','W'], #20
    ['W','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','W'], #21
    ['W','U','W','W','W','W','W','W','U','W','W','W','W','W','U','W','W','W','W','U','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','U','W','W','W','U','W'], #22
    ['W','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','W','U','W'], #23
    ['W','W','W','W','W','W','U','W','W','W','W','W','W','W','W','W','W','W','W','U','U','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','U','W','W','W','W','W','W','W','U','W'], #24
    ['W','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','W','U','U','U','U','U','U','U','W','U','W'], #25
    ], dtype='<U1')
