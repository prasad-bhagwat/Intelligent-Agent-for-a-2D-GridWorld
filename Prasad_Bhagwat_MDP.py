# Imports required for the program
from collections import deque

# Dictionary Macro used
string_dict = {"0":"Walk Up", "1":"Walk Down", "2":"Walk Left", "3":"Walk Right", "4": "Run Up", "5": "Run Down", "6": "Run Left", "7": "Run Right", "8" : "Exit"}


# Value iteration algorithm
def val_iteration():
    global grid_rows, grid_cols, grid_mdp, grid_dummy, grid_action_state_lookup, epsilon_compare
    global coord_string, state_list

    grid_util       = [[float(0)] * grid_cols for i in range(grid_rows)]
    grid_rewards    = [[grid_mdp[i][j] for j in range(grid_cols)] for i in range(grid_rows)]
    grid_policy     = [[grid_mdp[i][j] for j in range(grid_cols)] for i in range(grid_rows)]

    while 1:
        delta = 0.0
        for state in state_list:
            row     = coord_string[state][0]
            column  = coord_string[state][1]
            if(grid_rewards[row][column] is not None):

                maximum         = -float('inf')
                maximum_index   = 8
                for index, action_dict in enumerate(grid_action_state_lookup[row][column]):
                    accumlated_sum 	= 0.0
                    for action, probability in action_dict.items():
                        next_state_row  = coord_string[action][0]
                        next_state_col  = coord_string[action][1]
                        accumlated_sum  += probability * grid_util[next_state_row][next_state_col]
                    accumlated_sum = (accumlated_sum * gamma) + (r_walk if index < 4 else r_run)
                    if accumlated_sum > maximum:
                        maximum         = accumlated_sum
                        maximum_index   = index
                if maximum_index == 8:
                    maximum = 0

                intermediate_utility        = grid_rewards[row][column] + maximum
                delta                       = float(max(delta, float(abs(intermediate_utility - grid_util[row][column]))))
                grid_util[row][column]      = intermediate_utility
                grid_policy[row][column]    = string_dict[str(maximum_index)]
        if delta < epsilon_compare:
            return grid_policy


# Updating dictonary values
def update_dictionary(dictionary, key, probability):
    dictionary[key] = probability + dictionary.get(key, 0.0)


# Creating grid action state list
def create_action_state_lookup():
    global grid_mdp, grid_dummy, grid_action_state_lookup, grid_rows, grid_cols, p_walk, p_run, p_inv_walk, p_inv_run
    grid_action_state_lookup = []

    for i in range(grid_rows):
        row = []
        for j in range(grid_cols):
            row.append([])
        grid_action_state_lookup.append(row)

    # Lookup table of actions available at each state of the grid
    for row in range(grid_rows):
        for column in range(grid_cols):
            if (grid_dummy[row][column] is not None):

                # Walk up Action
                walk_up_dict = {}
                # Upward state is not wall and it is within boundary
                if (row + 1 < grid_rows and grid_mdp[row + 1][column] is not None):
                    update_dictionary(walk_up_dict, str(((row + 1) * 10000) + column), p_walk)

                # Upward state is a wall or it is not within boundary
                else:
                    update_dictionary(walk_up_dict, str((row * 10000) + column), p_walk)

                # Conditional probabilities for uncertainty
                if (column - 1 >= 0 and grid_mdp[row][column - 1] is not None):
                    update_dictionary(walk_up_dict, str((row * 10000) + (column - 1)), p_inv_walk)

                else:
                    update_dictionary(walk_up_dict, str((row * 10000) + column), p_inv_walk)

                if (column + 1 < grid_cols and grid_mdp[row][column + 1] is not None):
                    update_dictionary(walk_up_dict, str((row * 10000) +(column + 1)), p_inv_walk)

                else:
                    update_dictionary(walk_up_dict, str((row * 10000) + column), p_inv_walk)

                grid_action_state_lookup[row][column].append(walk_up_dict)

                # Walk down Action
                walk_down_dict = {}
                # Downward state is not wall and it is within boundary
                if (row - 1 >= 0 and grid_mdp[row - 1][column] is not None):
                    update_dictionary(walk_down_dict, str(((row - 1) * 10000) + column), p_walk)

                # Downward state is a wall or it is not within boundary
                else:
                    update_dictionary(walk_down_dict, str((row * 10000) + column), p_walk)

                # Conditional probabilities for uncertainty
                if (column - 1 >= 0 and grid_mdp[row][column - 1] is not None):
                    update_dictionary(walk_down_dict, str((row * 10000) + (column - 1)), p_inv_walk)

                else:
                    update_dictionary(walk_down_dict, str((row * 10000) + column), p_inv_walk)

                if (column + 1 < grid_cols and grid_mdp[row][column + 1] is not None):
                    update_dictionary(walk_down_dict, str((row * 10000) + (column + 1)), p_inv_walk)

                else:
                    update_dictionary(walk_down_dict, str((row * 10000) + column), p_inv_walk)

                grid_action_state_lookup[row][column].append(walk_down_dict)

                # Walk left Action
                walk_left_dict = {}
                # Leftward state is not wall and it is within boundary
                if (column - 1 >= 0 and grid_mdp[row][column - 1] is not None):
                    update_dictionary(walk_left_dict, str((row * 10000) + (column - 1)), p_walk)

                # Leftward state is a wall or it is not within boundary
                else:
                    update_dictionary(walk_left_dict, str((row * 10000) + column), p_walk)

                # Conditional probabilities for uncertainty
                if (row - 1 >= 0 and grid_mdp[row - 1][column] is not None):
                    update_dictionary(walk_left_dict, str(((row - 1) * 10000) + column), p_inv_walk)

                else:
                    update_dictionary(walk_left_dict, str((row * 10000) + column), p_inv_walk)

                if (row + 1 < grid_rows and grid_mdp[row + 1][column] is not None):
                    update_dictionary(walk_left_dict, str(((row + 1) * 10000) + column), p_inv_walk)

                else:
                    update_dictionary(walk_left_dict, str((row * 10000) + column), p_inv_walk)

                grid_action_state_lookup[row][column].append(walk_left_dict)

                # Walk right Action
                walk_right_dict = {}
                # Rightward state is not wall and it is within boundary
                if (column + 1 < grid_cols and grid_mdp[row][column + 1] is not None):
                    update_dictionary(walk_right_dict, str((row * 10000) + (column + 1)), p_walk)

                # Rightward state is a wall or it is not within boundary
                else:
                    update_dictionary(walk_right_dict, str((row * 10000) + column), p_walk)

                # Conditional probabilities for uncertainty
                if (row - 1 >= 0 and grid_mdp[row - 1][column] is not None):
                    update_dictionary(walk_right_dict, str(((row - 1) * 10000) + column), p_inv_walk)

                else:
                    update_dictionary(walk_right_dict, str((row * 10000) + column), p_inv_walk)

                if (row + 1 < grid_rows and grid_mdp[row + 1][column] is not None):
                    update_dictionary(walk_right_dict, str(((row + 1) * 10000) + column), p_inv_walk)

                else:
                    update_dictionary(walk_right_dict, str((row * 10000) + column), p_inv_walk)

                grid_action_state_lookup[row][column].append(walk_right_dict)

                # Run up Action
                run_up_dict = {}
                # Upward state is not wall and it is within boundary
                if (row + 2 < grid_rows and grid_mdp[row + 2][column] is not None and grid_mdp[row + 1][column] is not None):
                    update_dictionary(run_up_dict, str(((row + 2) * 10000) + column), p_run)

                # Upward state is a wall or it is not within boundary
                else:
                    update_dictionary(run_up_dict, str((row * 10000) + column), p_run)

                # Conditional probabilities for uncertainty
                if (column - 2 >= 0 and grid_mdp[row][column - 2] is not None and grid_mdp[row][column - 1] is not None):
                    update_dictionary(run_up_dict, str((row * 10000) + (column - 2)), p_inv_run)

                else:
                    update_dictionary(run_up_dict, str((row * 10000) + column), p_inv_run)

                if (column + 2 < grid_cols and grid_mdp[row][column + 2] is not None and grid_mdp[row][column + 1] is not None):
                    update_dictionary(run_up_dict, str((row * 10000) + (column + 2)), p_inv_run)

                else:
                    update_dictionary(run_up_dict, str((row * 10000) + column), p_inv_run)

                grid_action_state_lookup[row][column].append(run_up_dict)

                # Run down Action
                run_down_dict = {}
                # Downward state is not wall and it is within boundary
                if (row - 2 >= 0 and grid_mdp[row - 2][column] is not None and grid_mdp[row - 1][column] is not None):
                    update_dictionary(run_down_dict, str(((row - 2) * 10000) + column), p_run)

                # Downward state is a wall or it is not within boundary
                else:
                    update_dictionary(run_down_dict, str((row * 10000) + column), p_run)

                # Conditional probabilities for uncertainty
                if (column - 2 >= 0 and grid_mdp[row][column - 2] != None and grid_mdp[row][column - 1] != None):
                    update_dictionary(run_down_dict, str((row * 10000) + (column - 2)), p_inv_run)

                else:
                    update_dictionary(run_down_dict, str((row * 10000) + column), p_inv_run)

                if (column + 2 < grid_cols and grid_mdp[row][column + 2] is not None and grid_mdp[row][column + 1] is not None):
                    update_dictionary(run_down_dict, str((row * 10000) + (column + 2)), p_inv_run)

                else:
                    update_dictionary(run_down_dict, str((row * 10000) + column), p_inv_run)

                grid_action_state_lookup[row][column].append(run_down_dict)

                # Run left Action
                run_left_dict = {}
                # Leftward state is not wall and it is within boundary
                if (column - 2 >= 0 and grid_mdp[row][column - 2] is not None and grid_mdp[row][column - 1] is not None):
                    update_dictionary(run_left_dict, str((row * 10000) + (column - 2)), p_run)

                # Leftward state is a wall or it is not within boundary
                else:
                    update_dictionary(run_left_dict, str((row * 10000) + column), p_run)

                # Conditional probabilities for uncertainty
                if (row - 2 >= 0 and grid_mdp[row - 2][column] is not None and grid_mdp[row - 1][column] is not None):
                    update_dictionary(run_left_dict, str(((row - 2) * 10000) + column), p_inv_run)

                else:
                    update_dictionary(run_left_dict, str((row * 10000) + column), p_inv_run)

                if (row + 2 < grid_rows and grid_mdp[row + 2][column] is not None and grid_mdp[row + 1][column] is not None):
                    update_dictionary(run_left_dict, str(((row + 2) * 10000) + column), p_inv_run)

                else:
                    update_dictionary(run_left_dict, str((row * 10000) + column), p_inv_run)

                grid_action_state_lookup[row][column].append(run_left_dict)

                # Run right Action
                run_right_dict = {}
                # Rightward state is not wall and it is within boundary
                if (column + 2 < grid_cols and grid_mdp[row][column + 2] is not None and grid_mdp[row][column + 1] is not None):
                    update_dictionary(run_right_dict, str((row * 10000) + (column + 2)), p_run)

                # Rightward state is a wall or it is not within boundary
                else:
                    update_dictionary(run_right_dict, str((row * 10000) + column), p_run)

                # Conditional probabilities for uncertainty
                if (row - 2 >= 0 and grid_mdp[row - 2][column] is not None and grid_mdp[row - 1][column] is not None):
                    update_dictionary(run_right_dict, str(((row - 2) * 10000) + column), p_inv_run)

                else:
                    update_dictionary(run_right_dict, str((row * 10000) + column), p_inv_run)

                if (row + 2 < grid_rows and grid_mdp[row + 2][column] is not None and grid_mdp[row + 1][column] is not None):
                    update_dictionary(run_right_dict, str(((row + 2) * 10000) + column), p_inv_run)

                else:
                    update_dictionary(run_right_dict, str((row * 10000) + column), p_inv_run)

                grid_action_state_lookup[row][column].append(run_right_dict)


# Reading file and variable assignment
def read_file():
    global r_walk, r_run, gamma, grid_mdp, grid_dummy, grid_rows, grid_cols
    global p_walk, p_inv_walk, p_run, p_inv_run, epsilon_compare, coord_string, state_list
    coord_string            = {}
    state_list              = []
    state_queue             = deque()                                                                   # State queue
    visited_state           = {}                                                                        # Visited list
    epsilon                 = 10 ** -45                                                                 # Epsilon
    grid_mdp                = [[]]                                                                      # Input grid
    grid_dummy              = [[]]
    grid_rows, grid_cols    = 0, 0

    with open("input.txt", "r") as f:
        data = f.read().strip().split('\n')
        grid_size   = data[0].rstrip().split(",")                                                       # Grid Size
        grid_rows   = int(grid_size[0])                                                                 # Grid Rows
        grid_cols   = int(grid_size[1])                                                                 # Grid Columns

        grid_mdp    = [[float(0)] * grid_cols for i in range(grid_rows)]
        grid_dummy  = [[float(0)] * grid_cols for i in range(grid_rows)]

        num_walls = int(data[1])                                                                        # Wall Count

        for walls in data[2 : (2 + num_walls)]:
            line = walls.rstrip().split(",")
            grid_mdp[int(line[0]) - 1][int(line[1]) - 1]    = None
            grid_dummy[int(line[0]) - 1][int(line[1]) - 1]  = None

        num_terminals = int(data[2 + num_walls])

        for terminal in data[3 + num_walls : (3 + num_walls + num_terminals)]:
            line = terminal.rstrip().split(",")
            grid_mdp[int(line[0]) - 1][int(line[1]) - 1]    = float(line[2])
            grid_dummy[int(line[0]) - 1][int(line[1]) - 1]  = None
            state_queue.append(str(((int(line[0]) - 1) * 10000) + (int(line[1]) - 1)))

        p_walk_run = data[3 + num_walls + num_terminals].rstrip().split(",")
        p_walk     = float(p_walk_run[0])                                                               # Walk probability
        p_run      = float(p_walk_run[1])                                                               # Run probability
        p_inv_walk = float(0.5 * float(1 - p_walk))
        p_inv_run  = float(0.5 * float(1 - p_run))

        r_walk_run = data[4 + num_walls + num_terminals].rstrip().split(",")
        r_walk     = float(r_walk_run[0])                                                               # Walk reward
        r_run      = float(r_walk_run[1])                                                               # Run reward

        gamma           = float(data[5 + num_walls + num_terminals])                                    # Gamma
        epsilon_compare = epsilon * (1.0 - gamma) / gamma

        # Creating dictionary of co-ordinate strings & list of visited states
        for row in range(grid_rows):
            for column in range(grid_cols):
                grid_location 				= (row * 10000) + column
                coord_string[str(grid_location)] 	= (row, column)
                visited_state[str(grid_location)] 	= False

        # Creating list of all the states to be visited from all the terminals in inside out fashion
        while state_queue:
            current_state = state_queue.popleft()
            if visited_state[current_state] is False:
                visited_state[current_state] = True
                current_row     = coord_string[current_state][0]
                current_column  = coord_string[current_state][1]
                state_list.append(current_state)
                # If walk up is a valid action
                if current_row + 1 < grid_rows:
                    state_queue.append(str(((current_row + 1) * 10000) + current_column))

                # If walk down is a valid action
                if current_row - 1 >= 0:
                    state_queue.append(str(((current_row - 1) * 10000) + current_column))

                # If walk left is a valid action
                if current_column - 1 >= 0:
                    state_queue.append(str((current_row * 10000) + (current_column - 1)))

                # If walk right is a valid action
                if current_column + 1 < grid_cols:
                    state_queue.append(str((current_row * 10000) + (current_column + 1)))

                # If run up is a valid action
                if current_row + 2 < grid_rows:
                    state_queue.append(str(((current_row + 2) * 10000) + current_column))

                # If walk down is a valid action
                if current_row - 2 >= 0:
                    state_queue.append(str(((current_row - 2) * 10000) + current_column))

                # If walk left is a valid action
                if current_column - 2 >= 0:
                    state_queue.append(str((current_row * 10000) + (current_column - 2)))

                # If walk right is a valid action
                if current_column + 2 < grid_cols:
                    state_queue.append(str((current_row * 10000) + (current_column + 2)))


# Writing result to output file
def write_file(output):
    with open("output.txt", "w") as f:
        f.write(output)


# Main function
def main():
    file_content = ""
    read_file()
    create_action_state_lookup()
    output = val_iteration()
    for row in output[::-1]:
        file_content += (','.join(map(str,row)))
        file_content += "\n"
    write_file(file_content)


# Entry point of the program
if __name__ == '__main__':
    main()
