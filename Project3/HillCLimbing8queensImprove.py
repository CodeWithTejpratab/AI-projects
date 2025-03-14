from random import randint, random, seed
from math import exp

N = 8

def configureRandomly(board, state):
    """Initialize board and state with a random configuration."""
    for i in range(N):
        state[i] = randint(0, N-1)
        board[state[i]][i] = 1

def printBoard(board):
    """Print the 2D array board configuration."""
    print("    " + "   ".join(str(i) for i in range(N)))
    print("  +" + "---+" * N)

    for i, row in enumerate(board):
        row_display = " | ".join("♛" if cell == 1 else " " for cell in row)
        print(f"{i} | {row_display} |") 
        print("  +" + "---+" * N) 

def printState(state):
    """Print the state array."""
    print(*state)
    
def compareStates(state1, state2):
    """Check if two states are identical."""
    return state1 == state2

def fill(board, value):
    """Fill the entire board with a specific value"""
    for i in range(N):
        for j in range(N):
            board[i][j] = value
        
def calculateObjective(state):
    """Calculate the number of attacking pairs on the board."""
    attacking = 0
    for i in range(N):
        for j in range(i + 1, N):
            if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                attacking += 1
    return attacking

def generateBoard(board, state):
    """Generate a board configuration given the state."""
    fill(board, 0)
    for i in range(N):
        board[state[i]][i] = 1
    
def copyState(state1, state2):
    """Copy one state array to another."""
    for i in range(N):
        state1[i] = state2[i]

def getNeighbour(state):
    """Generate a neighbouring state by randomly moving one queen."""
    neighbour = state[:]
    col = randint(0, N-1)
    row = randint(0, N-1)
    while row == state[col]:
        row = randint(0, N-1)
    neighbour[col] = row
    return neighbour

def simulatedAnnealing(board, state):
    """Solve N-Queens problem using Simulated Annealing."""
    current_state = state[:]
    current_board = [[0] * N for _ in range(N)]
    generateBoard(current_board, current_state)
    current_objective = calculateObjective(current_state)
    
    T = 1.0
    T_min = 0.0001
    alpha = 0.99

    while T > T_min:
        neighbour_state = getNeighbour(current_state)
        neighbour_board = [[0] * N for _ in range(N)]
        generateBoard(neighbour_board, neighbour_state)
        neighbour_objective = calculateObjective(neighbour_state)
        
        if neighbour_objective < current_objective:
            current_state = neighbour_state
            current_objective = neighbour_objective
        else:
            delta = neighbour_objective - current_objective
            if random() < exp(-delta / T):
                current_state = neighbour_state
                current_objective = neighbour_objective
        
        T *= alpha

        if current_objective == 0:
            copyState(state, current_state)
            generateBoard(board, state)
            return True
    
    copyState(state, current_state)
    generateBoard(board, state)
    return current_objective == 0

# Main function to run Simulated Annealing
successful_solutions = 0
total_attempts = 0

while successful_solutions == 0 or total_attempts < 20:
    total_attempts += 1
    state = [0] * N
    board = [[0 for _ in range(N)] for _ in range(N)]
    configureRandomly(board, state)

    if simulatedAnnealing(board, state):
        print(f"Solution #{successful_solutions + 1} (Found in attempt {total_attempts}):")
        printBoard(board)
        successful_solutions += 1

print(f"\nTotal attempts: {total_attempts}")
print(f"Successful solutions: {successful_solutions}")
print(f"Success rate: {successful_solutions / total_attempts * 100}%")