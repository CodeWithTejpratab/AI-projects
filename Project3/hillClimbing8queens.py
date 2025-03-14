from random import randint

N = 8

def configureRandomly(board, state):
    """Initialize board and state with a random configuration."""
    for i in range(N):
        # Getting a random row index
        state[i] = randint(0, N-1)
		
        # Placing a queen on the
 		# obtained place in
 		# chessboard.
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
		
def calculateObjective( board, state):

	# For each queen in a column, we check
	# for other queens falling in the line
	# of our current queen and if found,
	# any, then we increment the variable
	# attacking count.

	attacking = 0

	# Variables to index a particular
	# row and column on board.
	for i in range(N):

		# At each column 'i', the queen is
		# placed at row 'state[i]', by the
		# definition of our state.

		# To the left of same row
		# (row remains constant
		# and col decreases)
		row = state[i]
		col = i - 1
		while (col >= 0 and board[row][col] != 1):
			col -= 1
		
		if (col >= 0 and board[row][col] == 1):
			attacking += 1
		
		# To the right of same row
		# (row remains constant
		# and col increases)
		row = state[i]
		col = i + 1
		while (col < N and board[row][col] != 1):
			col += 1
		
		if (col < N and board[row][col] == 1):
			attacking += 1
		
		# Diagonally to the left up
		# (row and col simultaneously
		# decrease)
		row = state[i] - 1
		col = i - 1
		while (col >= 0 and row >= 0 and board[row][col] != 1):
			col-= 1
			row-= 1
		
		if (col >= 0 and row >= 0 and board[row][col] == 1):
			attacking+= 1
		
		# Diagonally to the right down
		# (row and col simultaneously
		# increase)
		row = state[i] + 1
		col = i + 1
		while (col < N and row < N and board[row][col] != 1):
			col+= 1
			row+= 1
		
		if (col < N and row < N and board[row][col] == 1):
			attacking += 1
		
		# Diagonally to the left down
		# (col decreases and row
		# increases)
		row = state[i] + 1
		col = i - 1
		while (col >= 0 and row < N and board[row][col] != 1):
			col -= 1
			row += 1
		
		if (col >= 0 and row < N and board[row][col] == 1):
			attacking += 1
		
		# Diagonally to the right up
		# (col increases and row
		# decreases)
		row = state[i] - 1
		col = i + 1
		while (col < N and row >= 0 and board[row][col] != 1):
			col += 1
			row -= 1
		
		if (col < N and row >= 0 and board[row][col] == 1):
			attacking += 1
		
	return attacking // 2

# function that
# generates a board configuration
# given the state.
def generateBoard(board, state):
	fill(board, 0)
	for i in range(N):
		board[state[i]][i] = 1
	
def copyState( state1, state2):

	for i in range(N):
		state1[i] = state2[i]
	
# This function gets the neighbor
# of the current state having
# the least objective value
# amongst all neighbors as
# well as the current state.
def getNeighbour(board, state):
    """Find the best neighboring state with the least attacking pairs."""
    opState = state[:]
    opBoard = [[0] * N for _ in range(N)]
    generateBoard(opBoard, opState)
    opObjective = calculateObjective(opBoard, opState)

    # Try moving each queen to a different row in its column
    for i in range(N):
        for j in range(N):
            if j != state[i]:  # Skip the current position
                tempState = state[:]
                tempState[i] = j  # Move queen in column `i` to row `j`
                tempBoard = [[0] * N for _ in range(N)]
                generateBoard(tempBoard, tempState)
                tempObjective = calculateObjective(tempBoard, tempState)

                # If this move improves the objective function, update optimal
                if tempObjective < opObjective:
                    opObjective = tempObjective
                    opState = tempState[:]

    # Apply the best move found
    copyState(state, opState)
    generateBoard(board, state)

def hillClimbing(board, state):

	# Declaring and initializing the
	# Neighbor board and state with
	# the current board and the state
	# as the starting point.
	neighbourBoard = [[0 for _ in range(N)] for _ in range(N)]
	neighbourState = [0 for _ in range(N)]

	copyState(neighbourState, state)
	generateBoard(neighbourBoard, neighbourState)
	
	while True:

		# Copying the neighbor board and
		# state to the current board and
		# state, since a neighbor
		# becomes current after the jump.
		copyState(state, neighbourState)
		generateBoard(board, state)

		# Getting the optimal neighbour
		getNeighbour(neighbourBoard, neighbourState)

		if (compareStates(state, neighbourState)):

			# found a solution.
			return calculateObjective(board, state) == 0
		
		elif (calculateObjective(board, state) == calculateObjective(neighbourBoard,neighbourState)):

			# escape local optimum.
			# Random neighbor
			neighbourState[randint(0, N-1)] = randint(0, N-1)
			generateBoard(neighbourBoard, neighbourState)
				
# Main
successfulSolutions = 0
totalAttempts = 0

while successfulSolutions == 0 or totalAttempts < 20:
    totalAttempts += 1
    state = [0] * N
    board = [[0 for _ in range(N)] for _ in range(N)]
    configureRandomly(board, state)

    if hillClimbing(board, state):
        print(f"Solution #{successfulSolutions + 1} (Found in attempt {totalAttempts}):")
        printBoard(board)
        successfulSolutions += 1

print(f"\nTotal attempts: {totalAttempts}")
print(f"Successful solutions: {successfulSolutions}")
print(f"Success rate: {successfulSolutions / totalAttempts * 100}%")
