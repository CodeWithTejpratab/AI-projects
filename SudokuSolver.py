from time import sleep
import copy


def print_board(board):
    """
    Helper function to print the Sudoku board in a formatted way.
    """
    for row in board:
        print(" ".join(row))
    print()


def copy_board(board):
    """
    Returns a deep copy of the board.
    """
    return copy.deepcopy(board)


def sudoku_solver(board, which=True, show_process=False):
    """
    Solves the given Sudoku puzzle using recursive backtracking.
    """
    # Trackers for rows, columns, and individual smaller boxes
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]

    # Counters for statistics
    stats = {
        "steps": 0,
        "recursive_calls": 0,
        "backtracks": 0
    }

    # Initialize trackers with the initial board values.
    for r in range(9):
        for c in range(9):
            if board[r][c] != '.':
                digit = board[r][c]
                rows[r].add(digit)
                cols[c].add(digit)
                box_index = (r // 3) * 3 + (c // 3)
                boxes[box_index].add(digit)

    if which:
        solved, stats = solve(stats, board, rows, cols, boxes, show_process)
    else:
        solved, stats = hard_solve(stats, board, rows, cols, boxes, show_process)
    return solved, stats


def solve(stats, board, rows, cols, boxes, show_stats):
    stats["recursive_calls"] += 1

    # Use MRV heuristic: choose the empty cell with the fewest candidates.
    min_count = 10
    chosen_cell = None
    for r in range(9):
        for c in range(9):
            if board[r][c] == '.':
                stats["steps"] += 1
                box_index = (r // 3) * 3 + (c // 3)
                candidates = [digit for digit in "123456789"
                              if digit not in rows[r]
                              and digit not in cols[c]
                              and digit not in boxes[box_index]]
                if len(candidates) < min_count:
                    min_count = len(candidates)
                    chosen_cell = (r, c, candidates)
                if min_count == 1:
                    break
        if min_count == 1:
            break

    # No empty cell found means the puzzle is solved.
    if chosen_cell is None:
        return True, stats

    r, c, candidates = chosen_cell
    box_index = (r // 3) * 3 + (c // 3)
    counter = 0

    for digit in candidates:
        stats["steps"] += 1
        board[r][c] = digit
        rows[r].add(digit)
        cols[c].add(digit)
        boxes[box_index].add(digit)

        if show_stats and counter <= 200:
            counter += 1
            sleep(0.4)  # Slow down for display purposes.
            print(f"Step {stats['steps']}: Placed {digit} at ({r}, {c})")
            print_board(board)

        solved, stats = solve(stats, board, rows, cols, boxes, show_stats)
        if solved:
            return True, stats

        # Backtracking: undo the placement.
        board[r][c] = '.'
        rows[r].remove(digit)
        cols[c].remove(digit)
        boxes[box_index].remove(digit)
        stats["backtracks"] += 1

        if show_stats:
            print(f"Step {stats['steps']}: Backtracking from {digit} at ({r}, {c})")
            print_board(board)

    return False, stats


def hard_solve(stats, board, rows, cols, boxes, show_stats):
    stats["recursive_calls"] += 1
    # Find the first empty cell (fixed order).
    for r in range(9):
        for c in range(9):
            if board[r][c] == '.':
                box_index = (r // 3) * 3 + (c // 3)
                for digit in "123456789":
                    stats["steps"] += 1  # counter for each attempt
                    if digit not in rows[r] and digit not in cols[c] and digit not in boxes[box_index]:
                        board[r][c] = digit
                        rows[r].add(digit)
                        cols[c].add(digit)
                        boxes[box_index].add(digit)

                        if show_stats and stats["steps"] < 200:
                            sleep(0.4)
                            print(f"Step {stats['steps']}: Placed {digit} at ({r}, {c})")
                            print_board(board)

                        solved, stats = hard_solve(stats, board, rows, cols, boxes, show_stats)
                        if solved:
                            return True, stats

                        # Backtracking: undo the placement.
                        board[r][c] = '.'
                        rows[r].remove(digit)
                        cols[c].remove(digit)
                        boxes[box_index].remove(digit)
                        stats["backtracks"] += 1

                        if show_stats:
                            print(f"Step {stats['steps']}: Backtracking from {digit} at ({r}, {c})")
                            print_board(board)
                # If no valid digit worked for this empty cell, trigger backtracking.
                return False, stats
    # If no empty cell is found, the puzzle is solved.
    return True, stats


# Define your puzzles
puzzle1 = [
    [".", "1", ".", "9", ".", "2", "6", ".", "4"],
    ["6", ".", "4", "3", ".", ".", ".", "7", "."],
    [".", "7", ".", "1", "6", "4", ".", ".", "8"],
    [".", ".", "3", ".", "1", "9", "8", ".", "."],
    ["1", "5", ".", ".", "4", ".", ".", "9", "7"],
    [".", ".", "7", "8", "2", ".", "3", ".", "."],
    [".", ".", ".", "2", ".", "6", ".", "5", "."],
    [".", "3", ".", ".", ".", "7", "1", ".", "2"],
    ["9", ".", "2", "5", ".", "1", ".", ".", "."]
]

puzzle2 = [
    ["6", "8", ".", ".", ".", ".", ".", "5", "."],
    [".", ".", ".", ".", ".", "5", ".", ".", "."],
    [".", ".", "3", "8", ".", ".", "2", "6", "."],

    ["1", ".", "7", ".", "2", ".", ".", ".", "."],
    [".", ".", "9", "5", ".", "8", "6", ".", "."],
    [".", ".", ".", ".", "1", ".", "7", ".", "2"],

    [".", "2", "1", ".", ".", "9", "4", ".", "."],
    [".", ".", ".", "4", ".", ".", ".", ".", "."],
    [".", "3", ".", ".", ".", ".", ".", "2", "8"]
]

puzzles = [puzzle1, puzzle2]
largerstats = []  # To compare the stats of using the MRV and brute force
i = 0
for puzzle in puzzles:
    i += 1
    print(f"Original Puzzle {i}:")
    print_board(puzzle)
    # Use a fresh copy for each run
    board_copy = copy_board(puzzle)
    solved, stats = sudoku_solver(board_copy, which=True, show_process=False)
    if solved:
        print(f"Solved Puzzle {i} (MRV):")
    else:
        print("Solution could not be found")
    print_board(board_copy)
    print(f"Puzzle {i} (MRV) Stats:", stats)
    largerstats.append(stats)

i = 0
# Iteration for solution without MRV (hard_solve)
for puzzle in puzzles:
    i += 1
    print(f"Original Puzzle {i}:")
    print_board(puzzle)
    board_copy = copy_board(puzzle)
    solved, stats = sudoku_solver(board_copy, which=False, show_process=False)
    if solved:
        print(f"Solved Puzzle {i} (Brute Force):")
    else:
        print("Solution could not be found")
    print_board(board_copy)
    print(f"Puzzle {i} (Brute Force) Stats:", stats, "\n\n")
    largerstats.append(stats)

print("All stats:", largerstats)
