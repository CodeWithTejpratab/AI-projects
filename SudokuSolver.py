def print_board(board):
    for row in board:
        print(" ".join(row))
    print()


def sudoku_solver(board):
    # Initialization of trackers to make sure we're putting the right number in
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]

    for r in range(9):
        for c in range(9):
            if board[r][c] != '.':
                digit = board[r][c]
                rows[r].add(digit)
                cols[c].add(digit)
                box_index = (r // 3) * 3 + (c // 3)
                boxes[box_index].add(digit)

    def solve():
        for r in range(9):
            for c in range(9):
                if board[r][c] == '.':
                    box_index = (r // 3) * 3 + (c // 3)
                    for digit in "123456789":
                        if digit not in rows[r] and digit not in cols[c] and digit not in boxes[box_index]:
                            board[r][c] = digit
                            rows[r].add(digit)
                            cols[c].add(digit)
                            boxes[box_index].add(digit)

                            if solve():
                                return True

                            board[r][c] = '.'
                            rows[r].remove(digit)
                            cols[c].remove(digit)
                            boxes[box_index].remove(digit)
                    return False
        return True

    solve()


puzzle1 = [
    [".", "1", ".", "9", ".", "2", "6", ".", "4"],
    ["6", "0", "4", "3", ".", ".", ".", "7", "."],
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

print("Original Puzzle 1:")
print_board(puzzle1)
sudoku_solver(puzzle1)
print("Solved Puzzle 1:")
print_board(puzzle1)

print("Original Puzzle 2:")
print_board(puzzle2)
sudoku_solver(puzzle2)
print("Solved Puzzle 2:")
print_board(puzzle2)
