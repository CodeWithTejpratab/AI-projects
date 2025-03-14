import numpy as np
import heapq
import re

from dataclasses import dataclass
from copy import deepcopy
from collections import deque
from typing import List


GOAL_STATE = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
DIRECTIONS = {
    (1, 0): "up", 
    (-1, 0): "down", 
    (0, -1): "right", 
    (0, 1): "left"
}
BOUND = 3


class EightPuzzleNode:    
    def __init__(self, initial_state: List[List[int]]):
        self.state = initial_state
        self.previous_node = None
        self.depth = 0
        self.heuristic = self.manhattan_distance()

    def __repr__(self):
        matrix_string = ""

        for row in self.state:
            for number in row:
                matrix_string += f"{number} "
            
            matrix_string += "\n"
        
        return matrix_string
    
    def __deepcopy__(self, memo):
        return EightPuzzleNode(deepcopy(self.state, memo))
    
    def __lt__(self, other):
        return (self.depth + self.heuristic) < (other.depth + other.heuristic)

    def in_bounds(self, row: int, col: int):
        n = BOUND

        if 0 <= row < n and 0 <= col < n:
            return True
        else:
            return False
        
    def is_goal(self):
        return np.array_equal(self.state, GOAL_STATE)   
    
    def expand(self):
        agent_location = get_value_index(self.state, 0)
        agent_row, agent_col = agent_location
        coordinates = []
        for (row, col) in [*DIRECTIONS.keys()]:
            coordinates.append((row+agent_row, col+agent_col))

        neighbors = []
        for (row, col) in coordinates:
            if self.in_bounds(row, col):
                self_copy = deepcopy(self)
                self_copy.state[agent_row][agent_col], self_copy.state[row][col] = self_copy.state[row][col], self_copy.state[agent_row][agent_col]
                self_copy.depth = self.depth + 1
                self_copy.heuristic = self_copy.manhattan_distance()
                neighbors.append(self_copy)

        return neighbors
    
    def manhattan_distance(self):
        agent_location = get_value_index(self.state, 0)
        distance = 0

        for row in range(BOUND):
            for col in range(BOUND):
                if (row, col) != agent_location:
                    value = self.state[row][col]
                    goal_x, goal_y = get_value_index(GOAL_STATE, value)

                    x_diff = abs(row - goal_x)
                    y_diff = abs(col - goal_y)
                    distance += x_diff + y_diff
        
        return distance
            
    
class Algorithm:
    def __init__(self, eight_puzzle: EightPuzzleNode):
        self.eight_puzzle = eight_puzzle
        self.steps = 0
        self.frontier_states = 0
        self.reached_states = 0

    def solution(self, node: EightPuzzleNode):
        result = [node]
        moves = []
        while node.previous_node:
            self.steps += 1
            curr_coord = get_value_index(node.state, 0)
            prev_coord = get_value_index(node.previous_node.state, 0)
            coord = tuple(np.subtract(prev_coord, curr_coord))
            moves.append(DIRECTIONS[coord])
            node = node.previous_node
            result.append(node)

        moves.reverse()
        result.reverse()

        return (result, moves)

    def search(self):
        if self.eight_puzzle.is_goal():
            return self.solution(self.eight_puzzle)
        
        frontier = self.init_frontier()
        reached = set([])

        node = None
        reached_goal = False
        while frontier:
            node = self.pop_frontier(frontier)
            if node.is_goal():
                self.frontier_states = len(frontier)
                self.reached_states = len(reached)
                reached_goal = True
                break

            state = tuple(tuple(row) for row in node.state)
            reached.add(state)

            for child in node.expand():
                state = tuple(tuple(row) for row in child.state)
                if state not in reached:
                    self.insert_frontier(frontier, child)
                    child.previous_node = node

        if not reached_goal:
            raise Exception("Frontier is empty but goal was not reached.")                
        return self.solution(node)
    
    def init_frontier(self):
        raise NotImplementedError(f"Class '{type(self).__name__}' does not implement 'init_frontier()'.")

    def pop_frontier(self, frontier):
        raise NotImplementedError(f"Class '{type(self).__name__}' does not implement 'pop_frontier()'.")

    def insert_frontier(self, frontier, child):
        raise NotImplementedError(f"Class '{type(self).__name__}' does not implement 'insert_frontier()'.")


class EightPuzzleBFS(Algorithm):
    def __init__(self, eight_puzzle: EightPuzzleNode):
        super().__init__(eight_puzzle)

    def init_frontier(self):
        return deque([self.eight_puzzle])
    
    def pop_frontier(self, frontier: deque):
        return frontier.popleft()
    
    def insert_frontier(self, frontier: deque, child: EightPuzzleNode):
        frontier.append(child)


class EightPuzzleAStar(Algorithm):
    def __init__(self, eight_puzzle: EightPuzzleNode):
        super().__init__(eight_puzzle)

    def init_frontier(self):
        frontier = [self.eight_puzzle]
        heapq.heapify(frontier)
        
        return frontier
    
    def pop_frontier(self, frontier: heapq):
        return heapq.heappop(frontier)
    
    def insert_frontier(self, frontier: heapq, child: EightPuzzleNode):
        heapq.heappush(frontier, child)


@dataclass
class EightPuzzleStatistics:
    solution: List[EightPuzzleNode]
    moves: List[str]
    steps: int
    frontier_states: int
    reached_states: int


def create_puzzle(input: str):
    input = re.sub('\D', '', input)

    if len(input) != pow(BOUND, 2) or set(map(int, input)) != set(range(9)):
        return None
    
    index = 0
    puzzle = [[0]*BOUND for i in range(BOUND)]
    for row in range(BOUND):
        for col in range(BOUND):
            puzzle[row][col] = int(input[index])
            index += 1

    return puzzle

def select_algorithm(input: str):
    input = re.sub('\D', '', input)
    if input == "":
        return None
    
    selection = int(input[0])
    if selection is None or 1 > selection > 2:
        return None
    else:
        return selection
    
def get_value_index(matrix: List[List[int]], value: int):
    np_array = np.atleast_1d(matrix)
    return [(int(row), int(col)) for row, col in zip(*np.where(np_array == value))][0]

def is_valid_puzzle(matrix: List[List[int]]):
    num_inverses = 0
    puzzle = [number for row in matrix for number in row]

    for i in range(len(puzzle)):
        for j in range(i + 1, len(puzzle)):
            if puzzle[i] != 0 and puzzle[j] != 0 and puzzle[i] > puzzle[j]:
                num_inverses += 1
    
    return num_inverses % 2 == 0

def get_inputs():
    puzzle = None
    input_text = "Enter these numbers in any order (012345678): "
    while puzzle is None:
        input_string = input(input_text)
        puzzle = create_puzzle(input_string)
        input_text = "Please ensure your input contains all numbers from 0-8 with no duplicates: "

        if puzzle != None:
            if not is_valid_puzzle(puzzle):
                puzzle = None
                input_text = "Puzzle is unsolvable! Please enter a new puzzle: "

    eight_puzzle = EightPuzzleNode(puzzle)

    selection = None
    input_text = "Choose either Breadth First Search (1) or A* Manhattan (2): "
    while selection is None:
        input_string = input(input_text)
        selection = select_algorithm(input_string)
        input_text = "Please ensure your input is either 1 (BFS) or 2 (A*): "

    algorithm = None
    if selection == 1:
        algorithm = EightPuzzleBFS(eight_puzzle)
    else:
        algorithm = EightPuzzleAStar(eight_puzzle)

    return (algorithm, selection)

def print_output(result: List[EightPuzzleNode], moves: List[str], algorithm: Algorithm):
    print("\nSolution")
    print("----------\n")
    print("Initial State:")
    print(result[0])

    for i, state in enumerate(result[1:]):
        print(f"Move: {moves[i]}\n")
        print(state)    

    
    print(f"Total Steps:\t{algorithm.steps}")
    print(f"Total Frontier States:\t{algorithm.frontier_states}")
    print(f"Total Reached States:\t{algorithm.reached_states}\n")

def continue_selection():
    input_text = "Would you like to continue (y/n)? "
    while True:
        input_string = input(input_text)
        if input_string.lower()[0] not in ['y', 'n']:
            input_text = "Please enter either 'y' or 'n': "
        else:
            selection = input_string.lower()[0]
            return selection == 'y'
        
def output_statistics(statistics: dict[int, List[EightPuzzleStatistics]]):
    bfs = statistics[1]
    a_star = statistics[2]

    bfs_avg_steps = 0
    bfs_avg_frontier_states = 0
    bfs_avg_reached_states = 0
    for item in bfs:
        bfs_avg_steps += item.steps
        bfs_avg_frontier_states += item.frontier_states
        bfs_avg_reached_states += item.reached_states
    
    bfs_len = len(bfs)
    if bfs_len > 0:
        bfs_avg_steps /= bfs_len
        bfs_avg_frontier_states /= bfs_len
        bfs_avg_reached_states /= bfs_len

    a_star_avg_steps = 0
    a_star_avg_frontier_states = 0
    a_star_avg_reached_states = 0
    for item in a_star:
        a_star_avg_steps += item.steps
        a_star_avg_frontier_states += item.frontier_states
        a_star_avg_reached_states += item.reached_states
    
    a_star_len = len(a_star)
    if a_star_len > 0:
        a_star_avg_steps /= a_star_len
        a_star_avg_frontier_states /= a_star_len
        a_star_avg_reached_states /= a_star_len

    print("\nStatistical Analysis:\t\tBFS\t|\tA* Manhattan")
    print("------------------------------------------------------------------")
    print(f"Number of Puzzles:\t\t{bfs_len}\t\t{a_star_len}")
    print(f"Avg. Steps:\t\t\t{bfs_avg_steps}\t\t{a_star_avg_steps}")
    print(f"Avg. Frontier States:\t\t{bfs_avg_frontier_states}\t\t{a_star_avg_frontier_states}")
    print(f"Avg. Reached States:\t\t{bfs_avg_reached_states}\t\t{a_star_avg_reached_states}")
    print("\n")


if __name__ == '__main__':
    statistics = {
        1: [],
        2: []
    }

    continue_program = True
    while continue_program:
        algorithm, selection = get_inputs()
        result, moves = algorithm.search()
        print_output(result, moves, algorithm)
        
        stats = EightPuzzleStatistics(
            result, 
            moves, 
            algorithm.steps,
            algorithm.frontier_states,
            algorithm.reached_states
        )
        statistics[selection].append(stats)

        continue_program = continue_selection()

    output_statistics(statistics)
