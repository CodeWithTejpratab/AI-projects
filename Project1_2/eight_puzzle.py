import numpy as np
import heapq
import re

from copy import deepcopy
from collections import deque
from typing import List


GOAL_STATE = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
DIRECTIONS = {
    (1, 0): "up", 
    (-1, 0): "down", 
    (0, -1): "left", 
    (0, 1): "right"
}
BOUND = 3

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


class EightPuzzleNode:    
    def __init__(self, initial_state: List[List[int]]):
        self.state = initial_state
        self.previous_node = None
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
        return self.heuristic < other.heuristic

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
        self.steps = -1
        self.frontier_states = 0
        self.reached_states = 0

    def solution(self, node: EightPuzzleNode):
        result = [node]
        while node.previous_node:
            node = node.previous_node
            result.append(node)

        result.reverse()

        return result

    def search(self):
        pass


class EightPuzzleBFS(Algorithm):
    def __init__(self, eight_puzzle: EightPuzzleNode):
        super().__init__(eight_puzzle)

    def search(self):
        if self.eight_puzzle.is_goal():
            return self

        frontier = deque([self.eight_puzzle])
        self.frontier_states += 1
        reached = set([])

        node = None
        while frontier:
            node = frontier.popleft()
            state = tuple(tuple(row) for row in node.state)
            reached.add(state)

            if node.is_goal():
                break
            else:
                for child in node.expand():
                    state = tuple(tuple(row) for row in child.state)
                    if state not in reached:
                        frontier.append(child)
                        self.frontier_states += 1
                        child.previous_node = node

            self.steps += 1
        
        self.reached_states = len(reached)
                
        return self.solution(node)
    

class EightPuzzleAStar(Algorithm):
    def __init__(self, eight_puzzle: EightPuzzleNode):
        super().__init__(eight_puzzle)

    def search(self):
        if self.eight_puzzle.is_goal():
            return self
        
        frontier = [self.eight_puzzle]
        heapq.heapify(frontier)
        self.frontier_states += 1
        reached = set([])

        node = None
        while frontier:
            node = heapq.heappop(frontier)
            state = tuple(tuple(row) for row in node.state)
            reached.add(state)

            if node.is_goal():
                break
            else:
                for child in node.expand():
                    state = tuple(tuple(row) for row in child.state)
                    if state not in reached:
                        heapq.heappush(frontier, child)
                        self.frontier_states += 1
                        child.previous_node = node
                    
            self.steps += 1
            
        self.reached_states = len(reached)

        return self.solution(node)
    

if __name__ == '__main__':
    puzzle = None
    input_text = "Enter a these numbers in any order (012345678): "
    while puzzle is None:
        input_string = input(input_text)
        puzzle = create_puzzle(input_string)
        input_text = "Please ensure your input contains all numbers from 0-8 with no duplicates: "

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

    result = algorithm.search()

    print("\nSolution")
    print("----------")
    for item in result:
        print(item)
    
    print(f"Total Steps:\t{algorithm.steps}")
    print(f"Total Frontier States:\t{algorithm.frontier_states}")
    print(f"Total Reached States:\t{algorithm.reached_states}\n")
