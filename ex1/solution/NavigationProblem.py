# Representation of the navigation problem: find a path from s_start to s_end on the grid.
# Contains all required methods-
import collections


class NavigationProblem:
    def __init__(self, s_start, s_end, grid_size, grid):
        self.s_start = s_start
        self.s_end = s_end
        self.grid_size = grid_size
        self.grid = grid
        self.moves = {
            'R': lambda x: (x[1] < self.grid_size - 1) and (self.grid[x[0]][x[1] + 1] != -1),
            'RD': lambda x: self.moves.get('R')(x) and self.moves.get('D')(x) and (self.grid[x[0] + 1][x[1] + 1] != -1),
            'D': lambda x: (x[0] < self.grid_size - 1) and (self.grid[x[0] + 1][x[1]] != -1),
            'LD': lambda x: self.moves.get('L')(x) and self.moves.get('D')(x) and (self.grid[x[0] + 1][x[1] - 1] != -1),
            'L': lambda x: (self.grid[x[0]][x[1] - 1] != -1) and (x[1] > 0),
            'LU': lambda x: self.moves.get('L')(x) and self.moves.get('U')(x) and (self.grid[x[0] - 1][x[1] - 1] != -1),
            'U': lambda x: (self.grid[x[0] - 1][x[1]] != -1) and (x[0] > 0),
            'RU': lambda x: self.moves.get('R')(x) and self.moves.get('U')(x) and (self.grid[x[0] - 1][x[1] + 1] != -1)
        }

        self.transitions = collections.OrderedDict([
            ('R', (0, 1)),
            ('RD', (1, 1)),
            ('D', (1, 0)),
            ('LD', (1, -1)),
            ('L', (0, -1)),
            ('LU', (-1, -1)),
            ('U', (-1, 0)),
            ('RU', (-1, 1))
        ])

    # returns possible actions from a given state using allowed moves only
    def actions(self, s):
        return [m for (m, f) in self.moves.items() if f(s)]

    # given a state and an action- returns the new state we reached after the action
    def succ(self, s, a):
        new_s = tuple(map(sum, zip(s, self.transitions[a])))
        return new_s

    # returns whether or not the given state is the goal state
    def is_goal(self, s):
        return s == self.s_end

    # returns the cost of a step to the successor(step from a given state using a given action)
    def step_cost(self, s, a):
        next_node_using_action = self.succ(s, a)
        return self.grid[next_node_using_action[0]][next_node_using_action[1]]

    # returns a strip representation of the given state
    def state_str(self, s):
        return '\n'.join([str(s[i * self.grid_size:(i + 1) * self.grid_size]) for i in range(0, self.grid_size)])
