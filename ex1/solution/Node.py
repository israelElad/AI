# Represents a node. A node contains a state(location in our problem), cost, action (which way the parent went
# to arrive to the current node), depth, and its parent Node.
# It also contains all required methods- expand, solution path, operators etc.
class Node:
    # constructor
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    # returns all children of the current node according to the valid actions which can be taken
    def expand(self, problem):
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    # returns a child according to the given action
    def child_node(self, problem, action):
        next_state = problem.succ(self.state, action)
        next_node = Node(next_state, self, action,
                         self.path_cost + problem.step_cost(self.state, action))
        return next_node

    # returns a list containing the path from the first node to the last
    def solution(self):
        return [node.action for node in self.path()[1:]]

    # retrace the steps to this node by iterating back then reversing the results
    def path(self):
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # returns a printable representation of the object
    def __repr__(self):
        return f"<{self.state}>"

    # less-then operator
    def __lt__(self, node):
        return self.state < node.state

    # equality operator
    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    # not-equal operator
    def __ne__(self, other):
        return not (self == other)

    # returns the hash value of this node object
    def __hash__(self):
        return hash(self.state)
