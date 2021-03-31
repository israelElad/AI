from Node import Node
from PriorityQueue import PriorityQueue


# best first search base algorithm. using priority queue as frontier(open list) and a closed list
def best_first_graph_search(problem, f):
    developed_counter = 0
    node = Node(problem.s_start)
    frontier = PriorityQueue(f)  # Priority Queue
    frontier.append(node)
    closed_list = set()
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node.solution(), developed_counter, node.path_cost
        developed_counter += 1
        closed_list.add(node.state)
        for child in node.expand(problem):
            if child.state not in closed_list and child not in frontier:
                frontier.append(child)
            elif child in frontier and f(child) < frontier[child]:
                del frontier[child]
                frontier.append(child)
    return None, developed_counter, None


# UCS algorithm, BestFS with f = g
def uniform_cost_search(problem):
    def g(node):
        return node.path_cost

    return best_first_graph_search(problem, f=g)


# A* algorithm, BestFS with f = g + h
def astar_search(problem, h):
    def g(node):
        return node.path_cost

    return best_first_graph_search(problem, f=lambda n: g(n) + h(n, problem))
