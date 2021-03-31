from Node import Node


# IDS search algorithm's inner function
def depth_limited_search(problem, limit):
    counter = 0
    frontier = [(Node(problem.s_start))]  # Stack
    while frontier:
        node = frontier.pop()
        counter += 1
        if problem.is_goal(node.state):
            return node.solution(), counter, node.path_cost
        if node.depth < limit:
            nodes = node.expand(problem)
            frontier.extend(reversed(nodes))
    return None, counter, None


# IDS search algorithm's outer function, iterating over depth_limited_search
def iterative_deepening_search(problem):
    developed_counter = 0
    for depth in range(1, 20):
        result, developed_counter_in_iteration, final_path_cost = depth_limited_search(problem, depth)
        developed_counter += developed_counter_in_iteration
        if result:
            return result, developed_counter, final_path_cost
    return None, developed_counter, None
