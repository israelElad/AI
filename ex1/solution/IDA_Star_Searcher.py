from Node import Node

# global variables
new_limit = float('inf')
developed_counter = 0
final_path_cost = None


# IDA* search algorithm's outer function, calls DFS_f_limit with updated limit each time.
def iterative_deepening_astar_search(problem, h):
    global developed_counter
    global new_limit
    global final_path_cost

    start_node = Node(problem.s_start)
    new_limit = h(start_node, problem)
    while True:
        f_limit = new_limit
        new_limit = float('inf')
        sol = DFS_f_limit(start_node, f_limit, h, problem)
        if sol:
            return sol, developed_counter, final_path_cost


# IDA* search algorithm's inner recursive function- DFS with f_limit
def DFS_f_limit(current_node, f_limit, h, problem):
    global developed_counter
    global new_limit
    global final_path_cost

    if (current_node.depth > 20):
        return None

    current_node_f = current_node.path_cost + h(current_node, problem)
    if (current_node_f > f_limit):
        new_limit = min(new_limit, current_node_f)
        return None
    if (problem.is_goal(current_node.state)):
        final_path_cost = current_node.path_cost
        return current_node.solution()
    for child in current_node.expand(problem):
        developed_counter += 1
        sol = DFS_f_limit(child, f_limit, h, problem)
        if sol:
            return sol
    return None
