# Computes Chebyshev's diagonal distance: The difference between our X-axis and the X-axis of the goal (distance - absolute value),
# and the difference between our Y-axis and that of the goal - then take the maximum between them.
def h_diagonal_distance(node, problem):
    return max(abs(problem.s_end[0] - node.state[0]), abs(problem.s_end[1] - node.state[1]))
