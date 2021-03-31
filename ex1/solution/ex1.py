import Heuristics
from BestFS_Searchers import uniform_cost_search, astar_search
from IDA_Star_Searcher import iterative_deepening_astar_search
from IDS_Searcher import iterative_deepening_search
from NavigationProblem import NavigationProblem


def main():
    # open input file and read its parameters
    inputFile = open('input.txt', 'r')
    searchAlg = inputFile.readline()[:-1]
    s_start = tuple(str_to_int_list(inputFile.readline(), ","))
    s_end = tuple(str_to_int_list(inputFile.readline(), ","))
    grid_size = int(inputFile.readline())
    grid = []

    # build the grid from input file
    for lineNum in range(grid_size):
        rowInt = str_to_int_list(inputFile.readline(), ",")
        grid.append(rowInt)

    inputFile.close()

    problem = NavigationProblem(s_start, s_end, grid_size, grid)

    # run the search
    pathSol = None
    developed_counter = 0
    final_path_cost = None
    try:
        if searchAlg == "IDS":
            pathSol, developed_counter, final_path_cost = iterative_deepening_search(problem)
        elif searchAlg == "UCS":
            pathSol, developed_counter, final_path_cost = uniform_cost_search(problem)
        elif searchAlg == "ASTAR":
            pathSol, developed_counter, final_path_cost = astar_search(problem, Heuristics.h_diagonal_distance)
        elif searchAlg == "IDASTAR":
            pathSol, developed_counter, final_path_cost = iterative_deepening_astar_search(problem, Heuristics.h_diagonal_distance)
    except Exception:
        pathSol = None

    # write solution to output file
    outputFile = open('output.txt', 'w')

    if (not pathSol):
        outputFile.write("no path")
    else:
        pathSol = '-'.join(pathSol)
        outputFile.write(pathSol + " " + str(final_path_cost) + " " + str(developed_counter))

    outputFile.close()


# convert a string containing numbers separated with a separator to list of integers
def str_to_int_list(str, separator):
    separatedStr = str.split(separator)
    # create a map object with each string converted to an integer
    separatedStrToIntMap = map(int, separatedStr)
    strAsInt = list(separatedStrToIntMap)
    return strAsInt


if __name__ == "__main__":
    main()
