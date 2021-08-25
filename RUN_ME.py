import tsp_solver

if __name__ == "__main__":
    a = tsp_solver.tsp_solver()
    a.optimize("./tour100.csv")