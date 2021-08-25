## Individual Semester Project in Genetic Algorithms and Evolutionary Computing (AI Master's Degree Program) : Solving the Traveling Salesperson Problem using an Evolutionary Algorithm

The traveling salesperson problem consists of minimizing the length of a cycle that visits all vertices (ex. cities) in a
weighted, directed graph.   
The length of a cycle is defined as the sum of the weights of all directed edges constituting the cycle (ex. distances between cities).
In this project, the distances between N cities are stored in a (N x N) matrix in the form of a .csv file.

* In order to run the traveling salesperson problem solver, use the **RUN_ME.py** file.  
* You can test the algorithm with any .csv file containing decimal numbers aranged in a (NxN) matrix format or with one of the four *tour<number_of_cities>.csv* files.  
Simply change the name of the tourXXX.csv in RUN_ME.py file.  
* **Reporter.py** creates a .csv file with the results of a succesful execution of the algorithm.

The larger the number of cities, the slower the calculations.
You can change the time the algorithm is allowed to run from the Reporter.py file (current configuration: 300 sec).
