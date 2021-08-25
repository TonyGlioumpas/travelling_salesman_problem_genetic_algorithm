import Reporter
import numpy as np
import random
import copy
import time
# import winsound
# import matplotlib
# import matplotlib.pyplot as plt


# Student name: Antonios Glioumpas.
class tsp_solver:

    # For the TSP: Traveling Salesperson Problem we need a (n x n) distance matrix
    # from which we can derive the number of cities
    class TSP:
        def __init__(self, matrix):
            self.matrix = matrix            # distance matrix
            self.n_cities = len(matrix)     # number of cities

    class Individual:
        def __init__(self, alpha, order):
            self.order = order              # Order with which the cities will be visited
            self.alpha = alpha              # Probability of applying a mutation operator to an individual

    class Parameters:
        def __init__(self, lam, mu, k, alpha_decr,num_of_mut):
            self.lam = lam	# lamda, λ : size of the population number of rand_individuals/candidate solutions
            self.mu = mu    # μ offspring are created by the application of variation operators
            self.k = k	    # for k tournament selection
            self.alpha_decr = alpha_decr  # the percentage drop of alpha for every evolutionary cycle
            self.num_of_mut = num_of_mut
   

    # Function name: "rand_individual" 
    # Creates an Individual with random city order and alpha value
    def rand_individual(self,tsp):
        order = np.random.permutation(tsp.n_cities)         # Will produce order = [0 1 2 ... n_cities]
        alpha = max(0.01, 0.5 + 0.01 * np.random.randn())   # Probability of applying a mutation operator to an individual
                                                            # max(1% , mu + sigma * np.random.randn()) , N(μ,σ^2)
        return self.Individual(alpha, order)        

    # Function name: "initialize"
    # Performs population initialization
    # returns a list of lamda (λ) random individuals
    def initialize(self,tsp, lam):
        return [self.rand_individual(tsp) for i in range(lam)] 

    # Function name: "selection"
    # Performs K tournament selection
    # Returns the best of the k randomly selected individuals, 
    # the one with the smallest covered distance
    def selection(self,tsp, popul, k):
        selected = np.random.choice(popul, k)
        fitnesses = [self.fitness(tsp, ind) for ind in selected]
        return selected[fitnesses.index(min(fitnesses))]

    # Function name: "recombination"
    # Plays the role of the Order Crossover operator (p.73 Eiben & Smith)
    # Returns a new individual (child) out of the inputed p1, p2 members of the population (parents)
    def recombination(self,tsp, p1, p2):
        segment_start = random.randint(0, tsp.n_cities - 1)
        segment_stop = random.randint(0, tsp.n_cities - 1)
        
        # in case segment_stop < segment_start, swap their values
        if segment_stop < segment_start:
            temp = segment_start
            segment_start = segment_stop
            segment_stop = temp

        segment = set(p1.order[segment_start:segment_stop+1])
        new_order = list(p1.order)
        new_index = segment_stop + 1
        p2_index = segment_stop + 1

        for i in range(0, tsp.n_cities):
            if new_index >= len(new_order):
                new_index = 0
            if p2_index >= len(p2.order):
                p2_index = 0
            if p2.order[p2_index] not in segment:
                new_order[new_index] = p2.order[p2_index]
                new_index += 1
            p2_index += 1

        # Calculate the alpha value for the offspring individual
        # depending on the values of the parents
        beta = 2 * random.random() - 0.5    
        alpha = p1.alpha + beta * (p2.alpha - p1.alpha)                                                
        return self.Individual(alpha, new_order)     
        
    #Function name: "replace"
    # Used in the "mutate_intervals" function
    # Inputs:  source -> an individual's order of cities 
    #          indexA -> list containing the start/end indices of the first interval
    #          indexB -> list containing the start/end indices of the second interval
    # Returns: a new individual order in which two interval have been swapped 
    #          according to the indexes of the input
    def replace(self,source, indexA, indexB):
        source = list(source)
        if indexB[1] < indexA[0]:
            temp = indexA
            indexA = indexB
            indexB = temp
        newList = source[:indexA[0]] + source[indexB[0]:indexB[1]]
        newList += source[indexA[1]:indexB[0]] + source[indexA[0]:indexA[1]]
        newList += source[indexB[1]:]
        #print("interval swap complete")
        return newList

    # Function name: "simple_swap"
    #  Used in the "mutate_intervals" function
    #  Performs a swap mutation of single cities on an individual's order
    def simple_swap(self,ind):
        ind_length = len(ind.order)
        a = random.choice(range(ind_length))
        b = random.choice(range(ind_length))  
        temp = ind.order[a]
        ind.order[a] = ind.order[b]
        ind.order[b] = temp
        #print("Simple swap complete")
        return ind.order

    # Function name: "mutate_intervals"    
    #  Performs: 
    #     - interval swap mutation using the "replace function" or
    #     - simple swap mutation of two cities in the individual's order in case 
    #       interval_length > ind_length or if the limits of the two intervals-to-be-swapped overlap
    def mutate_intervals(self,tsp,ind,num_of_iter,alpha_decr,interval_length):
        # print(ind.order)
        if random.random() < ind.alpha - num_of_iter*alpha_decr:   # As the number of iterations increases, 
            ind_length = len(ind.order)
            
            if interval_length < ind_length:
                int1_start = random.choice(range(ind_length-interval_length))
                int1_end = int1_start + interval_length  
                
                int2_start = random.choice(range(ind_length-interval_length))
                int2_end = int2_start + interval_length

                # Check if the limits of the two intervals-to-be-swapped overlap
                if (int2_start > int1_end) or (int2_end < int1_start):
                    ind.order = self.replace(ind.order, [int1_start,int1_end], [int2_start,int2_end])
                else:
                    ind.order = self.simple_swap(ind)  
            else: 
                ind.order = self.simple_swap(ind)      
        #else: #print("Mutation didn't take place because random.random() > ind.alpha - num_of_iter*alpha_decr") 
        #print(ind.order)                
        return ind

    # Function name: "fitness"
    # Calculates the fitness value for an individual using the distance matrix of the tsp
    # This function plays the role of the objective function
    # It calculates the total route length of an individual/candidate solution 
    def fitness(self,tsp, ind):
        value = 0
        prev_i = ind.order[len(ind.order)-1]  # last element/city of order
        for i in ind.order:
            value += tsp.matrix[prev_i][i]
            prev_i = i
        return value

    # Function name: "cost_change"
    # Used in the "two_opt" function
    # Calculates the fitness difference of an individual's order if the two-opt local search algorithm is applied
    def cost_change(self,tsp, n1, n2, n3, n4):
        return tsp.matrix[n1][n3] + tsp.matrix[n2][n4] - tsp.matrix[n1][n2] - tsp.matrix[n3][n4]

    # Function name: "two_opt"
    # Returns a new order for an individual, improved by 2-opt local search algorithm
    def two_opt(self,ind, tsp):
        best = ind.order
        improved = True
        count = 0
        while improved and (count<10):
            improved = False
            for i in range(1, len(ind.order) - 2):
                for j in range(i + 1, len(ind.order)):
                    if j - i == 1: continue
                    if self.cost_change(tsp, best[i - 1], best[i], best[j - 1], best[j]) < 0:
                        best[i:j] = best[j - 1:i - 1:-1]
                        improved = True
            ind.order = best
            count += 1
        return best  

    # Function name: "elimination"
    # First the popul+offspring list is sorted in ascending order
    # lamda x is the sorting criterion, here it's the fitness value of an individual
    # The method returns a list of length lam which contains the first lam individuals, that is,
    # the individuals with the lowest fitness values, an elistist approach
    def elimination(self,tsp, popul, offspring, lam):
        return sorted(popul + offspring, key=lambda x: self.fitness(tsp, x), reverse=False)[:lam]  

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here begin.
        t_start = time.time()
        timestamps = []
        best_fitness_list = []
        mean_fitness_list = []

        tsp = self.TSP(distanceMatrix)
        p = self.Parameters(150, 150, 4, 0.0001,10) # lamda, mu, k, alpha_decr, num_of_mut
       
        # Initialize the population in order to contain lam random individuals
        population = self.initialize(tsp, p.lam)

        # Counter measuring the number of iterations of the while loop
        current_iter = 0
        
        prev_fit = float('inf')
        
        # Counter used as a convergence criterion
        convergence = 0

        while convergence < 80:  
            # Your code here.
            # Recombination
            offspring = list()

            # Create a new list with mu offspring from the population through Recombination and Mutation 
            for jj in range(p.mu):
                p1 = self.selection(tsp, population, p.k)
                p2 = self.selection(tsp, population, p.k)
                off = self.recombination(tsp, p1, p2)
                offspring.append(off)
                self.mutate_intervals(tsp,offspring[jj],current_iter,p.alpha_decr,10)
            
            # Insert mutated clones of the parents into the offspring list
            for ind in population:
                offspring.append(self.mutate_intervals(tsp,copy.deepcopy(ind),current_iter,p.alpha_decr,10))
            
            # Perform 2-opt local search algorithm for the first individual of the offspring list_of_max_values
            # Note: The two_opt function could be applied on the entire population but the computational cost would be much greater  
            #       and each iteration would need a lot more time
            offspring[0].order = self.two_opt(offspring[0],tsp)
            # This step can be improved by selecting a random member of the population instead of the first one.

            # The survivors of the populations will be the lam best individuals,
            # the ones with the lowest total route length
            population = self.elimination(tsp, population, offspring, p.lam)

            # Create a list with the fitness values of the surviving individuals of the population
            fitnesses = [self.fitness(tsp, x) for x in population]
            
            best_ind = np.array(population[fitnesses.index(min(fitnesses))].order)
            best_fit = min(fitnesses)
            print("Mean fit.:", np.mean(fitnesses), " | Best fit.:", min(fitnesses), " | Iteration:",current_iter)

            t_now = time.time()
            timestamps.append(t_now - t_start)


            best_fitness_list.append(min(fitnesses))
            #mean_fitness_list.append(np.mean(fitnesses))

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(np.mean(fitnesses), min(fitnesses), best_ind)
            current_iter += 1

            # If the difference between two consecutive population-best fitness values 
            # is less than 0.0000001%, the convergence counter increases otherwise it's reset
            if (prev_fit - best_fit) / best_fit < 0.00000001:
                convergence += 1
            else:
                convergence = 0
            prev_fit = best_fit

            if timeLeft < 0:
                break

        # Block of code used to prepare the data for plotting the fitness results:       
        # fig, ax = plt.subplots()
        # t = np.array(timestamps)
        # best_fitness_array = np.array(best_fitness_list)
        # #mean_fitness_array = np.array(mean_fitness_list)
        # #ax.plot(t, mean_fitness_array, label='mean fitness')
        # ax.plot(t, best_fitness_array, label='best fitness')
        # ax.set(xlabel='time (s)', ylabel='fitness value')
        # ax.legend()
        # fig_name = time.strftime("%Y%m%d%H%M%S",time.gmtime(time.time())) + 'Populations best fitness graph' + '.png'
        # fig.savefig(fig_name)

        # Your code here end.
        return best_fit

