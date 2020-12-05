import numpy as np
import random
import copy
import Reporter
import winsound
import time
import matplotlib
import matplotlib.pyplot as plt

# For the TSP: Traveling Salesperson Problem we need a (n x n) distance matrix
# from which we can derive the number of cities
class TSP:
    def __init__(self, matrix):
        self.matrix = matrix            # distance matrix
        self.n_cities = len(matrix)     # number of cities


class Individual:
    def __init__(self, alpha, order):
        self.order = order              # Order with which the cities will be visited
        self.alpha = alpha


class Parameters:
    def __init__(self, lam, mu, k, its):
        self.lam = lam	# lamda, λ : size of the population number of rand_individuals/candidate solutions
        self.mu = mu    # μ offspring are created by the application of variation operators
        self.k = k	    # for k tournament selection
        self.its = its	# number of iterations


# 0-1-2
# 1-2-0
def rand_individual(tsp):
    order = np.random.permutation(tsp.n_cities)         # Will produce order = [0 1 2 ... n_cities]
    alpha = max(0.01, 0.2 + 0.02 * np.random.randn())   # Probability of applying a mutation operator to an individual
    return Individual(alpha, order)


# For a 1-4-2-3 path representation
# This method plays the role of the objective function
# It calculates the total route length of an individual/candidate solution 
def fitness(tsp, ind):
    value = 0
    prev_i = ind.order[len(ind.order)-1]  # last element/city of order
    for i in ind.order:
        value += tsp.matrix[prev_i][i]
        prev_i = i
    return value


def mutate(ind):
    # Swap mutation
    if random.random() < ind.alpha:
        ind_length = len(ind.order)
        a = random.choice(range(ind_length))
        b = random.choice(range(ind_length))  # Can be the same as a, maybe test performance in ind phase.
        temp = ind.order[a]
        ind.order[a] = ind.order[b]
        ind.order[b] = temp
    return ind


def elimination(tsp, popul, offspring, lam):
    return sorted(popul + offspring, key=lambda x: fitness(tsp, x), reverse=False)[:lam]


# K tournament selection
# return the best of the k individuals, 
# the one with the smallest covered distance
def selection(tsp, popul, k):
    selected = np.random.choice(popul, k)
    fitnesses = [fitness(tsp, ind) for ind in selected]
    return selected[fitnesses.index(min(fitnesses))]

# Order Crossover operator (p.73 Eiben & Smith)
def recombination(tsp, p1, p2):
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

    beta = 2 * random.random() - 0.5
    alpha = p1.alpha + beta * (p2.alpha - p1.alpha)
    return Individual(alpha, new_order)     # Α new individual is created from p1, p2


def initialize(tsp, lam):
    return [rand_individual(tsp) for i in range(lam)]   # λ, lamda individuals will be created


# Modify the class name to match your student number.
class r0827913:

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
        tsp = TSP(distanceMatrix)
        p = Parameters(180, 180, 3, 80)
        # p = Parameters(100, 100, 3, 30) --> 35986.32394451325
        population = initialize(tsp, p.lam)
        its = 0
        prev_fit = float('inf')
        convergence = 0

        while convergence < 10:  # its < p.its for iteration test
            # Your code here.
            # Recombination
            offspring = list()

            # Creates mu offspring from the population through Recombination and Mutation 
            for jj in range(p.mu):
                p1 = selection(tsp, population, p.k)
                p2 = selection(tsp, population, p.k)
                off = recombination(tsp, p1, p2)
                offspring.append(off)
                mutate(offspring[jj])

            # Insert a mutated version of the general population into the offspring list
            for ind in population:
                offspring.append(mutate(copy.deepcopy(ind)))

            population = elimination(tsp, population, offspring, p.lam)
            fitnesses = [fitness(tsp, x) for x in population]
            
            best_ind = np.array(population[fitnesses.index(min(fitnesses))].order)
            best_fit = min(fitnesses)
            print("Mean fit:", np.mean(fitnesses), " | Best fit:", min(fitnesses))
            #print("Best fit:", min(fitnesses))

            t_now = time.time()
            timestamps.append(t_now - t_start)

            best_fitness_list.append(min(fitnesses))
            mean_fitness_list.append(np.mean(fitnesses))


            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(np.mean(fitnesses), min(fitnesses), best_ind)
            its += 1

            if (prev_fit - best_fit) / best_fit < 0.0001:
                convergence += 1
            else:
                convergence = 0
            prev_fit = best_fit

            if timeLeft < 0:
                break

        # Your code here end.
       
        # Segment used in order to plot the results:       
        # fig, ax = plt.subplots()
        # t = np.array(timestamps)
        # best_fitness_array = np.array(best_fitness_list)
        # mean_fitness_array = np.array(mean_fitness_list)
        # ax.plot(t, mean_fitness_array, label='mean fitness')
        # ax.plot(t, best_fitness_array, label='best fitness')
        # ax.set(xlabel='time (s)', ylabel='fitness value')
        # ax.legend()
        # fig_name = time.strftime("%Y%m%d%H%M%S",time.gmtime(time.time())) + '.png'
        # fig.savefig(fig_name)
        
        return best_fit


bests = []
for _ in range(20):
    test = r0827913()
    bests.append(test.optimize("tour29.csv"))   # The "optimize" method returns best_fit = min(fitnesses)
    #winsound.Beep(440, 2000)


# Segment used in order to plot the results:

# fig, ax = plt.subplots()
# t = np.arange(0, len(bests))
# results_array = np.array(bests)
# print("Variance of best fitnesses: {}".format(str(np.var(results_array))))
# ax.plot(t, results_array)
# ax.set(xlabel='Experiment No.', ylabel='best fitness value')
# fig.savefig('experiment.png')