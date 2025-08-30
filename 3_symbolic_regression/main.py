from deap import base, creator, tools, algorithms, gp
import pandas as pd
import numpy as np
import sys
import random
import os
from sklearn.feature_selection import mutual_info_regression
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, log_loss
import gp_operator as op
import math
import operator
import graphviz



# GA Functions:

# Fitness calculation

def fitness_function(input, output,toolbox, ind):

    func = toolbox.compile(expr=ind)
    # Pass the inputs through the individual
    pred = [func(x) for x in input]
    
    correct_res = 0
        
    mse = mean_squared_error(output, pred)
    return (mse,)
    


def custom_ea(pop, toolbox, mate_rate, mut_rate, ngen,elite_size =5, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"]
    logbook.header.extend(stats.functions.keys())
    # Evalutes the the intial population
    fitness = toolbox.map(toolbox.evaluate, pop)
    for (ind, fit) in zip(pop, fitness):
        ind.fitness.values = fit

    for gen in range(ngen):

        elite = tools.selBest(pop, elite_size)
        elite = [toolbox.clone(ind) for ind in elite]

        # Select the next gen
        offspring = toolbox.select(pop, len(pop) - elite_size)

        # clone the selected ind
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < mate_rate:
                toolbox.mate(child1,child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mut_rate:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Find indivudals where their fitness has not been calculated yet
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness = toolbox.map(toolbox.evaluate, invalid_ind)
        for (ind, fit) in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        # Replace population with off spring
        pop[:] = offspring + elite

        halloffame.update(pop)

        if(stats != None):
            record = stats.compile(pop)
            logbook.record(gen = gen, nevals = len(invalid_ind), **record)
            print(logbook.stream)

    return pop, logbook
    
def symbolic_regression(input, output, seed):
    
    random.seed(seed)

    pset = gp.PrimitiveSet("main", 1)

    # Basic arithmetic operators
    pset.addPrimitive(op.add, 2)
    pset.addPrimitive(op.sub, 2)
    pset.addPrimitive(op.mult, 2)
    pset.addPrimitive(op.protected_div, 2)

    pset.addPrimitive(op.if_op, 4)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.tan, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(op.squared, 1)


    constants = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for c in constants:
        pset.addTerminal(c)

    pset.renameArguments(ARG0="x")

    toolbox = base.Toolbox()

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", fitness_function, input, output, toolbox)


    # #     # Add size restrictions
    # MAX_HEIGHT = 8
    # MAX_SIZE = 50
    # toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT))
    # toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT))
    # toolbox.decorate("mate", gp.staticLimit(key=len, max_value=MAX_SIZE))
    # toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=MAX_SIZE))

    pop = toolbox.population(100)

    hall_of_fame = tools.HallOfFame(1)

    stat = tools.Statistics()
    stat.register("mean", lambda pop: np.mean([ind.fitness.values[0] for ind in pop]))
    stat.register("min", lambda pop: min([ind.fitness.values[0] for ind in pop]))  # Change max to min
    stat.register("mean_of_5_best", lambda pop: np.mean(sorted([ind.fitness.values[0] for ind in pop], reverse=False)[:5]))

    pop, logbook = custom_ea(pop, toolbox, 0.8, 0.3, 100, stats=stat,halloffame=hall_of_fame)

    return hall_of_fame[0], logbook.select("mean_of_5_best")

def run_with_3_seeds(input, output):
    seeds = [1, 10, 100]

    results_of_seeds = {}

    for seed in seeds:

        input_c= input.copy()
        output_c = output.copy()

        (best_ind, fit_over_gen)= symbolic_regression(input_c, output_c, seed)


        results_of_seeds[str(seed)] = {
        "best_ind": best_ind,
        "fitness": best_ind.fitness.values,
        "tree_size": len(best_ind),
        "tree_depth": best_ind.height,
        "fitness_over_gen": [float(x) for x in fit_over_gen]
        }


    return results_of_seeds


def target_function(x):
    if( x > 0):
        return math.sin(x) + 1/x
    else:
        return 2 * x + x ** 2  + 3

input = [3, 2, 1,0.1, 0.5, 1.0, 2.0, 0.0, -0.5, -1.0, -2.0, -3.0]

output= [ target_function(x) for x in input]

result_seed = run_with_3_seeds(input,output)

pd.DataFrame.from_dict(result_seed , orient='index').to_csv("results.csv")