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



# GA Functions:

# Fitness calculation

def fitness_function(input, output,toolbox, ind):

    func = toolbox.compile(expr=ind)
    pred = [func(x) for x in input]
    
    correct_res = 0

    for (p, o) in zip(pred, output):
        if p == o:
            correct_res += 1
        

    # mse = log_loss(output, pred)
    return (correct_res/len(output),)
    

    
def feature_selection(input, output, seed):

    pset = gp.PrimitiveSet("main", 1)
    pset.addPrimitive(op.add, 2)
    pset.addPrimitive(op.sub, 2)
    pset.addPrimitive(op.mult, 2)
    pset.addPrimitive(op.protected_div, 2)
    pset.addPrimitive(op.if_op, 4)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(op.squared, 1)
    pset.addTerminal(3)
    pset.addTerminal(2)
    pset.addTerminal(1)
    pset.renameArguments(ARG0="x")

    toolbox = base.Toolbox()

    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", fitness_function, input, output, toolbox)

    pop = toolbox.population(50)

    hall_of_fame = tools.HallOfFame(1)

    stat = tools.Statistics(lambda ind: ind.fitness.values[0])
    stat.register("mean", np.mean)
    stat.register("max", max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, 0.8, 0.5, 100, stats=stat,halloffame=hall_of_fame)

    for i, ind in enumerate(pop):
        print(f"Indiviudal {i} Fitness: {ind.fitness.values}")

    best = hall_of_fame[0]
    print("Best Ind:", best)
    print("Fitness:", best.fitness.values)



input = [0.1, 0.5, 1.0, 2.0, 0.0, -0.5, -1.0, -2.0]
output= [10.09983, 2.47943, 1.84147, 1.40930, 3.0, 2.25, 2.0, 3.0]

feature_selection(input,output, 100000)