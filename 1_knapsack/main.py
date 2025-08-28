from deap import base, creator, tools, algorithms
import numpy as np
import sys
import random
import os
import pandas as pd

# Utility Function
def read_file(file_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(base_dir, file_name)    

    file_contents = ""
    data = []

    with open (target_path) as f:
        file_contents = f.readlines()

    for line in file_contents:
        data.append(tuple(map(int, (line.split()))))

    return data


def get_item_att(items, ind):
    value = 0
    weight = 0

    for i, att in enumerate(ind):
        if(att == 1):
           (value_ind, weight_ind) = items[i]
           value += value_ind
           weight += weight_ind 

    return (value, weight)



# GA Functions:

#Fitness calculation
def calc_fitness(max_weight, items, ind):
    
    value = 0
    weight = 0

    for i, att in enumerate(ind):
        if(att == 1):
           (value_ind, weight_ind) = items[i]
           value += value_ind
           weight += weight_ind 

    if weight > max_weight:
        return (max_weight - value,)
    else:
        return (value,)


def mut_ind(ind):
    id = random.randrange(len(ind))
    ind[id] = 1 - ind[id]
    return (ind,)
        
def knapsack_problem(items, seed):
    random.seed(seed)
    #` Get the number of items, and the bag cap
    (num_items, bag_cap)= items.pop(0)

    IND_SIZE = num_items
    toolbox = base.Toolbox()

    # Set max length of individual to the number of items 
    # Set the attribute of indivudallto be randomly either 0, or 1
    toolbox.register("attribute", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mut_ind)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", calc_fitness, bag_cap, items)

    pop = toolbox.population(50)

    hall_of_fame = tools.HallOfFame(1)

    stat = tools.Statistics()
    stat.register("mean", np.mean)
    stat.register("max", np.max)
    stat.register("mean_of_5_best", lambda x: np.mean(sorted([ind.fitness.values[0] for ind in x], reverse=True)[:5]))

    pop, logbook = custom_ea(pop, toolbox, 0.8, 0.4, 100, stats=stat,halloffame=hall_of_fame)

    return (hall_of_fame[0], logbook.select("mean_of_5_best"))

items_269 = read_file("knapsack-data/10_269")
items_10000 = read_file("knapsack-data/23_10000")
items_995 = read_file("knapsack-data/100_995")


def custom_ea(pop, toolbox, mate_rate, mut_rate, ngen, elite_size =3, stats=None, halloffame=None, verbose=__debug__):
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






def run_with_5_seeds(items):
    seeds = [1, 10, 100, 1000, 10000]

    results_of_seeds = {}

    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    for seed in seeds:

        items_c = items.copy()

        (best_ind, mean) = knapsack_problem(items_c, seed)

        (max_value, max_weight) = get_item_att(items_c, best_ind)

        results_of_seeds[str(seed)] = {
        "max_value": max_value,
        "max_weight": max_weight,
        "mean_over_gen": [float(x) for x in mean],
        }

    return results_of_seeds


res_items_269 = run_with_5_seeds(items_269)
res_items_10000 = run_with_5_seeds(items_10000)
res_items_995 = run_with_5_seeds(items_995)

df_items_269 = pd.DataFrame.from_dict(res_items_269, orient='index')
df_items_10000 = pd.DataFrame.from_dict(res_items_10000 , orient='index')
df_items_995 = pd.DataFrame.from_dict(res_items_995 , orient='index')

df_items_269.to_csv("results_269.csv")
df_items_10000.to_csv("results_10000.csv")
df_items_995.to_csv("results_995.csv")


