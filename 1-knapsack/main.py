from deap import base, creator, tools, algorithms
import numpy as np
import sys
import random
import os

# class Item:
#     def __init__(self, value, weight, used=True):
#         self.value = value
#         self.weight = weight
#         self.used = used

#     def mut_used(self):
#         self.used = not(self.used)

# Utility Function
def read_file(file_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(base_dir, file_name)    

    file_contents = ""
    knapsack_data = []


    with open (target_path) as f:
        file_contents = f.readlines()

    for line in file_contents:
        knapsack_data.append(tuple(map(int, (line.split()))))

    return knapsack_data


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

# Generate inital Pop:
def gen_pop(items, prob_used):
    pop = items
    
    for item in pop:
        if random.random() > prob_used:
            item.mut_used()

    return pop

#Genetic Operators

#Cross over

#Selection 

#Mutation

def mut_ind(ind):

    id = random.randrange(len(ind))
    ind[id] = 1 - ind[id]
    return (ind,)
        
# Read the contents
items = read_file("knapsack-data/23_10000")

# Get the number of items, and the bag cap
(num_items, bag_cap)= items.pop(0)

# generate pops
# pop = gen_pop(items, 0.7)

IND_SIZE = len(items) 
toolbox = base.Toolbox()

# Create the types:
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

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

# stats = tools.Statistics()
# stats.register("mean", np.mean)
# stats.register("max", np.max)

hall_of_fame = tools.HallOfFame(1)

stat = tools.Statistics()
stat.register("mean", np.mean)
stat.register("max", max)

pop, logbook = algorithms.eaSimple(pop, toolbox, 0.8, 0.5, 100, stats=stat,halloffame=hall_of_fame)

for i, ind in enumerate(pop):
    print(f"Indiviudal {i} Fitness: {ind.fitness.values}")

best = hall_of_fame[0]
print("Best Ind:", best)
print("Fitness:", best.fitness.values)
