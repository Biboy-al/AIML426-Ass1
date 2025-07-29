from deap import base, creator, tools, algorithms
import numpy as np
import random
import os



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
def calc_fitness(max_weight, ind):
    value = 0
    weight = 0


    for item in ind:
        value += item[0]
        weight += item[1]

    return (value * (max_weight/weight),)

#Genetic Operators

#Cross over

#Selection 

#Mutation
        

items = read_file("knapsack-data/10_269")

(num_items, bag_cap)= items.pop(0)

IND_SIZE = len(items) 

toolbox = base.Toolbox()
# Create the types:
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Set max length of individual to the number of items 

toolbox.register("attribute", random.sample, items, len(items))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", calc_fitness, bag_cap)

pop = toolbox.population(50)

stats = tools.Statistics()
stats.register("mean", np.mean)
stats.register("max", np.max)

hall_of_fame = tools.HallOfFame(1)

algorithms.eaSimple(pop, toolbox, 0.8, 0.5, 50, stats, hall_of_fame)



