from deap import base, creator, tools, algorithms
import numpy as np
import random
import os

class Item:
    def __init__(self, value, weight, used=True):
        self.value = value
        self.weight = weight
        self.used = used

    def mut_used(self):
        self.used = not(self.used)

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
        
        # If item is not being used 
        if (not(item.used)): 
            continue

        value += item.value
        weight += item.weight

    if weight > max_weight:
        return (0,)
    else :
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

    random.sample(ind, 1)[0].mut_used()
    return (ind,)
        

items = read_file("knapsack-data/10_269")

(num_items, bag_cap)= items.pop(0)

items = [Item(value, weight) for (value, weight) in items]

pop = gen_pop(items, 0.7)


IND_SIZE = len(items) 
toolbox = base.Toolbox()

# Create the types:
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Set max length of individual to the number of items 

toolbox.register("attribute", gen_pop, items, 0.7)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mut_ind)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", calc_fitness, bag_cap)

pop = toolbox.population(50)

# stats = tools.Statistics()
# stats.register("mean", np.mean)
# stats.register("max", np.max)

hall_of_fame = tools.HallOfFame(1)

pop = algorithms.eaSimple(pop, toolbox, 0.8, 0.5, 50, halloffame=hall_of_fame)

for item in pop[0]:
    print(item[0].weight)



