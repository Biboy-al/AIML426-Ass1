from deap import base, creator, tools, algorithms
import pandas as pd
import numpy as np
import sys
import random
import os
import time
from sklearn.feature_selection import mutual_info_regression
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


# Utility Function
def read_file(file_name, type=int, drop_n=0):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(base_dir, file_name)    

    file_contents = ""
    data = []


    with open (target_path) as f:
        file_contents = f.readlines()

    file_contents = file_contents[drop_n:]

    for line in file_contents:
        data.append(tuple(map(type, (line.split()))))

    return data


# GA Functions:

# Fitness calculation

def fitness_function(dataset, features, ind):
    
    features_to_calc = list(zip(ind, [feat for feat in features]))

    columns_to_keep = [feat[1] for feat in tuple(features_to_calc) if feat[0] != 0]

    target = dataset.iloc[:, -1]

    important_features = dataset[columns_to_keep]

    X = important_features.values
    n_features = X.shape[1]

    if n_features < 2:
        return (0.0, 1.0) 
    

    classifier = KNeighborsClassifier(n_neighbors=2)

    scores = cross_val_score(classifier, important_features, target, cv=5, scoring='accuracy')
    return (scores.mean(), len(columns_to_keep) / len(features))
    

def mut_ind(ind):

    id = random.randrange(len(ind))
    ind[id] = 1 - ind[id]
    return (ind,)

def custom_ea(pop, toolbox, mate_rate, mut_rate, ngen, elite_size =3, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"]
    logbook.header.extend(stats.functions.keys())
    # Evalutes the the intial population
    fitness = toolbox.map(toolbox.evaluate, pop)
    for (ind, fit) in zip(pop, fitness):
        ind.fitness.values = fit

    pop = tools.selNSGA2(pop, len(pop))

    for gen in range(ngen):

        # Select the next gen
        offspring = toolbox.select(pop, len(pop))

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

        combined = pop + offspring
        pop = tools.selNSGA2(combined, len(pop))

        halloffame.update(pop)

        if(stats != None):
            record = stats.compile(pop)
            logbook.record(gen = gen, nevals = len(invalid_ind), **record)
            print(logbook.stream)

    return pop, logbook
        
def feature_selection(features, dataset, seed, filter):

    
    random.seed(seed)

    IND_SIZE = len(features)
    toolbox = base.Toolbox()

    # Set max length of individual to the number of items 
    # Set the attribute of indivudallto be randomly either 0, or 1
    toolbox.register("attribute", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mut_ind)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("evaluate", fitness_function, dataset, features)

    pop = toolbox.population(50)

    hall_of_fame = tools.HallOfFame(1)

    stat = tools.Statistics()
    stat.register("mean", np.mean)
    stat.register("max", max)


    start_time = time.time()
    pop, logbook = custom_ea(pop, toolbox, 0.8, 0.5, 100, stats=stat,halloffame=hall_of_fame)
    end_time = time.time()


    best = hall_of_fame[0]
    print("Best Ind:", best)
    print("Fitness:", best.fitness.values)

    return hall_of_fame[0], logbook, (end_time - start_time)


def find_class_acc(subset, features, dataset):
    columns_to_keep = [features[i] for i, selected in enumerate(subset) if selected != 0]

    target = dataset.iloc[:, -1]

    important_features = dataset[columns_to_keep]

    classifier = KNeighborsClassifier(n_neighbors=2)

    scores = cross_val_score(classifier, important_features, target, cv=5, scoring='accuracy')
    return scores.mean()


def run_with_5_seeds(features, dataset, filter=True):
    seeds = [1, 10, 100, 1000, 10000]

    results_of_seeds = {}

    creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    for seed in seeds:

        features_c= features.copy()
        dataset_c = dataset.copy()

        (best_ind, logbook, compute_time) = feature_selection(features_c, dataset_c, seed, filter)


        results_of_seeds[str(seed)] = {
        "best_subset": best_ind,
        "compute_time": compute_time,
        "accuracy": find_class_acc(best_ind, features, dataset)
        }

    return results_of_seeds


def give_column_names(dataset):
    num_features = len(dataset.columns) - 1

    features = [f"col_{i}" for i in range(num_features)]

    new_column_names = [f"col_{i}" for i in range(num_features)]
    new_column_names.append("class")  # Last column as target

    dataset.columns = new_column_names

    return dataset, features
    


# Breast Cancer Dataset
dataset_v = pd.read_csv("data/vehicle/vehicle.dat", header=None, delim_whitespace=True)
dataset_v, features_v = give_column_names(dataset_v)
label_encoder = LabelEncoder()
dataset_v['class'] = label_encoder.fit_transform(dataset_v['class'])


dataset_m = pd.read_csv("data/musk/clean1.data", header=None)

dataset_m, features_m = give_column_names(dataset_m)
label_encoder = LabelEncoder()
dataset_m['col_0'] = label_encoder.fit_transform(dataset_m['col_0'])
label_encoder = LabelEncoder()
dataset_m['col_1'] = label_encoder.fit_transform(dataset_m['col_1'])


result_seed_bc = run_with_5_seeds(features_m, dataset_m)

result_seed_bc_df = pd.DataFrame.from_dict(result_seed_bc, orient='index')
result_seed_bc_df.to_csv("results_m.csv")