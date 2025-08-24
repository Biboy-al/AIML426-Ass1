from deap import base, creator, tools, algorithms
import pandas as pd
import numpy as np
import sys
import random
import os
from sklearn.feature_selection import mutual_info_regression
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

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


features = read_file("feature_selection_data/sonar/sonar.names", type=str, drop_n=1)


# GA Functions:

# Fitness calculation

def fitness_function(dataset, features, type_of_fitness, ind):
    
    features_to_calc = list(zip(ind, [feat[0] for feat in features]))

    columns_to_keep = [feat[1] for feat in tuple(features_to_calc) if feat[0] != 0]

    target = dataset.iloc[:, -1]

    important_features = dataset[columns_to_keep]

    X = important_features.values
    n_features = X.shape[1]

    if n_features < 2:
        return (0.0,) 
    
    if type_of_fitness == "filter":
        return filter_fitness(important_features)
    else:
        return wrapper_fitness(important_features, target)
    



def filter_fitness(important_features):

    total_mi = []

    for column in important_features.columns:

        y = important_features[column]
        x = important_features.drop(columns=[column])
        total_mi.append(mutual_info_regression(x, y, discrete_features='auto').mean())
    

    return ((sum(total_mi)/len(total_mi)) * len(important_features.columns),)


def wrapper_fitness(important_features, target):
    classifier = KNeighborsClassifier(n_neighbors=2)

    scores = cross_val_score(classifier, important_features, target, cv=5, scoring='accuracy')
    return (scores.mean(),)

    

#Genetic Operators

#Cross over

#Selection 

#Mutation

def mut_ind(ind):

    id = random.randrange(len(ind))
    ind[id] = 1 - ind[id]
    return (ind,)
        
def feature_selection(features, dataset, seed):

    IND_SIZE = len(features)
    toolbox = base.Toolbox()

#     # Create the types:
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
    toolbox.register("evaluate", fitness_function, dataset, features, "filter")

    pop = toolbox.population(50)

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



features = read_file("feature_selection_data/wbcd/wbcd.names", type=str, drop_n=1)

column_names = [f[0] for f in features]
column_names.append("class")

dataset = pd.read_csv("feature_selection_data/wbcd/wbcd.data", names=column_names)

feature_selection(features,dataset, 100000)