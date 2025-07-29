from deap import base, creator
import os

def read_file(file_name):
    
    with open (file_name) as f:
        print(f.read())

# read_file("knapsack-data/10_269")

if os.path.isdir("knapsack-data"):
    print("does exists")
else:
    print("nope")

# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMin)





