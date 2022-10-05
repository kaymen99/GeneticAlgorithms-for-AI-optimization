from utils.elitism import eaSimpleWithElitism
from utils.models import Model

from deap import base
from deap import creator
from deap import tools

import random
import numpy

import matplotlib.pyplot as plt
import seaborn as sns

class GASelector:

    # Genetic Algorithm parameters:
    POPULATION_SIZE = 50
    P_CROSSOVER = 0.9  # crossover probability
    P_MUTATION = 0.2   # mutation probability
    MAX_GENERATIONS = 50
    HALL_OF_FAME_SIZE = 5

    FEATURE_PENALTY_FACTOR = 0.001

    def __init__(self, data_x, data_y, model, metric, randomSeed=42):
        self.model = Model(
            data_x,
            data_y,
            model,
            metric,
            randomSeed
        )
        self.toolbox = base.Toolbox()
        random.seed(randomSeed)

        # initialize the Genetic algorithm
        self.intialize_ga()
    
    def intialize_ga(self):
        # define a single objective, maximizing fitness strategy:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # create the Individual class based on list:
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # create an operator that randomly returns 0 or 1:
        self.toolbox.register("zeroOrOne", random.randint, 0, 1)

        # create the individual operator to fill up an Individual instance:
        self.toolbox.register("individualCreator", tools.initRepeat, creator.Individual, self.toolbox.zeroOrOne, len(self.model))

        # create the population operator to generate a list of individuals:
        self.toolbox.register("populationCreator", tools.initRepeat, list, self.toolbox.individualCreator)

        self.toolbox.register("evaluate", self.fitness)

        # genetic operators:
        # Tournament selection with tournament size of 2:
        self.toolbox.register("select", tools.selTournament, tournsize=2)

        # Single-point crossover:
        self.toolbox.register("mate", tools.cxTwoPoint)

        # Flip-bit mutation:
        # indpb: Independent probability for each attribute to be flipped
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/len(self.model))

    # fitness calculation
    def fitness(self, individual):
        numFeaturesUsed = sum(individual)
        if numFeaturesUsed == 0:
            return 0.0,
        else:
            accuracy = self.model.getMeanAccuracy(individual)
            return accuracy - self.FEATURE_PENALTY_FACTOR * numFeaturesUsed,  # 
    
    def plot_stat(self, max_values, mean_values):
        sns.set_style("whitegrid")
        plt.plot(max_values, color='red')
        plt.plot(mean_values, color='green')
        plt.xlabel('Generation')
        plt.ylabel('Max / Average Fitness')
        plt.title('Max and Average fitness over Generations')
        plt.show()
    
    def log(self, hof):
        print("- Best solutions are:")
        for i in range(self.HALL_OF_FAME_SIZE):
            fitness_value = hof.items[i].fitness.values[0]
            mean_accuracy = self.model.getMeanAccuracy(hof.items[i])
            print(f'{i}: {hof.items[i]}\n')
            print(f"fitness = {fitness_value}, accuracy = {mean_accuracy}, features = {sum(hof.items[i])}\n")

    # Genetic Algorithm flow
    def run(self):
        # create initial population
        population = self.toolbox.populationCreator(n=self.POPULATION_SIZE)

        # statistics object
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", numpy.max)
        stats.register("avg", numpy.mean)

        # define the hall-of-fame object
        hof = tools.HallOfFame(self.HALL_OF_FAME_SIZE)

        # Genetic Algorithm flow with hof 
        population, logbook = eaSimpleWithElitism(
            population, 
            self.toolbox, 
            cxpb=self.P_CROSSOVER, 
            mutpb=self.P_MUTATION,
            ngen=self.MAX_GENERATIONS, 
            stats=stats, 
            halloffame=hof, 
            verbose=True
            )

        # print best solution found
        self.log(hof)

        # plot statistics
        maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")
        self.plot_stat(maxFitnessValues, meanFitnessValues)
