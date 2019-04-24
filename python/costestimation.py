import operator
import math
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

def estimate(file, ngen):

    attributes = []
    datas = []
    numerics = []

    with open(file + '.arff') as f:
        read_data = f.readlines()
    for x in range(len(read_data)):
        if read_data[x][0] == '@':
            if read_data[x].split()[0] == '@attribute':
                if read_data[x].split()[2] == 'numeric':
                    attributes.append(read_data[x].split()[1])
                numerics.append(read_data[x].split()[2] == 'numeric')
            elif read_data[x] == '@data\n':
                break
    x += 1
    for y in range(x, len(read_data)):
        row = read_data[y].strip().split(',')
        a = []
        for z in range(len(row)):
            if numerics[z]:
                a.append(float(row[z]))
        datas.append(a)

    print(attributes)
    print(datas)
    print(numerics)

    def protectedDiv(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1

    pset = gp.PrimitiveSet("MAIN", len(attributes) - 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    # pset.addPrimitive(math.cos, 1)
    # pset.addPrimitive(math.sin, 1)
    # pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
    for x in range(len(attributes)):
        exec('pset.renameArguments(ARG' + str(x) + '=\''+attributes[x]+'\')')

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    # mean magnitude of relative error
    def mmre(individual, points):
        func = toolbox.compile(expr=individual)
        mres = (abs(x[len(x) - 1] - func(*x[0:len(x)-1])) / x[len(x) - 1] for x in points)
        return math.fsum(mres) / len(points),

    # proportion of estimates within n% of the actual
    def pred(n, individual, points):
        # unimplemented
        return 0,

    # mean absolute error
    def mae(individual, points):
        func = toolbox.compile(expr=individual)
        aes = (abs(x[len(x) - 1] - func(*x[0:len(x)-1])) for x in points)
        return math.fsum(aes) / len(points),

    toolbox.register("evaluate", mmre, points=datas)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6)) # default 17
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6)) # default 17

    random.seed(318)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, ngen, stats=mstats,
                                   halloffame=hof, verbose=True)
    print(str(hof[0]))
    return toolbox.compile(expr=hof[0])
