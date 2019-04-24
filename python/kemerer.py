import operator
import math
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 7)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
# pset.addPrimitive(math.cos, 1)
# pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.renameArguments(ARG0='ID')
pset.renameArguments(ARG1='Language')
pset.renameArguments(ARG2='Hardware')
pset.renameArguments(ARG3='Duration')
pset.renameArguments(ARG4='KSLOC')
pset.renameArguments(ARG5='AdjFP')
pset.renameArguments(ARG6='RAWFP')

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
    mres = (abs(x[7] - func(x[0], x[1], x[2], x[3], x[4], x[5], x[6])) / x[7] for x in points)
    return math.fsum(mres) / len(points),

# proportion of estimates within n% of the actual
def pred(n, individual, points):
    # unimplemented
    return 0;

# mean absolute error
def mae(individual, points):
    func = toolbox.compile(expr=individual)
    aes = (abs(x[7] - func(x[0], x[1], x[2], x[3], x[4], x[5], x[6])) for x in points)
    return math.fsum(aes) / len(points),

toolbox.register("evaluate", mmre, points=[[1, 1,1,17,253.6,1217.1,1010,287    ],
                                           [2, 1,2,7, 40.5, 507.3, 457, 82.5   ],
                                           [3, 1,3,15,450,  2306.8,2284,1107.31],
                                           [4, 1,1,18,214.4,788.5, 881, 86.9   ],
                                           [5, 1,2,13,449.9,1337.6,1583,336.3  ],
                                           [6, 1,4,5, 50,   421.3, 411, 84     ],
                                           [7, 2,4,5, 43,   99.9,  97,  23.2   ],
                                           [8, 1,2,11,200,  993,   998, 130.3  ],
                                           [9, 1,1,14,289,  1592.9,1554,116    ],
                                           [10,1,1,5, 39,   240,   250, 72     ],
                                           [11,1,1,13,254.2,1611,  1603,258.7  ],
                                           [12,1,5,31,128.6,789,   724, 230.7  ]])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
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

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 400, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof

if __name__ == "__main__":
    main()
