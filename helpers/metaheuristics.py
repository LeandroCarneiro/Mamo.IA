import numpy as np
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import cross_val_score
import numpy as np
from deap import base, creator, tools, algorithms
import random
from pyswarm import pso

def run_pso_with_progress(X, Y, estimator, n_features, swarmsize=30, maxiter=100, threshold=0.7):
    lb = [0]*n_features
    ub = [1]*n_features
    progress = []

    # Create a scorer for Cohen's Kappa
    kappa_scorer = make_scorer(cohen_kappa_score)

    def objective_with_progress(weights, est, X_, Y_):
        Xw = X_ * weights
                
        # Use it in cross_val_score
        fit = 1 - cross_val_score(est, Xw, Y_, cv=5, scoring=kappa_scorer).mean()
        progress.append(fit)
        
        if len(progress) % 10 == 0:
            print(f"Eval {len(progress)}: best fitness so far = {min(progress):.4f}")
        return fit

    best_pos, best_fit = pso(
        objective_with_progress,
        lb, ub,
        args=(estimator, X, Y),
        swarmsize=swarmsize,
        maxiter=maxiter
    )

    mask = best_pos > threshold
    selected_features = np.where(mask)[0].tolist()
    return best_pos, best_fit, progress, selected_features

def run_ga_with_progress(X, Y, estimator, n_features,
                        pop_size=20, n_generations=100, threshold=0.7,
                        cx_prob=0.5, mut_prob=0.2, tournament_size=3):
   
    progress = []
    
    # Create a scorer for Cohen's Kappa
    kappa_scorer = make_scorer(cohen_kappa_score)
    
    # DEAP setup
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # Gene initialization (random float between 0 and 1)
    toolbox.register("attr_float", random.random)
    
    # Individual initialization
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                    toolbox.attr_float, n_features)
    
    # Population initialization
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate_individual(individual):
        """Evaluate an individual using cross-validation with Cohen's Kappa"""
        weights = np.array(individual)
        Xw = X * weights
        
        # Use cross-validation with Cohen's Kappa
        kappa_scores = cross_val_score(estimator, Xw, Y, cv=5, scoring=kappa_scorer)
        fitness = 1 - kappa_scores.mean()  # Minimize (1 - kappa)
        
        progress.append(fitness)
        
        if len(progress) % 10 == 0:
            print(f"Eval {len(progress)}: best fitness so far = {min(progress):.4f}")
        
        return (fitness,)
    
    # Register genetic operators
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    
    # Create initial population
    pop = toolbox.population(n=pop_size)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # Evolution loop
    for generation in range(n_generations):
        print(f"Generation {generation + 1}/{n_generations}")
        
        # Select parents
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Apply mutation
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                # Clip values to [0, 1] range
                for i in range(len(mutant)):
                    mutant[i] = max(0, min(1, mutant[i]))
                del mutant.fitness.values
        
        # Evaluate offspring with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population
        pop[:] = offspring
    
    # Find best individual
    best_individual = tools.selBest(pop, 1)[0]
    best_fitness = best_individual.fitness.values[0]
    
    # Select features based on threshold
    mask = np.array(best_individual) > threshold
    selected_features = np.where(mask)[0].tolist()
    
    # Clean up DEAP classes to avoid conflicts in future runs
    del creator.FitnessMin
    del creator.Individual
    
    return list(best_individual), best_fitness, progress, selected_features