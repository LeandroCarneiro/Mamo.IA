import numpy as np
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import cross_val_score
import numpy as np
from deap import base, creator, tools, algorithms
import random
from pyswarm import pso

def run_pso_with_progress(X, Y, estimator, n_features,
                          swarmsize=50, maxiter=10, threshold=0.7):
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

def run_two_stage_pso(X, Y, estimator, n_features,
                      # Stage 1 parameters
                      stage1_swarmsize=30, stage1_maxiter=25, stage1_threshold=0.4,
                      # Stage 2 parameters  
                      stage2_swarmsize=25, stage2_maxiter=25, stage2_threshold=0.7,
                      # General parameters
                      verbose=True):
    if verbose:
        print("="*60)
        print("STARTING TWO-STAGE PSO FEATURE SELECTION")
        print("="*60)
    
    # STAGE 1: Coarse feature selection
    if verbose:
        print(f"\nSTAGE 1: Coarse Selection")
        print(f"- Original features: {n_features}")
        print(f"- Swarm size: {stage1_swarmsize}")
        print(f"- Max iterations: {stage1_maxiter}")
        print(f"- Threshold: {stage1_threshold}")
        print(f"shape of X: {X.shape}")
        print(f"shape of Y: {Y.shape}")
        print("-" * 40)
    
    stage1_weights, stage1_fitness, stage1_progress, stage1_features = run_pso_with_progress(
        X, Y, estimator, n_features,
        swarmsize=stage1_swarmsize,
        maxiter=stage1_maxiter,
        threshold=stage1_threshold
    )
    
    if verbose:
        print(f"\nSTAGE 1 RESULTS:")
        print(f"- Features selected: {len(stage1_features)}")
        print(f"- Best fitness: {stage1_fitness:.4f}")
        print(f"- Feature reduction: {n_features} → {len(stage1_features)} "
              f"({len(stage1_features)/n_features*100:.1f}%)")
    
    # Check if stage 1 selected any features
    if len(stage1_features) == 0:
        print("WARNING: Stage 1 selected no features! Lowering threshold...")
        # Retry with lower threshold
        stage1_weights, stage1_fitness, stage1_progress, stage1_features = run_pso_with_progress(
            X, Y, estimator, n_features,
            swarmsize=stage1_swarmsize,
            maxiter=stage1_maxiter,
            threshold=stage1_threshold * 0.7  # Lower threshold
        )
        
        if len(stage1_features) == 0:
            raise ValueError("No features selected even with lowered threshold!")
    
    # STAGE 2: Refined selection on reduced feature set
    if verbose:
        print(f"\nSTAGE 2: Refined Selection")
        print(f"- Swarm size: {stage2_swarmsize}")
        print(f"- Max iterations: {stage2_maxiter}")
        print(f"- Threshold: {stage2_threshold}")
        print("-" * 40)
    
    # Create reduced dataset with only stage 1 selected features
    print(f"Original feature set shape: {X.shape}")
    print(f"Selected features from stage 1: {len(stage1_features)}")

    max_idx = max(stage1_features)
    if max_idx >= X.shape[1]:
        print(f"Index {max_idx} is too large for {X.shape[1]} features")
    
    X_reduced = X.iloc[:, np.array(stage1_features)].values
    print(f"Reduced feature set shape: {X_reduced}")
    
    # Run PSO on reduced feature set
    stage2_weights, stage2_fitness, stage2_progress, stage2_features_reduced = run_pso_with_progress(
        X_reduced, Y, estimator, len(stage1_features),
        swarmsize=stage2_swarmsize,
        maxiter=stage2_maxiter,
        threshold=stage2_threshold
    )
    
    # Map stage 2 features back to original feature indices
    final_features = [stage1_features[i] for i in stage2_features_reduced]
    
    if verbose:
        print(f"\nSTAGE 2 RESULTS:")
        print(f"- Features selected: {len(final_features)}")
        print(f"- Best fitness: {stage2_fitness:.4f}")
        print(f"- Feature reduction: {len(stage1_features)} → {len(final_features)} "
              f"({len(final_features)/len(stage1_features)*100:.1f}%)")
        
        print(f"\nFINAL RESULTS:")
        print(f"- Total features: {n_features} → {len(final_features)} "
              f"({len(final_features)/n_features*100:.1f}%)")
        print(f"- Final fitness: {stage2_fitness:.4f}")
        print(f"- Total evaluations: {len(stage1_progress) + len(stage2_progress)}")
        print("="*60)
    
    # Return comprehensive results
    results = {
        'stage1_weights': stage1_weights,
        'stage1_fitness': stage1_fitness,
        'stage1_progress': stage1_progress,
        'stage1_features': stage1_features,
        'stage2_weights': stage2_weights,
        'stage2_fitness': stage2_fitness,
        'stage2_progress': stage2_progress,
        'final_features': final_features,
        'total_evaluations': len(stage1_progress) + len(stage2_progress)
    }
    
    return results


# Convenience function that mimics your original function interface
def run_two_stage_pso_simple(X, Y, estimator, n_features,
                             stage1_threshold=0.4, stage2_threshold=0.7):
    """
    Simplified two-stage PSO with default parameters
    Returns same format as original function for easy replacement
    """
    results = run_two_stage_pso(
        X, Y, estimator, n_features,
        stage1_threshold=stage1_threshold,
        stage2_threshold=stage2_threshold
    )
    
    # Return in same format as original function
    return (results['stage2_weights'], 
            results['stage2_fitness'], 
            results['stage1_progress'] + results['stage2_progress'],
            results['final_features'])



def run_ga_with_progress(X, Y, estimator, n_features,
                        pop_size=50, n_generations=100, threshold=0.7,
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