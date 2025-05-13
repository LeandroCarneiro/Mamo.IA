import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import xgboost as xgb
import random
from tqdm import tqdm

import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, X, y, population_size=50, generations=100, crossover_rate=0.8, 
                 mutation_rate=0.1, tournament_size=3):
        """
        Genetic algorithm for feature selection with XGBoost classifier
        """
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        
        # Tracking
        self.best_individual = None
        self.best_fitness = 0
        self.fitness_history = []
        self.feature_count_history = []
        
    def initialize_population(self):
        """Create initial population of binary chromosomes"""
        population = []
        for _ in range(self.population_size):
            # Random binary string (1=feature selected, 0=not selected)
            individual = np.random.randint(0, 2, self.n_features)
            # Ensure at least one feature is selected
            if np.sum(individual) == 0:
                individual[np.random.randint(0, self.n_features)] = 1
            population.append(individual)
        return population
    
    def fitness_function(self, individual):
        """Evaluate fitness using XGBoost classifier accuracy"""
        selected_features = np.where(individual == 1)[0]
        
        if len(selected_features) == 0:
            return 0
        
        X_subset = self.X[:, selected_features]
        
        # Cross-validation for more robust evaluation
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        cv_scores = cross_val_score(model, X_subset, self.y, cv=5, scoring='accuracy')
        accuracy = np.mean(cv_scores)
        
        # Small penalty for using too many features
        penalty = 0.0005 * len(selected_features) / self.n_features
        fitness = accuracy - penalty
        
        return fitness
    
    def tournament_selection(self, population, fitnesses):
        """Select individual using tournament selection"""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx]
    
    def crossover(self, parent1, parent2):
        """Perform single-point crossover"""
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutation(self, individual):
        """Mutate by flipping bits"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip bit
        
        # Ensure at least one feature is selected
        if np.sum(mutated) == 0:
            mutated[random.randint(0, len(mutated) - 1)] = 1
            
        return mutated
    
    def run(self):
        """Run the GA for specified number of generations"""
        # Initialize population
        population = self.initialize_population()
        
        # Main GA loop
        for generation in tqdm(range(self.generations)):
            # Evaluate fitness
            fitnesses = [self.fitness_function(ind) for ind in population]
            
            # Update best solution
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_individual = population[best_idx].copy()
            
            # Record history
            self.fitness_history.append(self.best_fitness)
            self.feature_count_history.append(np.sum(self.best_individual))
            
            # Create new population
            new_population = []
            
            # Elitism - keep the best individual
            new_population.append(population[best_idx].copy())
            
            # Generate the rest of the new population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                # Crossover and mutation
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                # Add to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Replace old population
            population = new_population
        
        return self.best_individual, self.best_fitness
    
    def plot_progress(self):
        """Plot fitness and feature count over generations"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(self.fitness_history)
        ax1.set_title('Best Fitness over Generations')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness (Accuracy)')
        
        ax2.plot(self.feature_count_history)
        ax2.set_title('Number of Selected Features over Generations')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Feature Count')
        
        plt.tight_layout()
        plt.show()
        
    def get_selected_features(self):
        """Return indices of selected features"""
        if self.best_individual is not None:
            return np.where(self.best_individual == 1)[0]
        return []

def main():
    # Human Activity Recognition dataset with 561 features
    print("Loading Human Activity Recognition dataset...")
    har = fetch_openml('har', version=1, as_frame=True)
    X = har.data.values
    y = har.target
    feature_names = np.array(har.feature_names)
    
    # Run the genetic algorithm
    ga = GeneticAlgorithm(X, y, population_size=50, generations=100)
    best_solution, best_fitness = ga.run()
    
    # Get the selected features
    selected_features = ga.get_selected_features()
    
    print(f"\nGA completed.")
    print(f"Best fitness (accuracy): {best_fitness:.4f}")
    print(f"Number of selected features: {len(selected_features)}")
    
    # Plot the progress
    ga.plot_progress()
    
    # Final evaluation with best feature subset
    X_selected = X.iloc[:, selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nFinal test accuracy with selected features: {final_accuracy:.4f}")

if __name__ == "__main__":
    main()