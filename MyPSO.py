from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from tqdm import tqdm

from models.MyXGboost import XGBoostMultiClass


class Particle:
    def __init__(self, n_features, init_prob=0.5):
        # Binary position (which features are selected)
        self.position = np.random.random(n_features) < init_prob
        # Velocity determines the probability of bit flipping
        self.velocity = np.random.uniform(-1, 1, n_features)
        # Personal best position and fitness
        self.pbest_position = self.position.copy()
        self.pbest_fitness = float('-inf')
    
    def update_velocity(self, gbest_position, w=0.7, c1=1.5, c2=1.5):
        """Update the velocity of the particle"""
        inertia = w * self.velocity
        r1 = np.random.random(len(self.position))
        cognitive = c1 * r1 * (self.pbest_position.astype(int) - self.position.astype(int))
        r2 = np.random.random(len(self.position))
        social = c2 * r2 * (gbest_position.astype(int) - self.position.astype(int))
        self.velocity = inertia + cognitive + social
        self.velocity = np.clip(self.velocity, -4, 4)
    
    def update_position(self):
        """Update the position based on velocity using sigmoid function"""
        sigmoid = 1 / (1 + np.exp(-self.velocity))
        self.position = np.random.random(len(self.position)) < sigmoid
        
    def evaluate_fitness(self, X, y, min_features=1):
        """Evaluate the fitness of the particle using XGBoost"""
        # Ensure at least min_features are selected
        if np.sum(self.position) < min_features:
            idx = np.random.choice(len(self.position), min_features, replace=False)
            temp_position = np.zeros_like(self.position)
            temp_position[idx] = 1
            selected_features = temp_position.astype(bool)
        else:
            selected_features = self.position.astype(bool)
        
        n_selected = np.sum(selected_features)
        if n_selected < min_features or n_selected == len(self.position):
            return float('-inf')
        
        # Select features based on position
        X_selected = X.iloc[:, selected_features]
        values_selected = X_selected.values

        # Use XGBoost classifier with cross-validation
        clf = XGBoostMultiClass()
        # clf.fit(values_selected, y)
        # print(clf.score(values_selected, y))


        cv_score = cross_val_score(clf, values_selected, y, cv=5, scoring='accuracy')
        
        # Calculate fitness (mean AUC - penalty for too many features)
        mean_auc = np.mean(cv_score)
        penalty = 0.001 * n_selected / len(self.position)
        fitness = mean_auc - penalty
        
        # Update personal best if current fitness is better
        if fitness > self.pbest_fitness:
            self.pbest_fitness = fitness
            self.pbest_position = self.position.copy()
            
        return fitness

class MyPSO:
    def __init__(self, n_particles, n_features, max_iter=100, w=0.7, c1=1.5, c2=1.5, init_prob=0.5, min_features=5):
        self.n_particles = n_particles
        self.n_features = n_features
        self.max_iter = max_iter
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.min_features = min_features
        
        # Initialize particles
        self.particles = [Particle(n_features, init_prob) for _ in range(n_particles)]
        
        # Global best position and fitness
        self.gbest_position = np.random.random(n_features) < init_prob
        self.gbest_fitness = float('-inf')
        
        # History for plotting
        self.fitness_history = []
        self.feature_count_history = []
        
    def optimize(self, X, y):
        """Execute the PSO optimization process"""
        for iteration in tqdm(range(self.max_iter)):
            # Evaluate fitness for each particle
            for particle in self.particles:
                fitness = particle.evaluate_fitness(X, y, self.min_features)
                
                # Update global best if current fitness is better
                if fitness > self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = particle.position.copy()
            
            # Update velocity and position for each particle
            for particle in self.particles:
                particle.update_velocity(self.gbest_position, self.w, self.c1, self.c2)
                particle.update_position()
            
            # Store history
            self.fitness_history.append(self.gbest_fitness)
            self.feature_count_history.append(np.sum(self.gbest_position))
            
            # Print progress occasionally
            if (iteration + 1) % 10 == 0:
                selected_count = np.sum(self.gbest_position)
                print(f"Iteration {iteration+1}/{self.max_iter}: Best fitness = {self.gbest_fitness:.4f}, "
                      f"Selected features = {selected_count}/{self.n_features}")
        
        print(f"PSO completed. Best fitness: {self.gbest_fitness:.4f}")
        print(f"Number of selected features: {np.sum(self.gbest_position)}/{self.n_features}")
        
        return self.gbest_position, self.gbest_fitness
    
    def plot_progress(self):
        """Plot the optimization progress"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(self.fitness_history)
        ax1.set_title('Fitness History')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fitness Value')
        
        ax2.plot(self.feature_count_history)
        ax2.set_title('Selected Feature Count History')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Number of Features')
        
        plt.tight_layout()
        plt.show()
        
    def evaluate_final_model(self, X, y, feature_names=None, test_size=0.2):
        """Evaluate the final model with the selected features"""
        selected_features = self.gbest_position.astype(bool)
        X_selected = X[:, selected_features]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=42)
        
        # Train the final model
        clf = XGBoostMultiClass()
        clf.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr', average='weighted')
        
        print(f"Final model evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        # Print selected features if names are provided
        if feature_names is not None:
            selected_indices = np.where(selected_features)[0]
            selected_names = feature_names[selected_indices]
            print("\nSelected features:")
            for i, feature in enumerate(selected_names):
                print(f"{i+1}. {feature}")
        
        return clf, (accuracy, f1, auc)