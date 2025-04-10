import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Example inscriptions for testing
test_inscriptions = [
    "αλεξανδρε ουδις βασιλευς",
    "αλεξανδρε ουδις μεγας",
    "ηρως αλεξανδρε ουδις",
    "αλεξανδρε ουδις ο φιλος",
    "αλεξανδρε ουδις σοφος",
    "αγαθος αλεξανδρε ουδις",
    "αλεξανδρε ουδις ο νικητης",
    "αλεξανδρε ουδις ο μεγας",
    "αλεξανδρε ουδις ισχυρος",
    "αλεξανδρε ουδις ο αγαθος"
]

# Convert test_inscriptions to DataFrame
df = pd.DataFrame({"text": test_inscriptions})

# Prepare the TF-IDF matrix
texts = df['text'].values
vectorizer = TfidfVectorizer(max_features=10)  # Limiting vocabulary to a small number
tfidf_matrix = vectorizer.fit_transform(texts)

# Get the vocabulary dictionary
vocab = vectorizer.vocabulary_

# Define the incomplete inscription
incomplete_inscription = "αλεξανδρε ουδις"

# Vectorize the incomplete inscription
incomplete_vector = vectorizer.transform([incomplete_inscription])

# Calculate cosine similarities with all inscriptions
similarities = cosine_similarity(incomplete_vector, tfidf_matrix)

# Select the top-10 closest inscriptions (in this case, it's all of them)
top_10_indices = np.argsort(similarities[0])[-10:]
top_10_vectors = tfidf_matrix[top_10_indices]

# Print the selected top-10 inscriptions
print("Top-10 inscriptions selected based on similarity:")
for idx in top_10_indices:
    print(f"- {texts[idx]}")

# Define Fitness and Individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Define the mutation operator and other functions
toolbox = base.Toolbox()
toolbox.register("attr_word", random.randint, 0, len(vocab) - 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_word, 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fitness Function: Calculate the similarity with the top-10 closest inscriptions
def fitness_function(individual):
    # Generate the completed inscription
    missing_words = [list(vocab.keys())[individual[0]], list(vocab.keys())[individual[1]]]
    completed_phrase = f"{missing_words[0]} αλεξανδρε ουδις {missing_words[1]}"
    
    # Vectorize the completed phrase
    completed_text_vector = vectorizer.transform([completed_phrase])
    
    # Calculate cosine similarities with the top-10 closest inscriptions
    similarities = cosine_similarity(completed_text_vector, top_10_vectors)
    
    # Compute fitness as the average of the top-10 similarities
    fitness_value = np.mean(similarities[0])
    
    print(f"Evaluating: {completed_phrase} | Fitness: {fitness_value}")
    return fitness_value,

# Registering crossover, mutation, and selection operators
toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxUniform, indpb=0.5) 
toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(vocab) - 1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Early stopping criterion function with elitism
def early_stopping_algorithm(population, toolbox, cxpb, mutpb, ngen, max_stagnation, elitism_size=1):
    best = None
    best_fitness = None
    no_improvement_counter = 0
    
    for gen in range(ngen):
        offspring = toolbox.select(population, len(population) - elitism_size)
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the offspring individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Add the best individuals from the previous generation (elitism)
        elite = tools.selBest(population, elitism_size)
        offspring.extend(elite)
        
        population[:] = offspring
        current_best = tools.selBest(population, 1)[0]
        
        if best is None or current_best.fitness.values[0] > best_fitness:
            best = current_best
            best_fitness = current_best.fitness.values[0]
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        # Early stopping if no improvement
        if no_improvement_counter >= max_stagnation:
            print(f"Terminating after {gen} generations due to lack of improvement")
            break
    
    return population, best

# Run the genetic algorithm with simplified parameters for testing
population_sizes = [10]  # Smaller population for testing
crossover_probs = [0.7]  # Only one setting for testing
mutation_probs = [0.1]
generations = 10  # Fewer generations for quicker testing
runs_per_setting = 2  # Fewer runs for quicker testing
max_stagnation = 5  # Early stopping after 5 generations with no improvement

# Prepare for plotting
for pop_size in population_sizes:
    for cx_prob in crossover_probs:
        for mut_prob in mutation_probs:
            all_runs_fitness_progress = []
            best_inscriptions = []
            
            for run in range(runs_per_setting):
                print(f"\nRun {run+1} with Population: {pop_size}, Crossover: {cx_prob}, Mutation: {mut_prob}")
                population = toolbox.population(n=pop_size)
                fitness_progress = []
                
                for gen in range(generations):
                    population, best_individual = early_stopping_algorithm(
                        population, toolbox, cxpb=cx_prob, mutpb=mut_prob, ngen=1, max_stagnation=max_stagnation
                    )
                    
                    # Track the best fitness value for this generation
                    fitness_progress.append(best_individual.fitness.values[0])
                    print(f"Generation {gen+1}: Best Fitness = {best_individual.fitness.values[0]}")
                
                # Record the fitness progress for this run
                all_runs_fitness_progress.append(fitness_progress)
                
                # Record the best inscription
                best_inscription = [list(vocab.keys())[idx] for idx in best_individual]
                best_inscriptions.append(best_inscription)
                print(f"Run {run+1}: Best Inscription: {' '.join(best_inscription)}")
            
            # Calculate the average fitness progress across all runs
            avg_fitness_progress = np.mean(np.array([np.pad(run, (0, generations - len(run)), 'edge')
                                                     for run in all_runs_fitness_progress]), axis=0)
            
            # Plot the evolution of the average fitness across generations
            plt.plot(avg_fitness_progress, label=f"Pop: {pop_size}, Cx: {cx_prob}, Mut: {mut_prob}")
            
            # Print final results
            print(f"\nResults for Population: {pop_size}, Crossover: {cx_prob}, Mutation: {mut_prob}")
            print(f"Average Best Fitness: {np.mean(avg_fitness_progress)}")
            print(f"Maximum Best Fitness: {np.max(avg_fitness_progress)}")
            print(f"Best Inscription: {' '.join(best_inscriptions[np.argmax([np.max(run) for run in all_runs_fitness_progress])])}\n")

plt.xlabel('Generations')
plt.ylabel('Average Fitness Across Runs')
plt.title('Fitness Evolution Across Generations')
plt.legend()
plt.show()
