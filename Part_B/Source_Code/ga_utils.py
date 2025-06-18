"""Module for Alzheimer's disease prediction and analysis using neural networks 
finding the best features using Genetic Algorithm (GA).

this is hte utils for the GA operations and the experiment configurations.
now lets see what we have here:
initialize_populationGA: creates the initial population of binary vectors
evaluate_individual_features: evaluates an individual based on model performance and loss
tournament_selection: selects individuals using tournament selection
uniform_crossover: performs uniform crossover between two parents
metalaxis: performs mutation on an individual
genenetikos_main: main function to run the genetic algorithm


Author: Orestis Antonis Makris AM 1084516
Date: 2025-4-21
License: MIT
University of Patras, Department of Computer Engineering and Informatics
This code is part of a project for the course "Computational Inteligence".
"""


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import ColumnTransformer

from configGA import *


def initialize_populationGA(pop_size, num_features):      

    """ Δημιουργεί αρχικό πληθυσμό: τυχαία δυαδικa διανύσματα, 
        με τουλάχιστον ένα bit = 1 μασκα για για nn input features."""
    
    population = []

    for _ in range(pop_size):

        individual = np.random.randint(0, 2, size=num_features)

          # 

        if np.sum(individual) == 0:

            individual[np.random.randint(0, num_features)] = 1

        population.append(individual)

    return np.array(population)
       

def evaluate_individual_features(indiv, model, x_val_full, y_val, device=torch.device('cuda')):

    """Αξιολογεί ένα άτομο (indiv) του πληθυσμού με βάση την ακρίβεια του μοντέλου
       και την απώλiο validation set. Επιστρέφει fitness, loss, accuracy και αριθμό επιλεγμένων χαρακτηριστικών."""
    
    #δημιουργία μάσκας για τα επιλεγμένα χαρακτηριστικά τα οπια τα φινταρουμε στο μοντέλο μας

    mask = torch.tensor(indiv, dtype=torch.bool)
    x_masked = x_val_full[:, mask]  
#
    x_input = x_val_full.clone() 

    x_input[:, ~mask] = 0.0 # τωρα τα μη επιλεγμένα χαρακτηριστικά θα τα κανουμε  ζερο

    model.to(device).eval()

    with torch.no_grad():
        
        out = model(x_input.to(device))

        criterion = nn.CrossEntropyLoss()

        loss = criterion(out, y_val.to(device)).item()

        preds = out.argmax(dim=1).cpu()

        accuracy = (preds == y_val).float().mean().item()

    # για την καταληλοτηρα του ατωμου, θα προσθεσουμε ένα ποινή για τα μη επιλεγμένα χαρακτηριστικα

    num_selected = indiv.sum()

    penalty = PENALTY_WEIGHT * (num_selected / NUM_FEATURES)
    fitness = loss + penalty

    return fitness, loss, accuracy, num_selected

               
                     

def tournament_selection(population, fitnesses, k=TOURNAMENTSIZE):

    """Επιλέγάτομα από τον πληθυσμό με βάση την τουρνουά επιλογή.
       Κάνει τυχαία επιλογή k ατόμων και επιστρέφει το καλύτερο από αυτά."""  
    
    select = []
    len_population = len(population)

    for _ in range(len_population):

        tournament_indices = random.sample(range(len_population), k)
        

        best_idx = tournament_indices[0]

        best_fitness = fitnesses[best_idx]
        
        for idx in tournament_indices[1:]:

            if fitnesses[idx] < best_fitness:
                best_fitness = fitnesses[idx]
                best_idx = idx
        
        select.append(population[best_idx].copy())
    
    return select

def uniform_crossover(goeneas1, goeneas2):

    """Εκτελείd ομοιόμορφο crossover μεταξi των δύο γονεων (goeneas1, goeneas2).
       Επιστρέφει δύο απογόνους με τυχαία ανταλλαγή χαρακτηριστικών."""
    
    mask = np.random.randint(0, 2, size=goeneas1.shape).astype(bool)

    c1 = goeneas1.copy()
    c2 = goeneas2.copy()

    c1[mask] = goeneas2[mask]
    c2[mask] = goeneas1[mask]

    if c1.sum() == 0:

        idx = np.random.randint(0, NUM_FEATURES)
        c1[idx] = 1

    if c2.sum() == 0:
        idx = np.random.randint(0, NUM_FEATURES)

        c2[idx] = 1

    return c1, c2



def metalaxis(indiv, mutation_prob=MUTATION_PROB):

    for i in range(len(indiv)):

        if random.random() < mutation_prob:
            # Αν η πιθανότητα μετάλλαξης συμβεί, αντιστρέφουμε το bitd
            indiv[i] = 1 - indiv[i]

    if indiv.sum() == 0:

        idx = np.random.randint(0, NUM_FEATURES)

        indiv[idx] = 1
    return indiv

def genenetikos_main(model, x_val_full, y_val, device, num_features,
                     population_size=POPULATION_SIZE, 
                     max_generations=MAX_GENERATIONS,
                     crossover_prob=CROSSOVER_PROB, 
                     mutation_prob=MUTATION_PROB, 
                     tournamentsize=TOURNAMENTSIZE,
                     elitism=ELITISM, 
                     penalty_weight=PENALTY_WEIGHT, 
                     patience=PATIENCE):

    population = initialize_populationGA(population_size, num_features)

    #print("Initial population:", population)
    best_solution = None        

    best_fitness = float('inf')             
    best_loss = float('inf')

    best_accuracy = 0.0  
    best_num_selected = 0
    
    history = []

    for generation in range(max_generations):

        fitnesses = []

        for individual in population:
            fitness, loss, acc, num_feat = evaluate_individual_features(individual, model, x_val_full, y_val, device)
            fitnesses.append(fitness)


        gene__best_idx = np.argmin(fitnesses)

        gene_best_fitness = fitnesses[gene__best_idx]

        genaration_best = population[gene__best_idx]

        history.append(gene_best_fitness)

        if gene_best_fitness < best_fitness:

            best_fitness = gene_best_fitness
            best_solution = genaration_best.copy()

        #epologi
        selected = tournament_selection(population, fitnesses)

        # Crossover
        paidia = []
        for i in range(0, population_size, 2):

            goneas1 = selected[i]
            goneas2 = selected[i + 1] if i + 1 < population_size else selected[0]

            if random.random() < crossover_prob:

                c1, c2 = uniform_crossover(goneas1, goneas2)

            else:


                c1, c2 = goneas1.copy(), goneas2.copy()

            paidia.append(c1)

            paidia.append(c2)

        # Μετάλλαξη
        neos_plithismos = []

        for indiv in paidia:

            metalaxisss = metalaxis(indiv)
            neos_plithismos.append(metalaxisss)

        #tora elitimsow
        if elitism:
            new_fitnesss = []
            for indiv in neos_plithismos:

                fitnesss, losss, acc, num_feat = evaluate_individual_features(indiv, model, x_val_full, y_val, device)
                new_fitnesss.append(fitnesss)

            
            kakotero_idx = np.argmax(new_fitnesss)

            neos_plithismos[kakotero_idx] = best_solution
        
        population = np.array(neos_plithismos)

        if generation > 10 and len(history) > 1 and abs(history[-2] - history[-1]) < patience:

            print(f"Convergence reached at generation {generation}.")

            break


    return best_solution, best_fitness, history, len(history)