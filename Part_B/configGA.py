"""Module for Alzheimer's disease prediction and analysis using neural networks 
finding the best features using Genetic Algorithm (GA).
thiis contains the configuration for the GA parameters and the experiment configurations.

Author: Orestis Antonis Makris AM 1084516
Date: 2025-5-12
License: MIT
University of Patras, Department of Computer Engineering and Informatics
This code is part of a project for the course "Computational Inteligence".
"""

POPULATION_SIZE = 200
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.01

MAX_GENERATIONS = 1000
PATIENCE =0.0001
NUM_RUNS = 20

TOURNAMENTSIZE = 4

PENALTY_WEIGHT = 0.06

ELITISM = True
NUM_FEATURES = 34



# αυτο για τα πειράματα που θα τρέξουμε
EXPERIMENT_CONFIGS = [
    {'id': 1, 'pop': 20, 'crosspro': 0.6, 'mutpro': 0.00},
    {'id': 2, 'pop': 20, 'crosspro': 0.6, 'mutpro': 0.01},
    {'id': 3, 'pop': 20, 'crosspro': 0.6, 'mutpro': 0.10},
    {'id': 4, 'pop': 20, 'crosspro': 0.9, 'mutpro': 0.01},
    {'id': 5, 'pop': 20, 'crosspro': 0.1, 'mutpro': 0.01},
    {'id': 6, 'pop': 200, 'crosspro': 0.6, 'mutpro': 0.00},
    {'id': 7, 'pop': 200, 'crosspro': 0.6, 'mutpro': 0.01},
    {'id': 8, 'pop': 200, 'crosspro': 0.6, 'mutpro': 0.10},
    {'id': 9, 'pop': 200, 'crosspro': 0.9, 'mutpro': 0.01},
    {'id': 10, 'pop': 200, 'crosspro': 0.1, 'mutpro': 0.01},
]